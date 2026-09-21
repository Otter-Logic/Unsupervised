using OtterLogic.Graphs;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Fuses several clusterings of the same samples into one grouping they
/// collectively support, and says how firmly they supported it.
/// <para>
/// Each method here assumes something different about what a cluster is, and on
/// real data each is right about some of it. Choosing one — what
/// <see cref="ClusterSelector"/> does — throws the others' evidence away. This
/// keeps all of it: two samples belong together in proportion to how many views,
/// by weight, put them together. That is evidence accumulation (Fred and Jain,
/// 2005) — a co-association between every pair, then a hierarchy over it.
/// </para>
/// <list type="number">
/// <item><b>Weight.</b> Each view's vote, scaled by how far the others agree with
/// it (see <see cref="ConsensusOptions.WeightByAgreement"/>).</item>
/// <item><b>Co-associate.</b> For a pair, the weighted share of the views placing
/// both that put them in one cluster. A view that left either sample unplaced
/// abstains rather than voting them apart — declining to place a sample is not a
/// claim about its company.</item>
/// <item><b>Build the tree.</b> Average linkage over one minus that share. Average
/// because it is the linkage whose height means the thing voted on: cut at h, and
/// on average across any two merged groups at most a share h of the vote was
/// against it.</item>
/// <item><b>Choose the count</b> with the longest lifetime — the widest range of
/// heights over which cutting the tree gives that many groups, Fred and Jain's own
/// rule. A silhouette of each cut is reported beside it but decides nothing: samples
/// sharing a label in every view sit at zero disagreement from each other, so a
/// silhouette on this distance scores every such signature as a perfect group of its
/// own and always asks for the most groups allowed.</item>
/// <item><b>Merge small groups</b> into the group their samples agree with most,
/// connected to them when a graph is given.</item>
/// </list>
/// <para>
/// Samples with the same label in every view are indistinguishable to all of the
/// above, so the fusion runs on those distinct signatures, weighted by how many
/// samples share each, rather than on every sample. A few clusterings of ten
/// groups or fewer produce a few hundred signatures however many thousand samples
/// there are, which is what makes an exact average-linkage tree affordable.
/// </para>
/// </summary>
public static class ConsensusClustering
{
    /// <summary>
    /// Least share of its weight a view keeps however far the others disagree
    /// with it — outvoted, never silenced.
    /// </summary>
    private const double AgreementFloor = 0.1;

    /// <summary>
    /// Most distinct label signatures fused exactly. The tree holds a square matrix
    /// over them, and this is about a hundred and thirty megabytes of it; past it
    /// the views are fragmenting the samples so finely that there is little
    /// consensus to find.
    /// </summary>
    private const int MaximumSignatures = 4096;

    /// <summary>
    /// Fuses the views into one grouping.
    /// </summary>
    /// <param name="views">At least one labelling, every one over the same samples in the same order.</param>
    /// <param name="connectivity">
    /// Optional. Which samples are related; when given, a small group is only ever
    /// merged into a group it is connected to.
    /// </param>
    /// <param name="options">Settings; null for the defaults.</param>
    public static ConsensusResult Fuse(
        IReadOnlyList<ClusterView> views, WeightedGraph? connectivity = null, ConsensusOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(views);
        options ??= new ConsensusOptions();

        int n = Validate(views, connectivity);
        options.Validate(n);

        var weights = Weights(views, options.WeightByAgreement);
        var (signatureOf, size, signatures) = Signatures(views, weights);
        int m = size.Length;

        if (m > MaximumSignatures)
            throw new ArgumentException(
                $"The views split the samples into {m} distinct combinations of labels, more than the {MaximumSignatures} "
                + "an exact fusion holds in memory. Fuse fewer views, or views with fewer groups.", nameof(views));

        var distance = Disagreement(signatures, weights);
        var merges = AverageLinkage(distance, size);

        var sweep = new List<ConsensusCandidate>();
        int[] cut;

        if (m == 1)
        {
            cut = new int[1];
            sweep.Add(new ConsensusCandidate(1, 1.0, 0.0));
        }
        else if (options.Groups is { } fixedCount)
        {
            int k = Math.Min(fixedCount, m);
            cut = Cut(merges, m, k);
            sweep.Add(new ConsensusCandidate(k, Lifetime(merges, m, k), Silhouette(distance, size, cut)));
        }
        else
        {
            int highest = Math.Min(options.MaximumGroups, m);
            int lowest = Math.Min(options.MinimumGroups, highest);

            cut = Cut(merges, m, lowest);
            double longest = double.NegativeInfinity;

            for (int k = lowest; k <= highest; k++)
            {
                var candidate = Cut(merges, m, k);
                double lifetime = Lifetime(merges, m, k);
                sweep.Add(new ConsensusCandidate(k, lifetime, Silhouette(distance, size, candidate)));

                // Strictly longer, so a tie goes to the fewer groups found first.
                if (lifetime > longest + 1e-12)
                {
                    longest = lifetime;
                    cut = candidate;
                }
            }
        }

        var labels = ClusterLabels.Canonical(signatureOf.Select(s => cut[s]).ToArray());
        int merged = MergeSmall(labels, signatureOf, distance, size, connectivity, options.MinimumGroupSize);
        labels = ClusterLabels.Canonical(labels);

        int groups = labels.Max() + 1;
        var groupOfSignature = new int[m];
        for (int i = 0; i < n; i++)
            groupOfSignature[signatureOf[i]] = labels[i];

        var perSignature = Agreement(distance, size, groupOfSignature, groups);
        double silhouette = groups < 2 ? 0.0 : Silhouette(distance, size, groupOfSignature);

        double total = weights.Sum();
        return new ConsensusResult(
            labels,
            groups,
            signatureOf.Select(s => perSignature[s]).ToArray(),
            views,
            weights.Select(w => w / total).ToArray(),
            views.Select(v => ClusterAgreement.AdjustedRand(v.Labels, labels)).ToArray(),
            sweep,
            silhouette,
            merged,
            m);
    }

    private static int Validate(IReadOnlyList<ClusterView> views, WeightedGraph? connectivity)
    {
        if (views.Count == 0)
            throw new ArgumentException("Need at least one view to fuse.", nameof(views));

        for (int v = 0; v < views.Count; v++)
        {
            if (views[v] is null || views[v].Labels is null)
                throw new ArgumentNullException(nameof(views), $"View {v} has no labels.");
            if (!double.IsFinite(views[v].Weight) || views[v].Weight < 0.0)
                throw new ArgumentOutOfRangeException(nameof(views), views[v].Weight,
                    $"View \"{views[v].Name}\" has weight {views[v].Weight}; weights must be finite and not negative.");
        }

        int n = views[0].Labels.Length;
        foreach (var view in views)
            if (view.Labels.Length != n)
                throw new ArgumentException(
                    $"View \"{views[0].Name}\" labels {n} samples but \"{view.Name}\" labels {view.Labels.Length}. "
                    + "Every view must label the same samples.", nameof(views));

        if (n < 2)
            throw new ArgumentException("Need at least two samples to group.", nameof(views));
        if (views.All(v => v.Weight == 0.0))
            throw new ArgumentException("Every view has weight zero, so there is no vote to count.", nameof(views));
        if (connectivity is not null && connectivity.NodeCount != n)
            throw new ArgumentException(
                $"The graph has {connectivity.NodeCount} nodes but the views label {n} samples.", nameof(connectivity));

        return n;
    }

    /// <summary>Each view's weight, scaled by its mean adjusted Rand index against the other weighted views.</summary>
    private static double[] Weights(IReadOnlyList<ClusterView> views, bool byAgreement)
    {
        var weights = views.Select(v => v.Weight).ToArray();
        var voting = Enumerable.Range(0, views.Count).Where(v => weights[v] > 0.0).ToArray();

        if (!byAgreement || voting.Length < 2)
            return weights;

        var scaled = (double[])weights.Clone();
        foreach (int v in voting)
        {
            double agreement = voting.Where(u => u != v)
                .Average(u => Math.Max(0.0, ClusterAgreement.AdjustedRand(views[v].Labels, views[u].Labels)));
            scaled[v] = weights[v] * (AgreementFloor + (1.0 - AgreementFloor) * agreement);
        }

        return scaled;
    }

    /// <summary>
    /// Every distinct combination of labels across the weighted views, how many
    /// samples share each, and which one each sample has. A sample every view left
    /// unplaced is a signature of its own: nothing says it is like anything else,
    /// including another sample nothing placed.
    /// </summary>
    private static (int[] SignatureOf, double[] Size, int[][] Signatures) Signatures(
        IReadOnlyList<ClusterView> views, double[] weights)
    {
        int n = views[0].Labels.Length;
        var voting = Enumerable.Range(0, views.Count).Where(v => weights[v] > 0.0).ToArray();

        var index = new Dictionary<int[], int>(new SignatureComparer());
        var signatures = new List<int[]>();
        var size = new List<double>();
        var signatureOf = new int[n];

        for (int i = 0; i < n; i++)
        {
            // Full length with -1 for views that do not vote, so positions line up
            // with the weights.
            var signature = new int[views.Count];
            Array.Fill(signature, -1);
            foreach (int v in voting)
                signature[v] = Math.Max(views[v].Labels[i], -1);

            bool placedAnywhere = signature.Any(label => label >= 0);
            if (placedAnywhere && index.TryGetValue(signature, out int existing))
            {
                signatureOf[i] = existing;
                size[existing]++;
                continue;
            }

            signatureOf[i] = signatures.Count;
            if (placedAnywhere)
                index[signature] = signatures.Count;

            signatures.Add(signature);
            size.Add(1.0);
        }

        return (signatureOf, size.ToArray(), signatures.ToArray());
    }

    /// <summary>
    /// One minus the co-association between every pair of signatures: the weighted
    /// share of views placing both that put them apart. A pair no view places
    /// together or apart has no evidence either way and reads as fully apart,
    /// because being grouped needs a reason.
    /// </summary>
    private static double[,] Disagreement(int[][] signatures, double[] weights)
    {
        int m = signatures.Length;
        var distance = new double[m, m];

        for (int a = 0; a < m; a++)
        {
            for (int b = a + 1; b < m; b++)
            {
                double together = 0.0;
                double voted = 0.0;
                for (int v = 0; v < weights.Length; v++)
                {
                    int la = signatures[a][v];
                    int lb = signatures[b][v];
                    if (la < 0 || lb < 0)
                        continue;

                    voted += weights[v];
                    if (la == lb)
                        together += weights[v];
                }

                double value = voted > 0.0 ? 1.0 - together / voted : 1.0;
                distance[a, b] = value;
                distance[b, a] = value;
            }
        }

        return distance;
    }

    /// <summary>
    /// Average linkage by the nearest-neighbour chain, each signature weighted by
    /// the samples sharing it, so the tree is the one average linkage would build
    /// over every sample. Merges come back sorted by height, each naming a
    /// signature from either side.
    /// </summary>
    private static (int A, int B, double Height)[] AverageLinkage(double[,] source, double[] weight)
    {
        int m = weight.Length;
        var distance = (double[,])source.Clone();
        var size = (double[])weight.Clone();
        var active = Enumerable.Repeat(true, m).ToArray();
        var chain = new int[m];
        int length = 0;
        var merges = new List<(int A, int B, double Height, int Order)>(Math.Max(m - 1, 0));

        for (int step = 0; step < m - 1; step++)
        {
            if (length == 0)
            {
                chain[0] = Array.IndexOf(active, true);
                length = 1;
            }

            int a, b;
            double nearest;

            while (true)
            {
                a = chain[length - 1];

                // Prefer the previous link on a tie, or two equidistant clusters
                // hand the chain back and forth forever.
                if (length > 1)
                {
                    b = chain[length - 2];
                    nearest = distance[a, b];
                }
                else
                {
                    b = -1;
                    nearest = double.PositiveInfinity;
                }

                for (int i = 0; i < m; i++)
                {
                    if (!active[i] || i == a)
                        continue;

                    if (distance[a, i] < nearest)
                    {
                        nearest = distance[a, i];
                        b = i;
                    }
                }

                if (length > 1 && b == chain[length - 2])
                    break;

                chain[length++] = b;
            }

            length -= 2;

            int keep = Math.Max(a, b);
            int retire = Math.Min(a, b);
            merges.Add((retire, keep, nearest, step));

            for (int i = 0; i < m; i++)
            {
                if (!active[i] || i == a || i == b)
                    continue;

                double updated = (size[a] * distance[i, a] + size[b] * distance[i, b]) / (size[a] + size[b]);
                distance[i, keep] = updated;
                distance[keep, i] = updated;
            }

            size[keep] = size[a] + size[b];
            active[retire] = false;
        }

        return merges.OrderBy(merge => merge.Height).ThenBy(merge => merge.Order)
            .Select(merge => (merge.A, merge.B, merge.Height)).ToArray();
    }

    /// <summary>
    /// The tree cut into <paramref name="k"/> groups: the lowest m − k merges applied,
    /// groups numbered by their lowest signature. Average linkage never merges
    /// below an earlier merge, so applying the lowest merges in any order draws the
    /// same cut.
    /// </summary>
    private static int[] Cut((int A, int B, double Height)[] merges, int m, int k)
    {
        var parent = Enumerable.Range(0, m).ToArray();
        int Find(int x)
        {
            while (parent[x] != x)
                x = parent[x] = parent[parent[x]];
            return x;
        }

        for (int t = 0; t < m - k; t++)
        {
            int ra = Find(merges[t].A);
            int rb = Find(merges[t].B);
            if (ra != rb)
                parent[Math.Max(ra, rb)] = Math.Min(ra, rb);
        }

        var number = new Dictionary<int, int>();
        var labels = new int[m];
        for (int s = 0; s < m; s++)
        {
            int root = Find(s);
            if (!number.TryGetValue(root, out int label))
                number[root] = label = number.Count;
            labels[s] = label;
        }

        return labels;
    }

    /// <summary>
    /// How long the cut into <paramref name="k"/> groups lasts: the height of the
    /// merge that would take it to k − 1, less the height of the merge that made
    /// it. Disagreement never exceeds one, so one group lasts up to one, and a
    /// group per signature starts from zero.
    /// </summary>
    private static double Lifetime((int A, int B, double Height)[] merges, int m, int k)
    {
        double upper = k > 1 ? merges[m - k].Height : 1.0;
        double lower = k < m ? merges[m - k - 1].Height : 0.0;
        return upper - lower;
    }

    /// <summary>
    /// Mean silhouette over every sample, from signature-level labels, each
    /// signature standing for the samples sharing it. Zero when fewer than two
    /// groups hold anything; a sample alone in its group scores zero, as
    /// <see cref="ClusterQuality.Silhouette"/> does.
    /// </summary>
    private static double Silhouette(double[,] distance, double[] size, int[] labels)
    {
        int m = size.Length;
        int k = labels.Max() + 1;
        if (k < 2)
            return 0.0;

        var groupSize = new double[k];
        for (int s = 0; s < m; s++)
            groupSize[labels[s]] += size[s];

        var sums = new double[k];
        double total = 0.0;

        for (int a = 0; a < m; a++)
        {
            Array.Clear(sums);
            for (int b = 0; b < m; b++)
                sums[labels[b]] += size[b] * distance[a, b];

            int own = labels[a];
            if (groupSize[own] <= 1.0)
                continue;

            double inside = sums[own] / (groupSize[own] - 1.0);
            double outside = double.PositiveInfinity;
            for (int c = 0; c < k; c++)
                if (c != own && groupSize[c] > 0.0)
                    outside = Math.Min(outside, sums[c] / groupSize[c]);

            double divisor = Math.Max(inside, outside);
            if (divisor > 0.0 && double.IsFinite(outside))
                total += size[a] * (outside - inside) / divisor;
        }

        return total / size.Sum();
    }

    /// <summary>
    /// Repeatedly merges the smallest group under <paramref name="minimumSize"/> into
    /// the group its samples agree with most — among the groups it is connected
    /// to, when a graph says which those are and there are any. Ties go to the
    /// lower-numbered group. Returns how many groups were merged away.
    /// </summary>
    private static int MergeSmall(
        int[] labels, int[] signatureOf, double[,] distance, double[] size, WeightedGraph? connectivity, int minimumSize)
    {
        int merged = 0;

        while (true)
        {
            int groups = labels.Max() + 1;
            if (groups < 2)
                return merged;

            var members = ClusterLabels.Members(labels, groups);
            int small = Enumerable.Range(0, groups)
                .Where(g => members[g].Length > 0 && members[g].Length < minimumSize)
                .OrderBy(g => members[g].Length).ThenBy(g => g)
                .DefaultIfEmpty(-1).First();

            if (small < 0)
                return merged;

            var candidates = new SortedSet<int>();
            if (connectivity is not null)
                foreach (int i in members[small])
                    foreach (int j in connectivity.Neighbours(i))
                        if (labels[j] != small)
                            candidates.Add(labels[j]);

            if (candidates.Count == 0)
                for (int g = 0; g < groups; g++)
                    if (g != small && members[g].Length > 0)
                        candidates.Add(g);

            if (candidates.Count == 0)
                return merged;

            // Mean co-association between the two groups' samples, read through
            // their signatures.
            var ownSignatures = Tally(members[small], signatureOf);
            int target = -1;
            double best = double.NegativeInfinity;

            foreach (int g in candidates)
            {
                double sum = 0.0;
                foreach (var (sb, countB) in Tally(members[g], signatureOf))
                    foreach (var (sa, countA) in ownSignatures)
                        sum += countA * countB * (1.0 - distance[sa, sb]);

                double mean = sum / ((double)members[small].Length * members[g].Length);
                if (mean > best + 1e-12)
                {
                    best = mean;
                    target = g;
                }
            }

            foreach (int i in members[small])
                labels[i] = target;

            // Close the gap the merged group left, so the labels stay 0..groups-2.
            for (int i = 0; i < labels.Length; i++)
                if (labels[i] > small)
                    labels[i]--;

            merged++;
        }
    }

    private static Dictionary<int, int> Tally(int[] samples, int[] signatureOf)
    {
        var tally = new Dictionary<int, int>();
        foreach (int i in samples)
            tally[signatureOf[i]] = tally.TryGetValue(signatureOf[i], out int count) ? count + 1 : 1;
        return tally;
    }

    /// <summary>
    /// Per signature, the mean co-association of one of its samples with every
    /// other sample in its group — samples of the same signature count fully, and
    /// the sample itself not at all. One for a sample alone in its group.
    /// </summary>
    private static double[] Agreement(double[,] distance, double[] size, int[] groupOf, int groups)
    {
        int m = size.Length;
        var groupSize = new double[groups];
        for (int s = 0; s < m; s++)
            groupSize[groupOf[s]] += size[s];

        var agreement = new double[m];
        for (int a = 0; a < m; a++)
        {
            int g = groupOf[a];
            if (groupSize[g] <= 1.0)
            {
                agreement[a] = 1.0;
                continue;
            }

            double sum = -1.0;
            for (int b = 0; b < m; b++)
                if (groupOf[b] == g)
                    sum += size[b] * (1.0 - distance[a, b]);

            agreement[a] = Math.Clamp(sum / (groupSize[g] - 1.0), 0.0, 1.0);
        }

        return agreement;
    }

    private sealed class SignatureComparer : IEqualityComparer<int[]>
    {
        public bool Equals(int[]? x, int[]? y) => x is not null && y is not null && x.AsSpan().SequenceEqual(y);

        public int GetHashCode(int[] signature)
        {
            var hash = new HashCode();
            foreach (int label in signature)
                hash.Add(label);
            return hash.ToHashCode();
        }
    }
}
