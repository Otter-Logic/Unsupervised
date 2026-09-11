namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// What every clustering does with its labels once it has them: number them
/// canonically, list the samples in each cluster, and average them.
/// <para>
/// Public because a toolkit holding labels from anywhere — a cut tree, its own
/// rule, a labelling somebody saved — needs the same three things, and had been
/// writing them itself: six copies of the bucketing loop across the results here
/// and more in the toolkits above. One copy means one convention for unplaced
/// samples, which every copy had to remember separately.
/// </para>
/// <para>
/// Canonical numbering is largest cluster first, ties to whichever holds the
/// lowest sample index. Every method here numbers its clusters in whatever order its internals
/// happened to produce — the k-means start that won, the order a tree was cut,
/// the component an eigenvector happened to favour. A small change upstream then
/// permutes the numbers, cluster zero becomes cluster two, and every colour and
/// geometry assignment downstream jumps for no reason a user can see. Numbering by
/// a property of the answer rather than of its history is what stops that, and
/// doing it in one place is what keeps the tie-break the same everywhere.
/// </para>
/// </summary>
public static class ClusterLabels
{
    /// <summary>
    /// Renumbers <paramref name="labels"/> canonically. Negative labels mean
    /// unplaced and come back as <c>-1</c>.
    /// </summary>
    /// <param name="labels">Any non-negative integers per sample, or negative for unplaced.</param>
    /// <param name="mapping">Old label to new, for callers that must reorder something alongside.</param>
    public static int[] Canonical(int[] labels, out IReadOnlyDictionary<int, int> mapping)
    {
        ArgumentNullException.ThrowIfNull(labels);

        var counts = new Dictionary<int, int>();
        var first = new Dictionary<int, int>();

        for (int i = 0; i < labels.Length; i++)
        {
            int label = labels[i];
            if (label < 0)
                continue;

            counts[label] = counts.TryGetValue(label, out int c) ? c + 1 : 1;
            first.TryAdd(label, i);
        }

        var renumber = new Dictionary<int, int>(counts.Count);
        int next = 0;
        foreach (int label in counts.Keys.OrderByDescending(l => counts[l]).ThenBy(l => first[l]))
            renumber[label] = next++;

        var result = new int[labels.Length];
        for (int i = 0; i < labels.Length; i++)
            result[i] = labels[i] < 0 ? -1 : renumber[labels[i]];

        mapping = renumber;
        return result;
    }

    /// <inheritdoc cref="Canonical(int[], out IReadOnlyDictionary{int, int})"/>
    public static int[] Canonical(int[] labels) => Canonical(labels, out _);

    /// <summary>
    /// The sample indices in each cluster, ascending, one array per label from
    /// <c>0</c> to <paramref name="clusterCount"/> − 1. Unplaced samples — negative
    /// labels — are left out; <see cref="Unplaced"/> lists them. A label with no
    /// samples gets an empty array rather than disappearing, so position is always
    /// the label.
    /// </summary>
    /// <param name="labels">One label per sample, negative for unplaced.</param>
    /// <param name="clusterCount">Number of clusters. Every label must be below it.</param>
    public static int[][] Members(int[] labels, int clusterCount)
    {
        ArgumentNullException.ThrowIfNull(labels);
        if (clusterCount < 0)
            throw new ArgumentOutOfRangeException(nameof(clusterCount), clusterCount, "Cannot be negative.");

        var sizes = new int[clusterCount];
        foreach (int label in labels)
        {
            if (label >= clusterCount)
                throw new ArgumentOutOfRangeException(nameof(labels),
                    $"Label {label} is outside 0..{clusterCount - 1}.");
            if (label >= 0)
                sizes[label]++;
        }

        var members = new int[clusterCount][];
        for (int c = 0; c < clusterCount; c++)
            members[c] = new int[sizes[c]];

        var filled = new int[clusterCount];
        for (int i = 0; i < labels.Length; i++)
            if (labels[i] >= 0)
                members[labels[i]][filled[labels[i]]++] = i;

        return members;
    }

    /// <summary>
    /// <see cref="Members(int[], int)"/> with the cluster count taken as one more
    /// than the largest label.
    /// </summary>
    public static int[][] Members(int[] labels)
    {
        ArgumentNullException.ThrowIfNull(labels);
        return Members(labels, labels.Length == 0 ? 0 : Math.Max(labels.Max() + 1, 0));
    }

    /// <summary>
    /// The inverse of <see cref="Members(int[], int)"/>: the label of each
    /// sample from the samples in each cluster, <c>-1</c> for a sample in none.
    /// For a caller that assembled its groups some other way — merged parts,
    /// added one-offs — and now needs a label per sample.
    /// </summary>
    /// <param name="members">Sample indices per cluster; position is the label.</param>
    /// <param name="sampleCount">Number of samples. Every index must be below it.</param>
    /// <exception cref="ArgumentException">A sample is listed in two clusters, or twice in one.</exception>
    public static int[] FromMembers(IReadOnlyList<IReadOnlyList<int>> members, int sampleCount)
    {
        ArgumentNullException.ThrowIfNull(members);
        if (sampleCount < 0)
            throw new ArgumentOutOfRangeException(nameof(sampleCount), sampleCount, "Cannot be negative.");

        var labels = new int[sampleCount];
        Array.Fill(labels, -1);

        for (int c = 0; c < members.Count; c++)
        {
            foreach (int i in members[c])
            {
                if (i < 0 || i >= sampleCount)
                    throw new ArgumentOutOfRangeException(nameof(members),
                        $"Cluster {c} lists sample {i}, outside 0..{sampleCount - 1}.");
                if (labels[i] >= 0)
                    throw new ArgumentException(
                        $"Sample {i} is listed in cluster {labels[i]} and again in cluster {c}.", nameof(members));

                labels[i] = c;
            }
        }

        return labels;
    }

    /// <summary>The samples with a negative label — noise, isolated, left unplaced — ascending.</summary>
    public static int[] Unplaced(int[] labels)
    {
        ArgumentNullException.ThrowIfNull(labels);
        return Enumerable.Range(0, labels.Length).Where(i => labels[i] < 0).ToArray();
    }

    /// <summary>
    /// Mean of the rows of <paramref name="x"/> in each cluster, unplaced rows
    /// excluded. A cluster with no rows keeps a mean of zero.
    /// </summary>
    /// <param name="x">n x d data, rows are samples.</param>
    /// <param name="labels">One label per row, negative for unplaced.</param>
    /// <param name="clusterCount">Number of clusters. Every label must be below it.</param>
    public static double[,] Means(double[,] x, int[] labels, int clusterCount)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(labels);
        if (labels.Length != x.GetLength(0))
            throw new ArgumentException(
                $"{labels.Length} labels for {x.GetLength(0)} rows.", nameof(labels));

        int d = x.GetLength(1);
        var means = new double[clusterCount, d];
        var counts = new int[clusterCount];

        for (int i = 0; i < labels.Length; i++)
        {
            int c = labels[i];
            if (c < 0)
                continue;

            counts[c]++;
            for (int j = 0; j < d; j++)
                means[c, j] += x[i, j];
        }

        for (int c = 0; c < clusterCount; c++)
            if (counts[c] > 0)
                for (int j = 0; j < d; j++)
                    means[c, j] /= counts[c];

        return means;
    }
}
