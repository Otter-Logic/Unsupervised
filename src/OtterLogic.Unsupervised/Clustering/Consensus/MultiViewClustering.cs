using OtterLogic.MachineLearning.Graphs;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Clusters the same samples three ways — by connection, by likeness, by density —
/// and fuses the three into one grouping with <see cref="ConsensusClustering"/>.
/// <para>
/// The mechanism behind a toolkit that has both a graph of how its samples relate
/// and features saying what each is like, and wants the grouping both support
/// rather than the grouping one method would find. Each view is a method this repo
/// already has, run with no count to be told:
/// </para>
/// <list type="bullet">
/// <item><b>Spectral</b> — <see cref="SpectralClustering"/> over the graph, each
/// edge weighted by how alike its ends are. Cuts where connected samples stop being
/// alike. The count is the one with the largest eigengap.</item>
/// <item><b>Hierarchical</b> — <see cref="HierarchicalClustering"/> over features
/// alone, so samples that are alike group together however far apart the graph puts
/// them. The count is the cut with the best silhouette: one method compared against
/// itself, the case a silhouette is fit to judge.</item>
/// <item><b>Density</b> — <see cref="Hdbscan"/>, which finds dense regions and, as
/// importantly, the samples in none. Those abstain from the vote rather than being
/// forced to cast one.</item>
/// <item><b>Profile</b>, when asked for — the hierarchical view's features widened by
/// <see cref="NeighbourhoodProfile"/>, so samples group by the part they play among
/// their neighbours rather than by what they are alone.</item>
/// </list>
/// <para>
/// Which features go to which view is the caller's decision, and is where all the
/// meaning is. Nothing here knows what a sample is — only that it is connected to
/// some samples and alike to others.
/// </para>
/// </summary>
public static class MultiViewClustering
{
    /// <summary>
    /// Clusters n samples by all three views and fuses them.
    /// </summary>
    /// <param name="connectivity">Which samples are related. One node per sample.</param>
    /// <param name="affinityFeatures">
    /// n x a, prepared. What decides how strongly two connected samples belong
    /// together in the spectral view.
    /// </param>
    /// <param name="hierarchyFeatures">n x h, prepared. What the hierarchical view groups on.</param>
    /// <param name="densityFeatures">n x d, prepared. What the density view groups on, and finds outliers in.</param>
    /// <param name="options">Settings; null for the defaults.</param>
    /// <param name="additionalViews">
    /// Further labellings to fuse alongside the three — a learned embedding's
    /// clustering, a labelling somebody corrected by hand. Null for none.
    /// </param>
    public static MultiViewClusteringResult Fit(
        WeightedGraph connectivity,
        double[,] affinityFeatures,
        double[,] hierarchyFeatures,
        double[,] densityFeatures,
        MultiViewClusteringOptions? options = null,
        IReadOnlyList<ClusterView>? additionalViews = null)
    {
        ArgumentNullException.ThrowIfNull(connectivity);
        options ??= new MultiViewClusteringOptions();

        int n = connectivity.NodeCount;
        CheckFeatures(affinityFeatures, n, nameof(affinityFeatures));
        CheckFeatures(hierarchyFeatures, n, nameof(hierarchyFeatures));
        CheckFeatures(densityFeatures, n, nameof(densityFeatures));

        if (n < 3)
            throw new ArgumentException($"Need at least three samples to compare groupings; got {n}.", nameof(connectivity));

        options.Validate(n);

        var views = new List<ClusterView>();
        var notes = new List<string>();

        var spectral = options.SpectralWeight > 0.0 ? Spectral(connectivity, affinityFeatures, options, notes) : null;
        if (spectral is not null)
            views.Add(new ClusterView("Spectral", spectral.Labels, options.SpectralWeight));

        HierarchicalClusteringResult? hierarchy = null;
        int[]? hierarchical = null;
        double hierarchicalSilhouette = double.NaN;
        if (options.HierarchicalWeight > 0.0)
        {
            (hierarchy, hierarchical, hierarchicalSilhouette) = Hierarchical(hierarchyFeatures, options);
            views.Add(new ClusterView("Hierarchical", hierarchical, options.HierarchicalWeight));
        }

        int[]? profile = null;
        if (options.ProfileWeight > 0.0)
        {
            var widened = NeighbourhoodProfile.Embed(hierarchyFeatures, connectivity, options.ProfileHops);
            profile = Hierarchical(widened, options).Labels;
            views.Add(new ClusterView("Profile", profile, options.ProfileWeight));
        }

        HdbscanResult? density = null;
        if (options.DensityWeight > 0.0)
        {
            int size = Math.Min(options.MinimumClusterSize ?? HdbscanOptions.DefaultMinimumClusterSize(n), n);
            density = Hdbscan.Fit(densityFeatures, new HdbscanOptions { MinimumClusterSize = size });

            if (density.ClusterCount == 0)
                notes.Add($"The density view found no region of {size} or more alike samples, so every sample abstains from its vote.");

            views.Add(new ClusterView("Density", density.Labels, options.DensityWeight));
        }

        if (additionalViews is not null)
            views.AddRange(additionalViews);

        if (views.Count == 0 || views.All(v => v.Weight == 0.0 || v.Labels.All(label => label < 0)))
            throw new ArgumentException(
                "No view produced a grouping to fuse. Give at least one view a weight above zero, and check the graph "
                + "connects at least three samples if the spectral view is the only one.", nameof(options));

        var consensus = ConsensusClustering.Fuse(views, connectivity, options.Consensus);

        return new MultiViewClusteringResult(
            spectral, hierarchy, hierarchical, hierarchicalSilhouette, density, views, consensus, notes, profile);
    }

    /// <summary>
    /// Spectral clustering at the count with the largest eigengap: one fit at the
    /// most groups allowed reads the whole spectrum, and a second only when a smaller
    /// count won.
    /// </summary>
    private static SpectralClusteringResult? Spectral(
        WeightedGraph connectivity, double[,] features, MultiViewClusteringOptions options, List<string> notes)
    {
        int placed = Enumerable.Range(0, connectivity.NodeCount).Count(i => connectivity.Degree(i) > 0.0);
        if (placed < 3)
        {
            notes.Add($"The spectral view was skipped: only {placed} sample(s) have any connection.");
            return null;
        }

        int highest = Math.Min(options.MaximumGroups, placed - 1);
        int lowest = Math.Min(options.MinimumGroups, highest);

        SpectralClusteringResult Fit(int k) => SpectralClustering.Fit(features, connectivity,
            new SpectralClusteringOptions { Clusters = k, Seed = options.Seed });

        var probe = Fit(highest);
        var eigenvalues = probe.Eigenvalues;

        int chosen = highest;
        double widest = double.NegativeInfinity;
        for (int k = lowest; k <= Math.Min(highest, eigenvalues.Length - 1); k++)
        {
            double gap = eigenvalues[k] - eigenvalues[k - 1];

            // Strictly wider, so a tie goes to the fewer clusters.
            if (gap > widest + 1e-12)
            {
                widest = gap;
                chosen = k;
            }
        }

        if (probe.GraphComponents > highest)
            notes.Add($"The graph falls into {probe.GraphComponents} disconnected pieces, more than the {highest} "
                + "spectral clusters allowed, so some pieces share a cluster for no reason but the count.");

        return chosen == highest ? probe : Fit(chosen);
    }

    /// <summary>
    /// Most samples a silhouette is measured over when choosing the hierarchical
    /// count. The silhouette is quadratic in the samples and is taken once per
    /// candidate count; past this it is taken over samples spread evenly through the
    /// input order, which keeps a few-thousand-sample solve interactive and, being
    /// evenly spread rather than random, gives the same count on every solve.
    /// </summary>
    private const int SilhouetteSamples = 1500;

    /// <summary>The full tree, cut at the count whose silhouette is best — ties to the fewer clusters.</summary>
    private static (HierarchicalClusteringResult Tree, int[] Labels, double Silhouette) Hierarchical(
        double[,] features, MultiViewClusteringOptions options)
    {
        int n = features.GetLength(0);
        var tree = HierarchicalClustering.Fit(features, new HierarchicalClusteringOptions { Linkage = options.Linkage });

        int highest = Math.Min(options.MaximumGroups, n - 1);
        int lowest = Math.Min(options.MinimumGroups, highest);

        int[] measured = n <= SilhouetteSamples
            ? Enumerable.Range(0, n).ToArray()
            : Enumerable.Range(0, SilhouetteSamples).Select(s => (int)((long)s * n / SilhouetteSamples)).ToArray();

        var subset = new double[measured.Length, features.GetLength(1)];
        for (int r = 0; r < measured.Length; r++)
            for (int j = 0; j < features.GetLength(1); j++)
                subset[r, j] = features[measured[r], j];

        var best = tree.Cut(lowest);
        double bestScore = double.NegativeInfinity;

        for (int k = lowest; k <= highest; k++)
        {
            var labels = tree.Cut(k);
            double score = ClusterQuality.Silhouette(subset, measured.Select(i => labels[i]).ToArray());
            if (score > bestScore + 1e-12)
            {
                bestScore = score;
                best = labels;
            }
        }

        return (tree, best, bestScore);
    }

    private static void CheckFeatures(double[,] features, int n, string name)
    {
        ArgumentNullException.ThrowIfNull(features, name);

        if (features.GetLength(0) != n)
            throw new ArgumentException($"The graph has {n} nodes but {name} has {features.GetLength(0)} rows.", name);
        if (features.GetLength(1) < 1)
            throw new ArgumentException($"{name} has no columns.", name);

        for (int i = 0; i < n; i++)
            for (int j = 0; j < features.GetLength(1); j++)
                if (!double.IsFinite(features[i, j]))
                    throw new ArgumentException($"{name}[{i}, {j}] is {features[i, j]}; features must be finite.", name);
    }
}
