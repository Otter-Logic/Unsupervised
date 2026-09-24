using OtterLogic.Graphs;
using OtterLogic.MachineLearning.Decomposition;
using OtterLogic.MachineLearning.Distances;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Spectral clustering: embed a similarity graph by the leading eigenvectors of
/// its normalised Laplacian, then partition the embedding with k-means.
/// <para>
/// It earns its place beside the other methods by what it clusters on. k-means
/// and a mixture ask whether samples are close; HDBSCAN asks whether the space
/// between them is dense. This asks whether they are <em>connected</em> — whether
/// there is a path of strong edges from one to the other. Two samples at opposite
/// ends of a long thin cluster are far apart and still belong together, and only
/// a method reasoning about the graph sees that.
/// </para>
/// <para>
/// That is also what makes it the method for combining connectivity with
/// behaviour. Handed a graph of which samples are related and the features
/// saying how each behaves, it weights each edge by similarity (see
/// <see cref="Affinity.Gaussian"/>) and cuts where related samples stop behaving
/// alike. Features alone build a nearest-neighbour graph; a graph alone is used
/// as given.
/// </para>
/// <para>
/// Conventions follow scikit-learn's <c>SpectralClustering</c> — the symmetrised
/// nearest-neighbour graph, the random-walk eigenvectors, k-means on the result —
/// so the two can be compared directly. See <c>python/fixtures</c>.
/// </para>
/// </summary>
public static class SpectralClustering
{
    /// <summary>
    /// Residual tolerance handed to the eigensolver. Far tighter than k-means on
    /// the embedding can notice, and cheap, because the leading eigenvalues of a
    /// clustered graph sit well clear of the rest.
    /// </summary>
    private const double EigenTolerance = 1e-8;

    /// <summary>
    /// Clusters by features alone: builds a nearest-neighbour graph, weights it by
    /// feature similarity, and cuts it.
    /// </summary>
    /// <param name="x">n x d data, rows are samples. Expected to be standardised already.</param>
    /// <param name="options">Fit settings.</param>
    public static SpectralClusteringResult Fit(double[,] x, SpectralClusteringOptions options)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(options);

        int n = x.GetLength(0);
        options.Validate(n);

        var graph = NeighbourGraph.Of(x, Math.Min(options.Neighbours, n - 1));
        return Cluster(Affinity.Gaussian(graph, x), options);
    }

    /// <summary>
    /// Clusters by connectivity alone, using the graph's weights as the
    /// similarity.
    /// </summary>
    /// <param name="affinity">Which samples are related, and how strongly. One node per sample.</param>
    /// <param name="options">Fit settings. <see cref="SpectralClusteringOptions.Neighbours"/> is ignored.</param>
    public static SpectralClusteringResult Fit(WeightedGraph affinity, SpectralClusteringOptions options)
    {
        ArgumentNullException.ThrowIfNull(affinity);
        ArgumentNullException.ThrowIfNull(options);

        options.Validate(affinity.NodeCount);
        return Cluster(affinity, options);
    }

    /// <summary>
    /// Clusters by connectivity and behaviour together: keeps the graph's edges,
    /// weights each by how alike its two ends are.
    /// </summary>
    /// <param name="x">n x d features, one row per node. Expected to be standardised already.</param>
    /// <param name="connectivity">Which samples are related. Its weights are kept as a multiplier.</param>
    /// <param name="options">Fit settings. <see cref="SpectralClusteringOptions.Neighbours"/> is ignored.</param>
    public static SpectralClusteringResult Fit(
        double[,] x, WeightedGraph connectivity, SpectralClusteringOptions options)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(connectivity);
        ArgumentNullException.ThrowIfNull(options);

        options.Validate(connectivity.NodeCount);
        return Cluster(Affinity.Gaussian(connectivity, x), options);
    }

    private static SpectralClusteringResult Cluster(WeightedGraph affinity, SpectralClusteringOptions options)
    {
        int n = affinity.NodeCount;
        int k = options.Clusters;

        // Samples with no edges are set aside before the eigenproblem rather than
        // after. Each contributes an empty row, and so an eigenvector of its own
        // at an eigenvalue unrelated to any cluster; leaving them in only gives
        // the solver more directions to wade through.
        var placed = Enumerable.Range(0, n).Where(i => affinity.Degree(i) > 0.0).ToArray();
        int m = placed.Length;

        if (m < k)
            throw new ArgumentException(
                $"Only {m} of {n} samples have any edges, too few for {k} clusters.", nameof(affinity));

        // The map itself — the shifted operator, the eigensolve, the random-walk
        // scaling — lives one layer down in SpectralEmbedding, where a component
        // that only wants to look at the map and a graph model that wants it as an
        // input read exactly what k-means partitions here. Moved down in 2026-09
        // with no expected value changing; the fixtures hold it to that.
        var spectral = SpectralEmbedding.Of(affinity, k, options.Seed, EigenTolerance);

        var embedded = new double[m, k];
        for (int i = 0; i < m; i++)
            for (int c = 0; c < k; c++)
                embedded[i, c] = spectral.Coordinates[placed[i], c];

        var partition = KMeans.Fit(embedded, new KMeansOptions
        {
            Clusters = k,
            Restarts = options.Restarts,
            Seed = options.Seed,
        });

        var raw = new int[n];
        Array.Fill(raw, -1);
        for (int i = 0; i < m; i++)
            raw[placed[i]] = partition.Labels[i];

        var labels = ClusterLabels.Canonical(raw, out var mapping);

        return new SpectralClusteringResult(
            labels, mapping.Count, spectral.Coordinates, spectral.Eigenvalues, spectral.GraphComponents,
            partition.Inertia, spectral.Iterations, spectral.Converged);
    }
}
