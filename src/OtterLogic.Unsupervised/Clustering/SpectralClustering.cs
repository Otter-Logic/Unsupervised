using OtterLogic.MachineLearning.Decomposition;
using OtterLogic.MachineLearning.Graphs;

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

        var graph = WeightedGraph.NearestNeighbours(x, Math.Min(options.Neighbours, n - 1));
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

        var graph = m == n ? affinity : Subgraph(affinity, placed);
        graph.ConnectedComponents(out int components);

        // The normalised adjacency has eigenvalues in [-1, 1]; the ones wanted are
        // the largest. Shifting by the identity and halving maps that onto [0, 1]
        // without moving any eigenvector, and makes the operator positive
        // semi-definite — so largest in value and largest in magnitude agree, and
        // a strongly bipartite piece of graph, with eigenvalues near -1, cannot
        // crowd out the ones near +1.
        double[,] Multiply(double[,] block)
        {
            var y = graph.Propagate(block, selfWeight: 0.0);
            for (int i = 0; i < y.GetLength(0); i++)
                for (int c = 0; c < y.GetLength(1); c++)
                    y[i, c] = 0.5 * (y[i, c] + block[i, c]);

            return y;
        }

        int wanted = Math.Min(k + 1, m);
        var (values, vectors, iterations, converged) =
            LeadingEigen.Solve(m, Multiply, wanted, options.Seed, EigenTolerance);

        // Back from the shifted adjacency to the Laplacian: L = I - A_norm, and
        // the shift made each value (1 + a) / 2, so the Laplacian's is 2 - 2v.
        var eigenvalues = values.Select(v => Math.Clamp(2.0 - 2.0 * v, 0.0, 2.0)).ToArray();

        // Divide by the square root of degree to turn the symmetric Laplacian's
        // eigenvectors into the random-walk Laplacian's — scikit-learn's
        // embedding, and von Luxburg's recommendation. Without it a
        // low-degree sample on a cluster's fringe sits nearer the origin than its
        // cluster does, and k-means can file it with the wrong one.
        var embedded = new double[m, k];
        for (int i = 0; i < m; i++)
        {
            double scale = 1.0 / Math.Sqrt(graph.Degree(i));
            for (int c = 0; c < k; c++)
                embedded[i, c] = vectors[i, c] * scale;
        }

        var partition = KMeans.Fit(embedded, new KMeansOptions
        {
            Clusters = k,
            Restarts = options.Restarts,
            Seed = options.Seed,
        });

        var raw = new int[n];
        Array.Fill(raw, -1);
        var embedding = new double[n, k];

        for (int i = 0; i < m; i++)
        {
            raw[placed[i]] = partition.Labels[i];
            for (int c = 0; c < k; c++)
                embedding[placed[i], c] = embedded[i, c];
        }

        var labels = Labelling.Canonical(raw, out var mapping);

        return new SpectralClusteringResult(
            labels, mapping.Count, embedding, eigenvalues, components,
            partition.Inertia, iterations, converged);
    }

    /// <summary>The graph restricted to <paramref name="nodes"/>, renumbered 0..m-1 in the same order.</summary>
    private static WeightedGraph Subgraph(WeightedGraph graph, int[] nodes)
    {
        var index = new Dictionary<int, int>(nodes.Length);
        for (int i = 0; i < nodes.Length; i++)
            index[nodes[i]] = i;

        var edges = graph.Edges()
            .Where(e => index.ContainsKey(e.A) && index.ContainsKey(e.B))
            .Select(e => (index[e.A], index[e.B], e.Weight));

        return WeightedGraph.FromEdges(nodes.Length, edges);
    }
}
