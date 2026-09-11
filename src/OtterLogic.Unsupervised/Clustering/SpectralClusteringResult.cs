namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// A fitted spectral clustering: where every sample went, the embedding it was
/// decided in, and the spectrum that says whether k was a sensible thing to ask
/// for.
/// </summary>
public sealed class SpectralClusteringResult
{
    internal SpectralClusteringResult(
        int[] labels,
        int clusterCount,
        double[,] embedding,
        double[] eigenvalues,
        int graphComponents,
        double inertia,
        int iterations,
        bool converged)
    {
        Labels = labels;
        ClusterCount = clusterCount;
        Embedding = embedding;
        Eigenvalues = eigenvalues;
        GraphComponents = graphComponents;
        Inertia = inertia;
        Iterations = iterations;
        Converged = converged;
    }

    /// <summary>
    /// Cluster index per sample, largest cluster first, or <c>-1</c> for a sample
    /// with no edges.
    /// <para>
    /// A sample the graph connects to nothing has no place in a spectral
    /// embedding — its row of the operator is empty — and any cluster it was
    /// given would be decided by where the origin happened to fall. It is reported
    /// as unplaced instead, the same convention HDBSCAN uses for noise.
    /// </para>
    /// </summary>
    public int[] Labels { get; }

    /// <summary>Number of clusters, not counting unplaced samples.</summary>
    public int ClusterCount { get; }

    /// <summary>
    /// The spectral embedding, n x k: one row per sample, in which k-means drew
    /// the partition. Rows of unplaced samples are zero.
    /// <para>
    /// These are the random-walk Laplacian's eigenvectors, as scikit-learn uses.
    /// Samples in the same well-connected region land close together here
    /// whatever shape the region had in feature space — which is the whole trick,
    /// and why this is worth feeding to another method as well.
    /// </para>
    /// </summary>
    public double[,] Embedding { get; }

    /// <summary>
    /// Smallest eigenvalues of the normalised graph Laplacian, ascending — k of
    /// them, and one more when there are samples to spare. Between 0 and 2.
    /// <para>
    /// Each connected piece of the graph contributes an eigenvalue of exactly
    /// zero, and each well-separated cluster one close to zero. So the spectrum
    /// is itself a reading of how many clusters there are: see
    /// <see cref="EigenGap"/>.
    /// </para>
    /// </summary>
    public double[] Eigenvalues { get; }

    /// <summary>
    /// Gap between the (k+1)-th and k-th eigenvalue, or NaN when there was no
    /// (k+1)-th to compute.
    /// <para>
    /// The eigengap heuristic. A large gap says the graph has k well-separated
    /// regions and asking for k was natural; a small one says the k-th and
    /// (k+1)-th regions are about as distinct as each other, and the split between
    /// them is closer to arbitrary. Compare it across k rather than reading one
    /// value on its own.
    /// </para>
    /// </summary>
    public double EigenGap => Eigenvalues.Length > ClusterCount
        ? Eigenvalues[ClusterCount] - Eigenvalues[ClusterCount - 1]
        : double.NaN;

    /// <summary>
    /// Connected components among the placed samples.
    /// <para>
    /// Worth checking whenever the graph came from outside. No signal crosses a
    /// disconnection, so every component lands in a cluster of its own or shares
    /// one with other whole components — spectral clustering never splits across
    /// one. When there are at least as many components as clusters, it has
    /// nothing left to decide, and which components end up sharing is arbitrary.
    /// </para>
    /// </summary>
    public int GraphComponents { get; }

    /// <summary>Total squared distance from every embedded sample to its cluster's centre.</summary>
    public double Inertia { get; }

    /// <summary>Eigensolver iterations taken.</summary>
    public int Iterations { get; }

    /// <summary>
    /// Whether the eigensolver met its tolerance. False means the embedding is an
    /// approximation that was still moving; the partition usually survives that,
    /// but the eigenvalues should not be trusted to many places.
    /// </summary>
    public bool Converged { get; }

    /// <summary>Number of samples fitted.</summary>
    public int SampleCount => Labels.Length;

    /// <summary>Number of samples left unplaced because the graph connects them to nothing.</summary>
    public int IsolatedCount => Labels.Count(l => l < 0);

    /// <summary>Sample indices bucketed by cluster, unplaced samples excluded.</summary>
    public int[][] Clusters() => ClusterLabels.Members(Labels, ClusterCount);

    /// <summary>Indices of the samples the graph connects to nothing.</summary>
    public int[] Isolated() => ClusterLabels.Unplaced(Labels);
}
