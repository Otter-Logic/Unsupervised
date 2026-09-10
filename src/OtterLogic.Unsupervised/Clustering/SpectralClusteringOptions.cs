namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// The knobs on a spectral fit. Defaults suit a few hundred to a few thousand
/// samples.
/// </summary>
public sealed record SpectralClusteringOptions
{
    /// <summary>Number of clusters, k — and so the number of eigenvectors the embedding keeps.</summary>
    public int Clusters { get; init; } = 4;

    /// <summary>
    /// Neighbours per sample when the graph is built from the features alone.
    /// Ignored when the caller supplies a graph.
    /// <para>
    /// Ten is scikit-learn's default. Too few and a genuine cluster fragments into
    /// pieces the graph never joins; too many and edges start bridging clusters
    /// that are merely close, which is exactly the distinction this method exists
    /// to draw.
    /// </para>
    /// </summary>
    public int Neighbours { get; init; } = 10;

    /// <summary>
    /// Restarts of the k-means run on the embedding, keeping the lowest inertia.
    /// Cheap: the embedding has only k columns.
    /// </summary>
    public int Restarts { get; init; } = 10;

    /// <summary>
    /// Seed for the eigensolver's start and the k-means on the embedding. Fixed,
    /// so a Grasshopper re-solve returns the same clusters.
    /// </summary>
    public int Seed { get; init; } = 1;

    internal void Validate(int sampleCount)
    {
        if (Clusters < 2)
            throw new ArgumentOutOfRangeException(nameof(Clusters), Clusters,
                "Spectral clustering needs at least two clusters; one is not a partition.");
        if (Clusters > sampleCount)
            throw new ArgumentOutOfRangeException(nameof(Clusters), Clusters,
                $"Cannot fit {Clusters} clusters to {sampleCount} samples.");
        if (Neighbours < 1)
            throw new ArgumentOutOfRangeException(nameof(Neighbours), Neighbours, "Need at least one neighbour.");
        if (Restarts < 1)
            throw new ArgumentOutOfRangeException(nameof(Restarts), Restarts, "Need at least one restart.");
    }
}
