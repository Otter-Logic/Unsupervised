namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// The knobs on an agglomerative fit. There is no cluster count here, and that is
/// the point: the tree holds every count at once, and choosing one is a cut made
/// afterwards — see <see cref="HierarchicalClusteringResult.Cut"/>.
/// </summary>
public sealed record HierarchicalClusteringOptions
{
    /// <summary>How the distance between two clusters is measured. Ward by default.</summary>
    public Linkage Linkage { get; init; } = Linkage.Ward;
}
