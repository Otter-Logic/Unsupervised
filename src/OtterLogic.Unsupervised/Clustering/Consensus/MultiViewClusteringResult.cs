namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// What every view found, and the grouping they agree on.
/// <para>
/// The views are kept, not only the consensus. When a group looks wrong, the
/// useful question is which view put those samples together and which did not —
/// and each view answers a different question, so their disagreements are findings
/// in their own right.
/// </para>
/// </summary>
public sealed class MultiViewClusteringResult
{
    internal MultiViewClusteringResult(
        SpectralClusteringResult? spectral,
        HierarchicalClusteringResult? hierarchy,
        int[]? hierarchicalLabels,
        double hierarchicalSilhouette,
        HdbscanResult? density,
        IReadOnlyList<ClusterView> views,
        ConsensusResult consensus,
        IReadOnlyList<string> notes)
    {
        Spectral = spectral;
        Hierarchy = hierarchy;
        HierarchicalLabels = hierarchicalLabels;
        HierarchicalSilhouette = hierarchicalSilhouette;
        Density = density;
        Views = views;
        Consensus = consensus;
        Notes = notes;
    }

    /// <summary>
    /// The spectral view — samples joined by strong connections that are also
    /// alike — at the count its eigengap chose. Null when skipped.
    /// </summary>
    public SpectralClusteringResult? Spectral { get; }

    /// <summary>The hierarchical view's whole tree. Null when skipped.</summary>
    public HierarchicalClusteringResult? Hierarchy { get; }

    /// <summary>The hierarchical tree cut at the count its silhouette chose. Null when skipped.</summary>
    public int[]? HierarchicalLabels { get; }

    /// <summary>Silhouette of that cut. NaN when skipped.</summary>
    public double HierarchicalSilhouette { get; }

    /// <summary>The density view, with its noise. Null when skipped.</summary>
    public HdbscanResult? Density { get; }

    /// <summary>Every labelling fused, in order: spectral, hierarchical, density, then any the caller added.</summary>
    public IReadOnlyList<ClusterView> Views { get; }

    /// <summary>The fused grouping.</summary>
    public ConsensusResult Consensus { get; }

    /// <summary>Anything that changed what ran — a view skipped, a view that found nothing — in plain words.</summary>
    public IReadOnlyList<string> Notes { get; }

    /// <summary>Group per sample, from the consensus. Every sample is placed.</summary>
    public int[] Labels => Consensus.Labels;

    /// <summary>Number of consensus groups.</summary>
    public int Groups => Consensus.Groups;

    /// <summary>Per sample, how firmly the views agreed about the company it keeps.</summary>
    public double[] Agreement => Consensus.Agreement;

    /// <summary>
    /// The second-smallest eigenvalue of the connectivity graph's normalised
    /// Laplacian, between 0 and 2, from the spectral view. Close to zero means the
    /// graph is only weakly one piece. NaN when the spectral view did not run.
    /// </summary>
    public double AlgebraicConnectivity
        => Spectral is { Eigenvalues.Length: > 1 } spectral ? spectral.Eigenvalues[1] : double.NaN;

    /// <summary>Samples in no dense region, by the density view. Empty when it did not run.</summary>
    public int[] Outliers() => Density?.Noise() ?? Array.Empty<int>();
}
