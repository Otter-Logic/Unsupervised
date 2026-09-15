namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Settings for <see cref="MultiViewClustering"/>. Every one has a default and the
/// intended call passes none.
/// </summary>
public sealed record MultiViewClusteringOptions
{
    /// <summary>Fewest clusters the spectral and hierarchical views consider.</summary>
    public int MinimumGroups { get; init; } = 2;

    /// <summary>Most clusters the spectral and hierarchical views consider.</summary>
    public int MaximumGroups { get; init; } = 10;

    /// <summary>How the hierarchical view measures the distance between clusters. Ward by default.</summary>
    public Linkage Linkage { get; init; } = Linkage.Ward;

    /// <summary>
    /// Smallest dense group the density view calls a cluster. Null derives it from
    /// the sample count — see <see cref="HdbscanOptions.DefaultMinimumClusterSize"/>.
    /// </summary>
    public int? MinimumClusterSize { get; init; }

    /// <summary>Vote of the spectral view — connected samples that are alike. Zero skips it.</summary>
    public double SpectralWeight { get; init; } = 1.0;

    /// <summary>Vote of the hierarchical view — samples alike wherever they are. Zero skips it.</summary>
    public double HierarchicalWeight { get; init; } = 1.0;

    /// <summary>Vote of the density view — dense regions, and what sits outside them. Zero skips it.</summary>
    public double DensityWeight { get; init; } = 1.0;

    /// <summary>Seed for the spectral view's eigensolver and k-means. Fixed, so a re-solve returns the same groups.</summary>
    public int Seed { get; init; } = 1;

    /// <summary>How the views are fused.</summary>
    public ConsensusOptions Consensus { get; init; } = new();

    /// <summary>
    /// Checks these settings against the sample count. Public so a caller preparing
    /// features for several views hears the complaint first.
    /// </summary>
    public void Validate(int sampleCount)
    {
        if (MinimumGroups < 2)
            throw new ArgumentOutOfRangeException(nameof(MinimumGroups), MinimumGroups,
                "A view needs at least two clusters to partition anything.");
        if (MaximumGroups < MinimumGroups)
            throw new ArgumentOutOfRangeException(nameof(MaximumGroups), MaximumGroups,
                $"Maximum groups ({MaximumGroups}) is below minimum ({MinimumGroups}).");
        if (MinimumClusterSize is { } size && size < 2)
            throw new ArgumentOutOfRangeException(nameof(MinimumClusterSize), size, "A cluster needs at least two samples.");

        foreach (var (name, weight) in new[]
                 {
                     (nameof(SpectralWeight), SpectralWeight),
                     (nameof(HierarchicalWeight), HierarchicalWeight),
                     (nameof(DensityWeight), DensityWeight),
                 })
            if (!double.IsFinite(weight) || weight < 0.0)
                throw new ArgumentOutOfRangeException(name, weight, "A view's weight must be finite and not negative.");

        ArgumentNullException.ThrowIfNull(Consensus);
        Consensus.Validate(sampleCount);
    }
}
