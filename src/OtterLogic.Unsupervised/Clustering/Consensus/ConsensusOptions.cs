namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Settings for <see cref="ConsensusClustering"/>. Every one has a default and the
/// intended call passes none.
/// </summary>
public sealed record ConsensusOptions
{
    /// <summary>Fewest groups the consensus considers when it chooses the count.</summary>
    public int MinimumGroups { get; init; } = 2;

    /// <summary>
    /// Most groups the consensus considers when it chooses the count. Ten, for the
    /// reason <see cref="ClusterSelectorOptions.MaximumGroups"/> gives: every group
    /// is something somebody has to interpret.
    /// </summary>
    public int MaximumGroups { get; init; } = 10;

    /// <summary>
    /// A fixed number of groups, instead of choosing one. Null chooses the count
    /// the views support over the widest range of thresholds.
    /// </summary>
    public int? Groups { get; init; }

    /// <summary>
    /// Groups smaller than this are merged into the group their samples agree with
    /// most — and, when a graph is given, only into a group they are connected to.
    /// One, the default, merges nothing.
    /// </summary>
    public int MinimumGroupSize { get; init; } = 1;

    /// <summary>
    /// Scale each view's weight by how far the other views agree with it.
    /// <para>
    /// On by default, because a view that disagrees with every other is more likely
    /// to be the one that failed on this data than the only one that saw it
    /// clearly — and nothing else in the fusion can tell those apart. A view is
    /// never scaled below a tenth of its weight, so a lone dissenter is outvoted,
    /// not silenced. Off when there is only one view, where there is nothing to
    /// agree with.
    /// </para>
    /// </summary>
    public bool WeightByAgreement { get; init; } = true;

    /// <summary>
    /// Checks these settings against the sample count. Public so a caller that
    /// runs several clusterings first can hear the complaint before it spends the
    /// time.
    /// </summary>
    public void Validate(int sampleCount)
    {
        if (MinimumGroups < 1)
            throw new ArgumentOutOfRangeException(nameof(MinimumGroups), MinimumGroups, "Need at least one group.");
        if (MaximumGroups < MinimumGroups)
            throw new ArgumentOutOfRangeException(nameof(MaximumGroups), MaximumGroups,
                $"Maximum groups ({MaximumGroups}) is below minimum ({MinimumGroups}).");
        if (Groups is { } groups && (groups < 1 || groups > sampleCount))
            throw new ArgumentOutOfRangeException(nameof(Groups), groups,
                $"Can make between 1 and {sampleCount} groups from {sampleCount} samples.");
        if (MinimumGroupSize < 1)
            throw new ArgumentOutOfRangeException(nameof(MinimumGroupSize), MinimumGroupSize,
                "A group needs at least one sample.");
    }
}
