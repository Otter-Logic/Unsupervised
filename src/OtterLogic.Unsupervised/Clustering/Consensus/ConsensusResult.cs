namespace OtterLogic.Unsupervised.Clustering;

/// <summary>One group count the consensus tried, and how well the views supported it.</summary>
/// <param name="Groups">The count.</param>
/// <param name="Lifetime">
/// Range of disagreement, between 0 and 1, over which cutting the fused tree gives
/// this many groups. The count with the longest is chosen: a grouping that survives
/// a wide range of thresholds is one the views support, not one a threshold produced.
/// </param>
/// <param name="Silhouette">
/// Silhouette of that cut, measured on disagreement between views. Reported, not
/// chosen by — see <see cref="ConsensusClustering"/> for why.
/// </param>
public readonly record struct ConsensusCandidate(int Groups, double Lifetime, double Silhouette);

/// <summary>
/// The fused grouping, and how firmly the views behind it agreed — per sample, per
/// view, and at every group count that was tried.
/// </summary>
public sealed class ConsensusResult
{
    internal ConsensusResult(
        int[] labels,
        int groups,
        double[] agreement,
        IReadOnlyList<ClusterView> views,
        double[] weights,
        double[] viewAgreement,
        IReadOnlyList<ConsensusCandidate> sweep,
        double silhouette,
        int mergedGroups,
        int signatures)
    {
        Labels = labels;
        Groups = groups;
        Agreement = agreement;
        Views = views;
        Weights = weights;
        ViewAgreement = viewAgreement;
        Sweep = sweep;
        Silhouette = silhouette;
        MergedGroups = mergedGroups;
        Signatures = signatures;
    }

    /// <summary>Group per sample, largest first. Every sample is placed.</summary>
    public int[] Labels { get; }

    /// <summary>Number of groups.</summary>
    public int Groups { get; }

    /// <summary>
    /// Per sample, between 0 and 1: on average over the rest of its group, the
    /// weighted share of views that put the two together.
    /// <para>
    /// The consensus's own confidence. One means every view that placed this
    /// sample agrees about the company it keeps; a half means the views split on
    /// it, and it is the sample to look at. A group of one scores one — nothing
    /// disagrees with it.
    /// </para>
    /// </summary>
    public double[] Agreement { get; }

    /// <summary>The views fused, in the order given.</summary>
    public IReadOnlyList<ClusterView> Views { get; }

    /// <summary>
    /// The weight each view's vote actually carried, normalised to sum to one —
    /// its own weight, scaled by how far the others agreed with it when
    /// <see cref="ConsensusOptions.WeightByAgreement"/> is on.
    /// </summary>
    public double[] Weights { get; }

    /// <summary>
    /// Adjusted Rand index of each view against the consensus, in view order. A
    /// view far below the rest saw something the others did not — or failed.
    /// </summary>
    public double[] ViewAgreement { get; }

    /// <summary>Every group count tried, with its silhouette. One entry when the count was fixed.</summary>
    public IReadOnlyList<ConsensusCandidate> Sweep { get; }

    /// <summary>
    /// Silhouette of the final grouping, measured on disagreement between views,
    /// after any small groups were merged.
    /// </summary>
    public double Silhouette { get; }

    /// <summary>Groups merged away for being smaller than <see cref="ConsensusOptions.MinimumGroupSize"/>.</summary>
    public int MergedGroups { get; }

    /// <summary>
    /// Distinct combinations of labels across the views. Samples sharing one are
    /// indistinguishable to the fusion, which works on these rather than on every
    /// sample — so this, not the sample count, is what its cost grows with.
    /// </summary>
    public int Signatures { get; }

    /// <summary>Number of samples.</summary>
    public int SampleCount => Labels.Length;

    /// <summary>Sample indices bucketed by group.</summary>
    public int[][] Members() => ClusterLabels.Members(Labels, Groups);
}
