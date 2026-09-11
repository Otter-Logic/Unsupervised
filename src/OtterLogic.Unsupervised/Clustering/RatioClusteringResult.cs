namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// A ratio grouping: which group each sample is in, what each group is sized
/// for, and how much of that the least of its members uses.
/// </summary>
public sealed class RatioClusteringResult
{
    internal RatioClusteringResult(int[] labels, double[,] peaks, double[] leastShare, double minimumShare)
    {
        Labels = labels;
        Peaks = peaks;
        LeastShare = leastShare;
        MinimumShare = minimumShare;
    }

    /// <summary>Group of each sample, largest group first, ties to the lowest sample.</summary>
    public int[] Labels { get; }

    /// <summary>
    /// The largest value of each column in each group, k x d — what one design
    /// for the whole group has to be sized for.
    /// </summary>
    public double[,] Peaks { get; }

    /// <summary>
    /// For each group, the smallest share of the group's peak any member
    /// carries in any column. Never below <see cref="MinimumShare"/>; how far
    /// above says how much slack the share left.
    /// </summary>
    public double[] LeastShare { get; }

    /// <summary>The share the grouping was asked to guarantee.</summary>
    public double MinimumShare { get; }

    /// <summary>Number of groups.</summary>
    public int ClusterCount => LeastShare.Length;

    /// <summary>Sample indices in each group.</summary>
    public int[][] Clusters() => ClusterLabels.Members(Labels, ClusterCount);
}
