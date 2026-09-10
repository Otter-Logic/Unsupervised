namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// An existing labelling after its neighbours have had their say: the updated
/// soft memberships, and exactly which samples changed their mind.
/// <para>
/// Cluster numbers are the input's own. Refinement adjusts an answer somebody
/// already has, so renumbering it would break every reference to it downstream.
/// </para>
/// </summary>
public sealed class RefinementResult
{
    internal RefinementResult(double[,] responsibilities, int[] labels, double[] confidence, int[] initialLabels)
    {
        Responsibilities = responsibilities;
        Labels = labels;
        Confidence = confidence;
        InitialLabels = initialLabels;
    }

    /// <summary>
    /// Refined soft membership, n x k. Rows sum to one, except for a sample the
    /// propagation never reached, whose row is all zero.
    /// </summary>
    public double[,] Responsibilities { get; }

    /// <summary>Most probable cluster per sample after refinement, or <c>-1</c> if never reached.</summary>
    public int[] Labels { get; }

    /// <summary>Highest refined membership per sample, between 1/k and 1; zero if never reached.</summary>
    public double[] Confidence { get; }

    /// <summary>The labelling that went in, as the most probable cluster per sample, <c>-1</c> for none.</summary>
    public int[] InitialLabels { get; }

    /// <summary>Number of clusters.</summary>
    public int ClusterCount => Responsibilities.GetLength(1);

    /// <summary>Number of samples.</summary>
    public int SampleCount => Labels.Length;

    /// <summary>
    /// Samples whose cluster differs from the one they came in with — including
    /// unlabelled samples that the propagation placed.
    /// <para>
    /// The output worth reviewing. A sample moved by its neighbours is one whose
    /// own features and whose surroundings disagreed, which is either a mistake in
    /// the first labelling corrected, or a genuinely unusual sample smoothed over.
    /// Only somebody who knows the data can say which.
    /// </para>
    /// </summary>
    public int[] Changed()
        => Enumerable.Range(0, Labels.Length).Where(i => Labels[i] != InitialLabels[i]).ToArray();

    /// <summary>Samples with no labelled sample within reach, still unplaced.</summary>
    public int[] Unreached()
        => Enumerable.Range(0, Labels.Length).Where(i => Labels[i] < 0).ToArray();

    /// <summary>Sample indices bucketed by refined cluster, unreached samples excluded.</summary>
    public int[][] Clusters() => Labelling.Buckets(Labels, ClusterCount);
}
