using System.Globalization;
using System.Text;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// What <see cref="ClusterSelector"/> decided: which model was chosen and why,
/// where every sample ended up, and how all three scored.
/// <para>
/// Everything here is in the space the selector ran in. A caller that prepared
/// its data — standardised it, projected it — owns mapping the answer back into
/// whatever units it started in, because only the caller knows what the columns
/// mean.
/// </para>
/// </summary>
public sealed class ClusterSelection
{
    internal ClusterSelection(
        ClusteringModel chosen,
        string rationale,
        ClusterCandidate winner,
        IReadOnlyList<ClusterCandidate> candidates,
        double[,] data)
    {
        Chosen = chosen;
        Rationale = rationale;
        Winner = winner;
        Candidates = candidates;
        Data = data;
    }

    /// <summary>Which model the comparison chose.</summary>
    public ClusteringModel Chosen { get; }

    /// <summary>
    /// One sentence saying why, in the terms the choice was actually made on.
    /// Meant to be shown to the user, not logged.
    /// </summary>
    public string Rationale { get; }

    /// <summary>The chosen model's candidate, with its scores.</summary>
    public ClusterCandidate Winner { get; }

    /// <summary>All three candidates, chosen or not, in model order.</summary>
    public IReadOnlyList<ClusterCandidate> Candidates { get; }

    /// <summary>The matrix the selection ran on, one row per sample.</summary>
    public double[,] Data { get; }

    /// <summary>Cluster per sample; <c>-1</c> means unassigned.</summary>
    public int[] Labels => Winner.Labels;

    /// <summary>Per-sample confidence in its assignment.</summary>
    public double[] Confidence => Winner.Confidence;

    /// <summary>Number of clusters found.</summary>
    public int Groups => Winner.Groups;

    /// <summary>Number of samples clustered.</summary>
    public int SampleCount => Labels.Length;

    /// <summary>Samples the chosen model declined to place. Only HDBSCAN can produce these.</summary>
    public int[] Unassigned() => ClusterLabels.Unplaced(Labels);

    /// <summary>Sample indices bucketed by cluster, unassigned samples excluded.</summary>
    public int[][] Members() => ClusterLabels.Members(Labels, Groups);

    /// <summary>
    /// Cluster centres in <see cref="Data"/>'s own space, one row per cluster,
    /// taken as the mean of the samples assigned to it.
    /// <para>
    /// Computed from the labels rather than from any one model's own centres, so
    /// all three report the same thing in the same way — HDBSCAN has no centroid
    /// of its own, and a mixture mean is not the mean of the samples assigned to
    /// it.
    /// </para>
    /// </summary>
    public double[,] GroupCentres() => ClusterLabels.Means(Data, Labels, Groups);

    /// <summary>
    /// How all three models scored, as a fixed-width block. The half of a report
    /// that is about clustering rather than about whatever the samples are, so a
    /// caller can head it with its own lines and print this underneath.
    /// </summary>
    public string ScoreTable()
    {
        var text = new StringBuilder();
        var invariant = CultureInfo.InvariantCulture;

        text.AppendLine("Model              Groups  Silhouette  Davies-Bouldin  Confidence  Boundary  Unplaced");

        foreach (var candidate in Candidates)
        {
            string db = double.IsNaN(candidate.DaviesBouldin)
                ? "     -"
                : candidate.DaviesBouldin.ToString("0.000", invariant).PadLeft(6);

            // Only the mixture's confidence is a probability, so only its
            // boundary share means anything. A k-means margin on the same scale
            // reads as though almost everything is borderline.
            string boundary = candidate.Model == ClusteringModel.GaussianMixture
                ? (candidate.AmbiguousFraction * 100.0).ToString("0.0", invariant) + "%"
                : "-";

            text.AppendLine(
                $"{(candidate.Model == Chosen ? "> " : "  ")}{Name(candidate.Model),-16} "
                + $"{candidate.Groups,6}  "
                + $"{candidate.Silhouette.ToString("0.000", invariant),10}  "
                + $"{db,14}  "
                + $"{candidate.MeanConfidence.ToString("0.000", invariant),10}  "
                + $"{boundary,8}  "
                + $"{(candidate.NoiseFraction * 100.0).ToString("0.0", invariant) + "%",8}");
        }

        text.AppendLine();
        text.Append(
            "Confidence is each model's own measure and is not comparable between rows. Boundary is "
            + $"the share of samples below {ClusterCandidate.AmbiguousBelow:0.00} posterior "
            + "probability, which only the mixture has; unplaced is the share in no cluster at all.");

        return text.ToString();
    }

    /// <summary>The model's name as it should be shown to a user.</summary>
    public static string Name(ClusteringModel model) => model switch
    {
        ClusteringModel.KMeans => "K-Means",
        ClusteringModel.GaussianMixture => "Gaussian Mixture",
        ClusteringModel.Hdbscan => "HDBSCAN",
        _ => model.ToString(),
    };
}
