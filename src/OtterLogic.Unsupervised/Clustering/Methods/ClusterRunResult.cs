using System.Globalization;
using OtterLogic.Core;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// A clustering read back into the units the samples arrived in, with the
/// readings a person needs to act on it.
/// </summary>
public sealed class ClusterRunResult
{
    private readonly int _topFeatures;

    internal ClusterRunResult(
        ClusteringOutcome outcome, double[,] centres, double silhouette, double daviesBouldin,
        double[,]? map, GroupSignatureResult? signature, bool standardised, int[] keptColumns,
        IReadOnlyList<Note> notes, int topFeatures)
    {
        Outcome = outcome;
        Centres = centres;
        Silhouette = silhouette;
        DaviesBouldin = daviesBouldin;
        Map = map;
        Signature = signature;
        Standardised = standardised;
        KeptColumns = keptColumns;
        Notes = notes;
        _topFeatures = topFeatures;
    }

    /// <summary>What the method itself returned.</summary>
    public ClusteringOutcome Outcome { get; }

    /// <summary>The method's name, or for an automatic choice the name of the one chosen.</summary>
    public string Method => Outcome.Method;

    /// <summary>Cluster per sample, largest first from zero; -1 for a sample in no cluster.</summary>
    public int[] Labels => Outcome.Labels;

    public int ClusterCount => Outcome.ClusterCount;

    public int SampleCount => Outcome.SampleCount;

    /// <summary>Per sample, how firmly it belongs where it was put; null when the method cannot say.</summary>
    public double[]? Confidence => Outcome.Confidence;

    /// <summary>One row per cluster: its mean over every column, in the units the samples arrived in.</summary>
    public double[,] Centres { get; }

    /// <summary>
    /// Mean silhouette over the placed samples, in the space the fit was made in.
    /// Above about 0.5 the clusters are well separated; below 0.25 they barely are.
    /// NaN with fewer than two clusters, where it is not defined.
    /// </summary>
    public double Silhouette { get; }

    /// <summary>Davies-Bouldin index, lower is better. NaN with fewer than two clusters.</summary>
    public double DaviesBouldin { get; }

    /// <summary>One row per sample: where it sits on a 2D or 3D map for looking at; null when no map was asked for.</summary>
    public double[,]? Map { get; }

    /// <summary>What sets each cluster apart, feature by feature; null when not asked for or not definable.</summary>
    public GroupSignatureResult? Signature { get; }

    /// <summary>Whether the columns were brought to a common scale before fitting.</summary>
    public bool Standardised { get; }

    /// <summary>The columns the fit used, in order. Every column when nothing was dropped.</summary>
    public int[] KeptColumns { get; }

    /// <summary>Everything a user should hear about this run, in the order it came up.</summary>
    public IReadOnlyList<Note> Notes { get; }

    /// <summary>Sample indices bucketed by cluster, one array each.</summary>
    public int[][] Members() => Outcome.Members();

    /// <summary>Indices of the samples in no cluster.</summary>
    public int[] Unplaced() => Outcome.Unplaced();

    /// <summary>
    /// The run in words, one line per fact, for a report a person reads without
    /// wiring anything else: what ran, how many clusters and how big, how well
    /// separated, then what sets each cluster apart.
    /// </summary>
    /// <param name="featureNames">One per column, for the explanation; unnamed columns are called Feature 0, Feature 1, ...</param>
    public IReadOnlyList<string> Report(IReadOnlyList<string>? featureNames = null)
    {
        var invariant = CultureInfo.InvariantCulture;
        var lines = new List<string>();

        string headline = $"{Method}: {ClusterCount} cluster{(ClusterCount == 1 ? "" : "s")} from {SampleCount} samples";
        if (Outcome.Rationale is { } rationale)
            headline += " — chosen because " + rationale;
        lines.Add(headline + (headline.EndsWith('.') ? "" : "."));

        var sizes = Members().Select(m => m.Length.ToString(invariant));
        string sizeLine = "Sizes: " + string.Join(", ", sizes);
        if (Outcome.UnplacedCount > 0)
            sizeLine += $" ({Outcome.UnplacedCount} unplaced)";
        lines.Add(sizeLine + ".");

        if (!double.IsNaN(Silhouette))
        {
            string reading = Silhouette >= 0.5 ? "well separated"
                : Silhouette >= 0.25 ? "separated, with some overlap"
                : "weakly separated — the clusters may not be real";
            lines.Add(
                $"Silhouette {Silhouette.ToString("0.00", invariant)} ({reading}); "
                + $"Davies-Bouldin {DaviesBouldin.ToString("0.00", invariant)}, lower is better.");
        }

        lines.Add(Standardised
            ? "Columns were brought to a common scale before fitting."
            : "Columns were used as they arrived; the largest units carry the most weight.");

        lines.AddRange(Outcome.Details);

        if (Signature is not null)
        {
            lines.Add("What sets each cluster apart:");
            lines.AddRange(Signature.Summary(featureNames, _topFeatures)
                .Split('\n').Select(l => l.TrimEnd('\r')).Where(l => l.Length > 0));
        }

        return lines;
    }
}
