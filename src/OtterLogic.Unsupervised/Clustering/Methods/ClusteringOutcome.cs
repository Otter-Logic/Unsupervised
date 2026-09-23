using OtterLogic.Core;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// What every clustering method hands back, whatever it is: the labelling and what
/// the method can honestly say about it.
/// <para>
/// The common ground between six results that are otherwise unalike. A mixture
/// has posteriors and a BIC, HDBSCAN has stabilities and noise, a hierarchy has a
/// tree — and the component that holds the data cannot know which it will get.
/// So a method reduces its result to this: labels, a count, a confidence where it
/// has an honest one, and notes. Anything else it wants seen goes into
/// <see cref="Details"/> as lines for the report.
/// </para>
/// </summary>
public sealed class ClusteringOutcome
{
    /// <param name="method">The method's name, or for an automatic choice the name of the one chosen.</param>
    /// <param name="labels">Cluster per sample, numbered largest first from zero; -1 for a sample in no cluster.</param>
    /// <param name="clusterCount">How many clusters the labels use. Every non-negative label is below it.</param>
    /// <param name="confidence">
    /// Per sample, between zero and one, how firmly it belongs where it was put — or
    /// null when the method has no honest number to give. A made-up one would be
    /// read as if it were honest, which is worse than none.
    /// </param>
    /// <param name="notes">What a user should hear about this fit that the labels alone do not say.</param>
    /// <param name="details">Lines for a report: a BIC, an eigengap, a score table. Informational only.</param>
    /// <param name="rationale">Why this method, when it was chosen rather than asked for. Null otherwise.</param>
    public ClusteringOutcome(
        string method, int[] labels, int clusterCount, double[]? confidence,
        IReadOnlyList<Note>? notes = null, IReadOnlyList<string>? details = null, string? rationale = null)
    {
        ArgumentNullException.ThrowIfNull(labels);
        if (string.IsNullOrWhiteSpace(method))
            throw new ArgumentException("The method needs a name.", nameof(method));
        if (clusterCount < 0)
            throw new ArgumentOutOfRangeException(nameof(clusterCount), clusterCount, "A cluster count cannot be negative.");
        if (confidence is not null && confidence.Length != labels.Length)
            throw new ArgumentException(
                $"{confidence.Length} confidences for {labels.Length} labels; give one per sample or none.", nameof(confidence));

        for (int i = 0; i < labels.Length; i++)
            if (labels[i] >= clusterCount)
                throw new ArgumentException(
                    $"Sample {i} is in cluster {labels[i]}, but there are only {clusterCount}.", nameof(labels));

        Method = method;
        Labels = labels;
        ClusterCount = clusterCount;
        Confidence = confidence;
        Notes = notes ?? Array.Empty<Note>();
        Details = details ?? Array.Empty<string>();
        Rationale = rationale;
    }

    /// <summary>The method's name, or for an automatic choice the name of the one chosen.</summary>
    public string Method { get; }

    /// <summary>Cluster per sample, largest first from zero; -1 for a sample in no cluster.</summary>
    public int[] Labels { get; }

    /// <summary>How many clusters the labels use.</summary>
    public int ClusterCount { get; }

    /// <summary>Per sample, how firmly it belongs where it was put; null when the method cannot say.</summary>
    public double[]? Confidence { get; }

    /// <summary>What a user should hear about this fit that the labels alone do not say.</summary>
    public IReadOnlyList<Note> Notes { get; }

    /// <summary>Lines for a report. Informational only.</summary>
    public IReadOnlyList<string> Details { get; }

    /// <summary>Why this method, when it was chosen rather than asked for.</summary>
    public string? Rationale { get; }

    public int SampleCount => Labels.Length;

    /// <summary>Samples in no cluster.</summary>
    public int UnplacedCount => Labels.Count(l => l < 0);

    /// <summary>Sample indices bucketed by cluster, one array each, in cluster order.</summary>
    public int[][] Members() => ClusterLabels.Members(Labels, ClusterCount);

    /// <summary>Indices of the samples in no cluster.</summary>
    public int[] Unplaced() => ClusterLabels.Unplaced(Labels);
}
