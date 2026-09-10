namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Settings for <see cref="MessagePassing.Fit"/>: how to smooth, then how to
/// partition what the smoothing produced.
/// <para>
/// The two halves are nested rather than flattened because they are two separate
/// decisions, and the nesting keeps that visible at the call site — the k-means
/// half is exactly the options <see cref="KMeans"/> takes on its own.
/// </para>
/// </summary>
public sealed record MessagePassingOptions
{
    /// <summary>How far and how hard features spread over the graph before clustering.</summary>
    public PropagationOptions Propagation { get; init; } = new();

    /// <summary>The k-means run on the smoothed features, including the cluster count.</summary>
    public KMeansOptions KMeans { get; init; } = new();
}
