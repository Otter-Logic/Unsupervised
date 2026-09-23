using OtterLogic.Core;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// HDBSCAN as a method on a wire: the smallest group worth calling a cluster, and
/// nothing else. It finds the count itself and leaves outliers unplaced.
/// </summary>
public sealed record HdbscanMethod : ClusteringMethod
{
    /// <summary>
    /// Smallest group called a cluster, or null to derive one from the number of
    /// samples — two per cent of them, between five and twenty-five.
    /// </summary>
    public int? MinimumClusterSize { get; init; }

    /// <summary>Above this share of unplaced samples the method has found nothing much, and says so.</summary>
    public const double NoiseWorthAWarning = 0.35;

    public override string Name => "HDBSCAN";

    public override string Describe()
        => MinimumClusterSize is { } size
            ? $"{Name}, clusters of at least {size}"
            : $"{Name}, minimum cluster size derived from the sample count";

    private HdbscanOptions OptionsFor(int sampleCount) => new()
    {
        MinimumClusterSize = MinimumClusterSize ?? HdbscanOptions.DefaultMinimumClusterSize(sampleCount),
    };

    public override void Validate(int sampleCount) => OptionsFor(sampleCount).Validate(sampleCount);

    public override ClusteringOutcome Fit(double[,] x)
    {
        ArgumentNullException.ThrowIfNull(x);
        int n = x.GetLength(0);
        var options = OptionsFor(n);
        var fit = Hdbscan.Fit(x, options);

        var notes = new List<Note>();
        if (MinimumClusterSize is null)
            notes.Add(Note.Remark(
                $"Minimum cluster size {options.MinimumClusterSize}, derived from {n} samples. "
                + "Set it to say what a cluster too small to matter is."));

        if (fit.ClusterCount == 0)
            notes.Add(Note.Warning(
                "No dense region was found, so every sample is unplaced. A smaller minimum cluster "
                + "size finds smaller groups; if none appear, the samples may not fall into groups at all."));
        else if (fit.NoiseFraction > NoiseWorthAWarning)
            notes.Add(Note.Warning(
                $"{fit.NoiseFraction:P0} of samples are in no cluster — more than outliers. Either the "
                + "groups are not dense enough for HDBSCAN to see, or the minimum cluster size is too large."));

        return new ClusteringOutcome(Name, fit.Labels, fit.ClusterCount, fit.Probabilities, notes);
    }
}
