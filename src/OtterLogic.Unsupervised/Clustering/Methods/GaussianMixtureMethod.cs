using OtterLogic.Core;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// A Gaussian mixture as a method on a wire: how many clusters, and what shape each
/// is allowed to take.
/// </summary>
public sealed record GaussianMixtureMethod : ClusteringMethod
{
    /// <summary>How many Gaussians to fit — the number of clusters.</summary>
    public int Clusters { get; init; } = 4;

    /// <summary>The shape each cluster may take. Diagonal by default: an axis-aligned ellipsoid.</summary>
    public CovarianceType Covariance { get; init; } = CovarianceType.Diagonal;

    /// <summary>Refits from different starts, keeping the best likelihood.</summary>
    public int Restarts { get; init; } = 10;

    /// <summary>Fixed, so the same samples give the same clusters every solve.</summary>
    public int Seed { get; init; } = 1;

    /// <summary>Below this posterior a sample is called borderline: the mixture's own reading of "between two clusters".</summary>
    public const double BorderlineBelow = 0.75;

    public override string Name => "Gaussian Mixture";

    public override string Describe()
        => $"{Name}, {Clusters} clusters, {Naming.Humanise(Covariance).ToLowerInvariant()} covariance";

    private GaussianMixtureOptions Options
        => new() { Components = Clusters, Covariance = Covariance, Restarts = Restarts, Seed = Seed };

    public override void Validate(int sampleCount) => Options.Validate(sampleCount);

    public override ClusteringOutcome Fit(double[,] x)
    {
        ArgumentNullException.ThrowIfNull(x);
        var fit = GaussianMixture.Fit(x, Options);

        var notes = new List<Note>();
        if (!fit.Converged)
            notes.Add(Note.Warning(
                $"The mixture hit its iteration cap after {fit.Iterations} rounds without settling. "
                + "Fewer clusters usually settles it."));

        // Largest first, so a small upstream change does not permute the clusters
        // and shuffle every colour downstream.
        var ordered = fit.OrderedByWeight();
        var confidence = ordered.Confidence();

        int borderline = confidence.Count(c => c < BorderlineBelow);
        if (borderline > 0)
            notes.Add(Note.Remark(
                $"{borderline} sample(s) sit below {BorderlineBelow:0.00} confidence, between two clusters. "
                + "Sort by Confidence to find them."));

        return new ClusteringOutcome(
            Name, ordered.Labels(), ordered.ComponentCount, confidence, notes,
            details: new[]
            {
                $"BIC {fit.Bic:0.#}, lower is better; only comparable between fits over the same samples.",
            });
    }
}
