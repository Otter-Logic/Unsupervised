using OtterLogic.Core;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// K-Means as a method on a wire: told how many clusters, and nothing else a user
/// has to understand.
/// <para>
/// Restarts and the seed are carried with defaults rather than exposed, because
/// they are about the optimiser and not about the data: ten restarts reliably
/// beat one for milliseconds, and a fixed seed is the only reason a re-solve
/// returns the same clusters. A caller that knows better can set them.
/// </para>
/// </summary>
public sealed record KMeansMethod : ClusteringMethod
{
    /// <summary>How many clusters. K-Means cannot decide this for itself.</summary>
    public int Clusters { get; init; } = 4;

    /// <summary>Refits from different starts, keeping the tightest.</summary>
    public int Restarts { get; init; } = 10;

    /// <summary>Fixed, so the same samples give the same clusters every solve.</summary>
    public int Seed { get; init; } = 1;

    public override string Name => "K-Means";

    public override string Describe() => $"{Name}, {Clusters} clusters";

    private KMeansOptions Options => new() { Clusters = Clusters, Restarts = Restarts, Seed = Seed };

    public override void Validate(int sampleCount) => Options.Validate(sampleCount);

    public override ClusteringOutcome Fit(double[,] x)
    {
        ArgumentNullException.ThrowIfNull(x);
        var fit = KMeans.Fit(x, Options);

        var notes = new List<Note>();
        if (!fit.Converged)
            notes.Add(Note.Warning(
                $"K-Means hit its iteration cap after {fit.Iterations} rounds without settling. "
                + "The clusters are usable but a different count may fit more cleanly."));

        return new ClusteringOutcome(
            Name, fit.Labels, fit.ClusterCount,
            CentroidMargin.Of(x, fit.Labels, fit.Centroids),
            notes,
            details: new[] { $"Total distance to centres (inertia) {fit.Inertia:0.###}, lower is tighter." });
    }
}
