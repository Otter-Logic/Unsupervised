using OtterLogic.Core;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Spectral clustering as a method on a wire: how many clusters, and how many
/// neighbours each sample is connected to. The graph is built here from the
/// samples, so nobody wires one.
/// </summary>
public sealed record SpectralMethod : ClusteringMethod
{
    /// <summary>How many clusters to cut the graph into. At least two.</summary>
    public int Clusters { get; init; } = 4;

    /// <summary>Neighbours each sample connects to. Ten by default; fewer follows thinner shapes, more smooths them.</summary>
    public int Neighbours { get; init; } = 10;

    /// <summary>Restarts of the k-means step in the embedding.</summary>
    public int Restarts { get; init; } = 10;

    /// <summary>Fixed, so the same samples give the same clusters every solve.</summary>
    public int Seed { get; init; } = 1;

    public override string Name => "Spectral";

    public override string Describe() => $"{Name}, {Clusters} clusters over {Neighbours} neighbours";

    private SpectralClusteringOptions Options
        => new() { Clusters = Clusters, Neighbours = Neighbours, Restarts = Restarts, Seed = Seed };

    public override void Validate(int sampleCount) => Options.Validate(sampleCount);

    public override ClusteringOutcome Fit(double[,] x)
    {
        ArgumentNullException.ThrowIfNull(x);
        var fit = SpectralClustering.Fit(x, Options);

        var notes = new List<Note>();
        if (fit.GraphComponents > fit.ClusterCount)
            notes.Add(Note.Warning(
                $"The neighbour graph falls into {fit.GraphComponents} separate pieces and only "
                + $"{fit.ClusterCount} clusters were asked for, so some clusters join pieces nothing "
                + "connects. More neighbours, or as many clusters as pieces."));

        if (!fit.Converged)
            notes.Add(Note.Warning("The k-means step in the embedding did not settle; the clusters may be rough."));

        // The margin in the embedding, where the partition was actually made, is the
        // honest confidence; the samples' own coordinates are not what was cut.
        var centres = ClusterLabels.Means(fit.Embedding, fit.Labels, fit.ClusterCount);
        var confidence = CentroidMargin.Of(fit.Embedding, fit.Labels, centres);

        return new ClusteringOutcome(
            Name, fit.Labels, fit.ClusterCount, confidence, notes,
            details: new[]
            {
                $"Eigengap {fit.EigenGap:0.###}: a large gap means {fit.ClusterCount} was a natural count to ask for.",
            });
    }
}
