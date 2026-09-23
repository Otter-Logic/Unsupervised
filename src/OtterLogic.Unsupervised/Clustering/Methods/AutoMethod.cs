using OtterLogic.Core;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// No method chosen: K-Means, a Gaussian mixture and HDBSCAN are each fitted, and
/// the one the data supports is kept, with the reason in words.
/// <para>
/// What a data component runs when nothing is wired to its Method input. The
/// comparison and the rules belong to <see cref="ClusterSelector"/>; this makes
/// it a method like the others, so the data component has one call to make.
/// </para>
/// </summary>
public sealed record AutoMethod : ClusteringMethod
{
    /// <summary>Fewest clusters K-Means and the mixture consider.</summary>
    public int MinimumClusters { get; init; } = 2;

    /// <summary>Most clusters K-Means and the mixture consider. Ten, because every cluster is something a person has to read.</summary>
    public int MaximumClusters { get; init; } = 10;

    /// <summary>Fixed, so the same samples give the same clusters every solve.</summary>
    public int Seed { get; init; } = 1;

    /// <summary>The selector needs this many samples to have anything to compare.</summary>
    public const int FewestSamples = 4;

    public override string Name => "Auto";

    public override string Describe()
        => $"{Name}: K-Means, Gaussian Mixture or HDBSCAN, {MinimumClusters} to {MaximumClusters} clusters";

    private ClusterSelectorOptions Options
        => new() { MinimumGroups = MinimumClusters, MaximumGroups = MaximumClusters, Seed = Seed };

    public override void Validate(int sampleCount)
    {
        if (sampleCount < FewestSamples)
            throw new ArgumentException(
                $"Choosing a method needs at least {FewestSamples} samples to compare on; there are {sampleCount}. "
                + "Wire a method to run one directly.");
        Options.Validate(sampleCount);
    }

    public override ClusteringOutcome Fit(double[,] x)
    {
        ArgumentNullException.ThrowIfNull(x);
        var selection = ClusterSelector.Select(x, Options);

        string chosen = ClusterSelection.Name(selection.Chosen);
        var details = new List<string> { "How each method did:" };
        details.AddRange(selection.ScoreTable().Split('\n').Select(l => l.TrimEnd('\r')).Where(l => l.Length > 0));

        return new ClusteringOutcome(
            chosen, selection.Labels, selection.Groups, selection.Confidence,
            Array.Empty<Note>(), details, rationale: selection.Rationale);
    }
}
