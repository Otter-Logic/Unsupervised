using OtterLogic.Core;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Hierarchical clustering as a method on a wire: the whole tree is built and cut
/// into the asked-for count. How many clusters, and how distance between clusters
/// is measured.
/// </summary>
public sealed record HierarchicalMethod : ClusteringMethod
{
    /// <summary>How many clusters to cut the tree into.</summary>
    public int Clusters { get; init; } = 4;

    /// <summary>How the distance between two clusters is measured when merging. Ward by default: the tightest merge.</summary>
    public Linkage Linkage { get; init; } = Linkage.Ward;

    public override string Name => "Hierarchical";

    public override string Describe()
        => $"{Name}, {Clusters} clusters, {Naming.Humanise(Linkage).ToLowerInvariant()} linkage";

    public override void Validate(int sampleCount)
    {
        if (Clusters < 1)
            throw new ArgumentOutOfRangeException(nameof(Clusters), Clusters, "Need at least one cluster.");
        if (Clusters > sampleCount)
            throw new ArgumentOutOfRangeException(nameof(Clusters), Clusters,
                $"Cannot cut {sampleCount} samples into {Clusters} clusters.");
    }

    public override ClusteringOutcome Fit(double[,] x)
    {
        ArgumentNullException.ThrowIfNull(x);
        var tree = HierarchicalClustering.Fit(x, new HierarchicalClusteringOptions { Linkage = Linkage });

        // Cut numbers clusters in tree order; largest first is what every method
        // promises, so a small upstream change does not shuffle colours downstream.
        var labels = ClusterLabels.Canonical(tree.Cut(Clusters));
        int count = labels.Length == 0 ? 0 : labels.Max() + 1;

        // No confidence: a cut tree says which side of a merge a sample fell, not
        // how nearly it fell the other way.
        return new ClusteringOutcome(Name, labels, count, confidence: null, Array.Empty<Note>());
    }
}
