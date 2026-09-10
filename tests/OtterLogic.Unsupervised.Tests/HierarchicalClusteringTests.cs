using OtterLogic.MachineLearning.Graphs;
using OtterLogic.Unsupervised.Clustering;
using Xunit;
using Xunit.Abstractions;

namespace OtterLogic.Unsupervised.Tests;

/// <summary>
/// Hierarchical clustering against SciPy and scikit-learn.
/// <para>
/// Unlike the mixture there is nothing to initialise and no local optimum: given
/// the data and the linkage, the tree is determined. So these are exactness
/// tests all the way — merge for merge, distance for distance.
/// </para>
/// </summary>
public sealed class HierarchicalClusteringTests
{
    private readonly ITestOutputHelper _output;

    public HierarchicalClusteringTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(Linkage.Ward, "ward")]
    [InlineData(Linkage.Complete, "complete")]
    [InlineData(Linkage.Average, "average")]
    [InlineData(Linkage.Single, "single")]
    public void MatchesScipyLinkage(Linkage linkage, string method)
    {
        var fixture = Fixture.Load("hierarchical");
        var x = fixture.Matrix("x");
        var expected = fixture.Section("expected").Section("linkage").Matrix(method);

        var result = HierarchicalClustering.Fit(x, new HierarchicalClusteringOptions { Linkage = linkage });

        Assert.Equal(expected.GetLength(0), result.Merges.Count);
        for (int t = 0; t < result.Merges.Count; t++)
        {
            var merge = result.Merges[t];
            Assert.Equal((int)expected[t, 0], merge.Left);
            Assert.Equal((int)expected[t, 1], merge.Right);
            Assert.Equal((int)expected[t, 3], merge.Size);
            Numeric.Close(expected[t, 2], merge.Distance, 1e-10, $"{method} merge {t} distance");
        }
    }

    /// <summary>
    /// scikit-learn's structured Ward tree over the same connectivity graph. Its
    /// children are written larger node first, so pairs are compared unordered.
    /// </summary>
    [Fact]
    public void ConstrainedWardMatchesScikitLearn()
    {
        var fixture = Fixture.Load("hierarchical");
        var x = fixture.Matrix("x");
        var graph = EdgesOf(fixture);
        var expected = fixture.Section("expected").Section("constrained_ward");
        var children = expected.Matrix("children");
        var distances = expected.Vector("distances");

        var result = HierarchicalClustering.Fit(x, graph);

        Assert.Equal(1, result.GraphComponents);
        for (int t = 0; t < result.Merges.Count; t++)
        {
            int a = (int)children[t, 0];
            int b = (int)children[t, 1];
            Assert.Equal(Math.Min(a, b), result.Merges[t].Left);
            Assert.Equal(Math.Max(a, b), result.Merges[t].Right);
            Numeric.Close(distances[t], result.Merges[t].Distance, 1e-10, $"merge {t} distance");
        }
    }

    /// <summary>
    /// The property that makes a tree worth more than several flat fits: a fine
    /// cut nests inside a coarse one, every fine cluster wholly inside one coarse
    /// cluster.
    /// </summary>
    [Theory]
    [InlineData(Linkage.Ward)]
    [InlineData(Linkage.Average)]
    public void FineCutsNestInsideCoarseOnes(Linkage linkage)
    {
        var x = Fixture.Load("hierarchical").Matrix("x");
        var result = HierarchicalClustering.Fit(x, new HierarchicalClusteringOptions { Linkage = linkage });

        var fine = result.Cut(9);
        var coarse = result.Cut(3);

        Assert.Equal(9, fine.Distinct().Count());
        Assert.Equal(3, coarse.Distinct().Count());

        foreach (int cluster in fine.Distinct())
        {
            var parents = Enumerable.Range(0, fine.Length).Where(i => fine[i] == cluster)
                .Select(i => coarse[i]).Distinct().ToArray();
            Assert.Single(parents);
        }
    }

    /// <summary>
    /// With a connectivity graph, every cluster at every level must be one
    /// connected piece of it — for every linkage, not only the one with a
    /// reference to compare against.
    /// </summary>
    [Theory]
    [InlineData(Linkage.Ward)]
    [InlineData(Linkage.Complete)]
    [InlineData(Linkage.Average)]
    [InlineData(Linkage.Single)]
    public void ConstrainedClustersStayInOnePiece(Linkage linkage)
    {
        var (graph, x, _) = Synthetic.Lattice(columns: 12, rows: 6);
        var result = HierarchicalClustering.Fit(x, graph, new HierarchicalClusteringOptions { Linkage = linkage });

        foreach (int k in new[] { 2, 3, 5, 8, 20 })
            Assert.True(Synthetic.EveryClusterIsConnected(graph, result.Cut(k)),
                $"{linkage}: a cluster in the {k}-way cut is not connected");
    }

    /// <summary>
    /// A disconnected graph cannot be merged into one cluster under the
    /// constraint, but the tree must still be complete. The top merges join the
    /// pieces, and cutting just below them returns the pieces exactly.
    /// </summary>
    [Fact]
    public void CompletesTheTreeAcrossADisconnectedGraph()
    {
        var (lattice, x, _) = Synthetic.Lattice(columns: 10, rows: 4);
        var severed = lattice.Reweight((a, b, w) => a / 4 == 4 && b / 4 == 5 ? 0.0 : w);

        var result = HierarchicalClustering.Fit(x, severed);
        var halves = result.Cut(2);

        Assert.Equal(2, result.GraphComponents);
        Assert.Equal(39, result.Merges.Count);
        Assert.Equal(1.0, Numeric.AdjustedRandIndex(
            Enumerable.Range(0, 40).Select(i => i / 4 < 5 ? 0 : 1).ToArray(), halves), 10);
    }

    [Fact]
    public void CuttingByDistanceAgreesWithCuttingByCount()
    {
        var x = Fixture.Load("hierarchical").Matrix("x");
        var result = HierarchicalClustering.Fit(x);

        // Halfway between the fourth- and fifth-last merge: four clusters.
        var merges = result.Merges;
        double threshold = 0.5 * (merges[^4].Distance + merges[^3].Distance);

        Assert.Equal(4, result.ClustersAtDistance(threshold));
        Assert.Equal(result.Cut(4), result.CutAtDistance(threshold));
        _output.WriteLine($"threshold {threshold:F4} gives 4 clusters");
    }

    private static WeightedGraph EdgesOf(Fixture fixture)
    {
        var edges = fixture.Matrix("edges");
        var list = Enumerable.Range(0, edges.GetLength(0)).Select(e => ((int)edges[e, 0], (int)edges[e, 1]));
        return WeightedGraph.FromEdges(fixture.Matrix("x").GetLength(0), list);
    }
}
