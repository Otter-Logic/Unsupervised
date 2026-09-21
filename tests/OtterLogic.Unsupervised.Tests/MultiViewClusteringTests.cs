using OtterLogic.Graphs;
using OtterLogic.Unsupervised.Clustering;
using Xunit;

namespace OtterLogic.Unsupervised.Tests;

public class MultiViewClusteringTests
{
    [Fact]
    public void PlantedCommunities_AreFoundByEveryViewAndTheConsensus()
    {
        var (graph, x, truth) = Synthetic.Communities(communities: 3, size: 40, inside: 0.2, between: 0.005, noise: 0.3, dimensions: 3);

        var result = MultiViewClustering.Fit(graph, x, x, x);

        Assert.Equal(3, result.Spectral!.ClusterCount);
        Assert.Equal(3, result.HierarchicalLabels!.Max() + 1);
        Assert.Equal(3, result.Groups);
        Assert.True(ClusterAgreement.AdjustedRand(truth, result.Labels) > 0.95);
        Assert.Equal(new[] { "Spectral", "Hierarchical", "Density" }, result.Views.Select(v => v.Name));
    }

    /// <summary>
    /// Samples far from every dense region are the density view's outliers, and the
    /// consensus still places them — outliers abstain, they are not dropped.
    /// </summary>
    [Fact]
    public void FarSamplesAreOutliersButStillGrouped()
    {
        var (graph, x, _) = Synthetic.Communities(communities: 3, size: 40, inside: 0.2, between: 0.005, noise: 0.3, dimensions: 3);
        int n = x.GetLength(0);
        const int far = 4;

        var withFar = new double[n + far, 3];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < 3; j++)
                withFar[i, j] = x[i, j];
        for (int f = 0; f < far; f++)
            for (int j = 0; j < 3; j++)
                withFar[n + f, j] = 12.0 + 3.0 * f + j;

        var edges = graph.Edges().Select(e => (e.A, e.B, e.Weight))
            .Concat(Enumerable.Range(0, far).Select(f => (n + f, f, 1.0)));
        var extended = WeightedGraph.FromEdges(n + far, edges);

        var result = MultiViewClustering.Fit(extended, withFar, withFar, withFar);

        Assert.Equal(Enumerable.Range(n, far), result.Outliers().Where(i => i >= n));
        Assert.All(result.Labels, label => Assert.True(label >= 0));
    }

    [Fact]
    public void AViewWeightedZeroIsSkipped()
    {
        var (graph, x, _) = Synthetic.Communities(communities: 3, size: 40, inside: 0.2, between: 0.005, noise: 0.3, dimensions: 3);

        var result = MultiViewClustering.Fit(graph, x, x, x, new MultiViewClusteringOptions { SpectralWeight = 0.0 });

        Assert.Null(result.Spectral);
        Assert.True(double.IsNaN(result.AlgebraicConnectivity));
        Assert.DoesNotContain(result.Views, v => v.Name == "Spectral");
    }

    [Fact]
    public void AGraphWithNoEdges_SkipsTheSpectralViewAndSaysSo()
    {
        var (_, x, truth) = Synthetic.Communities(communities: 3, size: 40, inside: 0.2, between: 0.005, noise: 0.3, dimensions: 3);
        var empty = WeightedGraph.FromEdges(x.GetLength(0), Array.Empty<(int, int)>());

        var result = MultiViewClustering.Fit(empty, x, x, x);

        // Features alone, noisy enough that the geometry view misfiles a few samples
        // (ARI 0.90) — the graph is what lifts the planted case above 0.95.
        Assert.Null(result.Spectral);
        Assert.Contains(result.Notes, note => note.Contains("spectral view was skipped"));
        Assert.True(ClusterAgreement.AdjustedRand(truth, result.Labels) > 0.85);
    }

    [Fact]
    public void FeaturesOfTheWrongLength_FailWithSomethingReadable()
    {
        var graph = WeightedGraph.FromEdges(5, new[] { (0, 1), (1, 2), (2, 3), (3, 4) });
        var error = Assert.Throws<ArgumentException>(
            () => MultiViewClustering.Fit(graph, new double[5, 2], new double[4, 2], new double[5, 2]));

        Assert.Contains("hierarchyFeatures has 4 rows", error.Message);
    }
}
