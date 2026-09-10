using OtterLogic.MachineLearning.Graphs;
using OtterLogic.Unsupervised.Clustering;
using Xunit;
using Xunit.Abstractions;

namespace OtterLogic.Unsupervised.Tests;

/// <summary>
/// Spectral clustering against scikit-learn on the case it exists for, and on
/// its own terms where there is no reference — combining a graph with features.
/// </summary>
public sealed class SpectralClusteringTests
{
    private readonly ITestOutputHelper _output;

    public SpectralClusteringTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Same graph, same spectrum: the Laplacian eigenvalues are determined by the
    /// graph and must match a dense eigensolve to rounding. The partition is then
    /// k-means on the embedding, so it is compared by ARI — and on interleaved
    /// crescents it should be exact, where k-means on the raw points is not close.
    /// </summary>
    [Fact]
    public void MatchesScikitLearnOnInterleavedCrescents()
    {
        var fixture = Fixture.Load("spectral");
        var x = fixture.Matrix("x");
        var truth = fixture.Integers("true_labels");
        var expected = fixture.Section("expected");
        int k = fixture.Int("clusters");

        var graph = WeightedGraph.NearestNeighbours(x, fixture.Int("neighbours"));
        var result = SpectralClustering.Fit(graph, new SpectralClusteringOptions { Clusters = k });

        double agreement = Numeric.AdjustedRandIndex(expected.Integers("labels"), result.Labels);
        double ourTruth = Numeric.AdjustedRandIndex(truth, result.Labels);
        double kMeansTruth = Numeric.AdjustedRandIndex(
            truth, KMeans.Fit(x, new KMeansOptions { Clusters = k }).Labels);

        _output.WriteLine($"eigenvalues   {string.Join("  ", result.Eigenvalues.Select(v => v.ToString("E3")))}");
        _output.WriteLine($"eigengap      {result.EigenGap:E3}   ({result.Iterations} iterations)");
        _output.WriteLine($"ARI vs scikit-learn  {agreement:F4}");
        _output.WriteLine($"ARI vs truth         {ourTruth:F4}   (k-means on raw points: {kMeansTruth:F4})");

        Assert.True(result.Converged);
        Assert.Equal(1, result.GraphComponents);
        Numeric.Close(expected.Vector("laplacian_eigenvalues"), result.Eigenvalues, 1e-8, "Laplacian eigenvalues");
        Assert.Equal(1.0, agreement, 10);
        Assert.Equal(1.0, ourTruth, 10);
        Assert.True(kMeansTruth < 0.5, $"k-means recovered the crescents (ARI {kMeansTruth:F4}), so the fixture tests nothing");
    }

    /// <summary>
    /// From features alone, the self-tuning kernel over a nearest-neighbour graph
    /// should do at least as well as the binary graph scikit-learn uses.
    /// </summary>
    [Fact]
    public void RecoversCrescentsFromFeaturesAlone()
    {
        var fixture = Fixture.Load("spectral");
        var truth = fixture.Integers("true_labels");

        var result = SpectralClustering.Fit(fixture.Matrix("x"), new SpectralClusteringOptions { Clusters = 2 });
        double ari = Numeric.AdjustedRandIndex(truth, result.Labels);

        _output.WriteLine($"ARI vs truth {ari:F4}, eigengap {result.EigenGap:E3}");
        Assert.Equal(1.0, ari, 10);
    }

    /// <summary>
    /// The reason for the three-argument overload. On a lattice whose behaviour
    /// changes at an off-centre column, connectivity alone cuts across the middle
    /// and features alone scatter stray samples; weighting the lattice's edges by
    /// feature similarity cuts exactly at the boundary, in one piece each side.
    /// </summary>
    [Fact]
    public void ConnectivityAndBehaviourTogetherFindWhatNeitherFindsAlone()
    {
        var (graph, x, truth) = Synthetic.Lattice();
        var options = new SpectralClusteringOptions { Clusters = 2 };

        var both = SpectralClustering.Fit(x, graph, options);
        var connectivityOnly = SpectralClustering.Fit(graph, options);
        var behaviourOnly = SpectralClustering.Fit(x, options);

        double ariBoth = Numeric.AdjustedRandIndex(truth, both.Labels);
        double ariConnectivity = Numeric.AdjustedRandIndex(truth, connectivityOnly.Labels);
        double ariBehaviour = Numeric.AdjustedRandIndex(truth, behaviourOnly.Labels);

        _output.WriteLine($"ARI vs truth: both {ariBoth:F4}, connectivity only {ariConnectivity:F4}, "
                          + $"behaviour only {ariBehaviour:F4}");

        Assert.Equal(1.0, ariBoth, 10);
        Assert.True(Synthetic.EveryClusterIsConnected(graph, both.Labels));
        Assert.True(ariBoth > ariConnectivity + 0.2, "connectivity alone did nearly as well");
        Assert.True(ariBoth > ariBehaviour, "behaviour alone did as well");
    }

    /// <summary>
    /// Disconnected pieces each give an eigenvalue of exactly zero, so asking for
    /// as many clusters as there are pieces returns the pieces, and the eigengap
    /// after them is wide.
    /// </summary>
    [Fact]
    public void ReadsDisconnectedPiecesFromTheSpectrum()
    {
        var edges = new List<(int, int)>();
        for (int piece = 0; piece < 3; piece++)
            for (int a = 0; a < 6; a++)
                for (int b = a + 1; b < 6; b++)
                    edges.Add((piece * 6 + a, piece * 6 + b));

        var graph = WeightedGraph.FromEdges(18, edges);
        var result = SpectralClustering.Fit(graph, new SpectralClusteringOptions { Clusters = 3 });

        _output.WriteLine($"eigenvalues {string.Join("  ", result.Eigenvalues.Select(v => v.ToString("F6")))}");

        Assert.Equal(3, result.GraphComponents);
        Assert.All(result.Eigenvalues.Take(3), v => Assert.True(v < 1e-8, $"eigenvalue {v} is not zero"));
        Assert.True(result.EigenGap > 0.5, $"eigengap {result.EigenGap} is not wide");
        Assert.Equal(1.0, Numeric.AdjustedRandIndex(
            Enumerable.Range(0, 18).Select(i => i / 6).ToArray(), result.Labels), 10);
    }

    /// <summary>
    /// A sample with no edges has an empty row and no place in the embedding.
    /// It comes back unplaced rather than filed wherever the origin fell.
    /// </summary>
    [Fact]
    public void LeavesSamplesWithNoEdgesUnplaced()
    {
        var edges = new List<(int, int)>();
        for (int a = 0; a < 5; a++)
            for (int b = a + 1; b < 5; b++)
            {
                edges.Add((a, b));
                edges.Add((a + 5, b + 5));
            }

        var graph = WeightedGraph.FromEdges(11, edges);
        var result = SpectralClustering.Fit(graph, new SpectralClusteringOptions { Clusters = 2 });

        Assert.Equal(-1, result.Labels[10]);
        Assert.Equal(new[] { 10 }, result.Isolated());
        Assert.Equal(2, result.ClusterCount);
        Assert.Equal(new[] { 0.0, 0.0 }, new[] { result.Embedding[10, 0], result.Embedding[10, 1] });
    }

    [Fact]
    public void IsDeterministicAcrossRepeatedFits()
    {
        var (graph, x, _) = Synthetic.Lattice();
        var options = new SpectralClusteringOptions { Clusters = 3, Seed = 9 };

        var first = SpectralClustering.Fit(x, graph, options);
        var second = SpectralClustering.Fit(x, graph, options);

        Assert.Equal(first.Labels, second.Labels);
        Numeric.Close(first.Embedding, second.Embedding, 0.0, "embedding");
    }
}
