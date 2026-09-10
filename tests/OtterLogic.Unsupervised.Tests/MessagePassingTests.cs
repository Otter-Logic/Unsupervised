using OtterLogic.MachineLearning.Graphs;
using OtterLogic.Unsupervised.Clustering;
using Xunit;
using Xunit.Abstractions;

namespace OtterLogic.Unsupervised.Tests;

/// <summary>
/// Message passing has no scikit-learn counterpart to compare with, so these test
/// the operator against its definition and the method against planted structure.
/// </summary>
public sealed class MessagePassingTests
{
    private readonly ITestOutputHelper _output;

    public MessagePassingTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Two hops of <c>H = (1 - a) S H + a X</c>, written out densely on a small
    /// weighted graph and compared with the sparse implementation.
    /// </summary>
    [Fact]
    public void SmoothingFollowsTheRecurrence()
    {
        var graph = WeightedGraph.FromEdges(4, new[] { (0, 1, 1.0), (1, 2, 0.5), (2, 3, 2.0), (0, 2, 0.25) });
        var x = new double[,] { { 1.0, 0.0 }, { 0.0, 2.0 }, { -1.0, 1.0 }, { 3.0, -2.0 } };
        const double retention = 0.3;

        var a = new double[4, 4];
        foreach (var (p, q, w) in graph.Edges())
            a[p, q] = a[q, p] = w;
        for (int i = 0; i < 4; i++)
            a[i, i] = 1.0;

        var degree = Enumerable.Range(0, 4).Select(i => Enumerable.Range(0, 4).Sum(j => a[i, j])).ToArray();
        var h = (double[,])x.Clone();
        for (int hop = 0; hop < 2; hop++)
        {
            var next = new double[4, 2];
            for (int i = 0; i < 4; i++)
                for (int c = 0; c < 2; c++)
                {
                    for (int j = 0; j < 4; j++)
                        next[i, c] += a[i, j] / Math.Sqrt(degree[i] * degree[j]) * h[j, c];

                    next[i, c] = (1.0 - retention) * next[i, c] + retention * x[i, c];
                }

            h = next;
        }

        var actual = MessagePassing.Smooth(x, graph, new PropagationOptions { Hops = 2, Retention = retention });
        Numeric.Close(h, actual, 1e-14, "smoothed");
    }

    [Fact]
    public void ZeroHopsOrFullRetentionReturnTheInput()
    {
        var (graph, x, _) = Synthetic.Communities(communities: 2, size: 10);

        Numeric.Close(x, MessagePassing.Smooth(x, graph, new PropagationOptions { Hops = 0 }), 0.0, "zero hops");
        Numeric.Close(x, MessagePassing.Smooth(x, graph, new PropagationOptions { Hops = 5, Retention = 1.0 }),
            0.0, "full retention");
    }

    /// <summary>
    /// The reason for the self-loop. Two samples joined by one edge are the
    /// smallest bipartite graph: pure neighbour averaging would swap their values
    /// every hop forever. With the self-loop they settle on a shared value.
    /// </summary>
    [Fact]
    public void SettlesOnBipartiteGraphsInsteadOfOscillating()
    {
        var graph = WeightedGraph.FromEdges(2, new[] { (0, 1) });
        var x = new double[,] { { 1.0 }, { 0.0 } };

        var odd = MessagePassing.Smooth(x, graph, new PropagationOptions { Hops = 9, Retention = 0.0 });
        var even = MessagePassing.Smooth(x, graph, new PropagationOptions { Hops = 10, Retention = 0.0 });

        Numeric.Close(odd, even, 1e-12, "consecutive hops");
        Numeric.Close(0.5, even[0, 0], 1e-12, "settled value");
    }

    /// <summary>
    /// Features so noisy that k-means on them alone barely beats chance, on a
    /// graph whose communities are the truth. Passing messages first recovers
    /// them — and the retention is what stops adding hops from undoing that.
    /// </summary>
    [Fact]
    public void TheGraphRecoversCommunitiesTheFeaturesAloneCannot()
    {
        var (graph, x, truth) = Synthetic.Communities();
        var kMeans = new KMeansOptions { Clusters = 4 };

        double alone = Numeric.AdjustedRandIndex(truth, KMeans.Fit(x, kMeans).Labels);

        double Score(int hops, double retention) => Numeric.AdjustedRandIndex(truth, MessagePassing.Fit(x, graph,
            new MessagePassingOptions
            {
                Propagation = new PropagationOptions { Hops = hops, Retention = retention },
                KMeans = kMeans,
            }).Labels);

        double defaults = Numeric.AdjustedRandIndex(truth,
            MessagePassing.Fit(x, graph, new MessagePassingOptions { KMeans = kMeans }).Labels);

        _output.WriteLine($"k-means on features alone           ARI {alone:F4}");
        _output.WriteLine($"message passing, defaults           ARI {defaults:F4}");
        foreach (int hops in new[] { 1, 2, 4, 8, 16, 32, 64 })
            _output.WriteLine($"hops {hops,2}: retention 0.0 ARI {Score(hops, 0.0):F4}   "
                              + $"retention 0.1 ARI {Score(hops, 0.1):F4}");

        Assert.True(alone < 0.5, $"features alone scored {alone:F4}, so the test proves nothing");
        Assert.True(defaults > 0.9, $"message passing scored only {defaults:F4}");
    }

    /// <summary>
    /// Refining a damaged labelling: some samples filed wrongly and some not filed
    /// at all. The neighbours put most of the wrong ones right, fill in the
    /// unfiled ones, and report exactly who moved.
    /// </summary>
    [Fact]
    public void RefinementRepairsAndCompletesADamagedLabelling()
    {
        var (graph, _, truth) = Synthetic.Communities();
        var rng = new Random(4);

        var damaged = (int[])truth.Clone();
        for (int i = 0; i < damaged.Length; i++)
        {
            double roll = rng.NextDouble();
            if (roll < 0.15)
                damaged[i] = (truth[i] + 1 + rng.Next(3)) % 4;
            else if (roll < 0.25)
                damaged[i] = -1;
        }

        var refined = MessagePassing.Refine(damaged, graph);

        double Accuracy(int[] labels) => labels.Zip(truth, (l, t) => l == t ? 1.0 : 0.0).Average();

        _output.WriteLine($"accuracy before {Accuracy(damaged):F4}, after {Accuracy(refined.Labels):F4}, "
                          + $"{refined.Changed().Length} changed, {refined.Unreached().Length} unreached");

        Assert.Empty(refined.Unreached());
        Assert.True(Accuracy(refined.Labels) > 0.97, $"refined accuracy only {Accuracy(refined.Labels):F4}");
        Assert.Equal(
            Enumerable.Range(0, truth.Length).Where(i => damaged[i] != refined.Labels[i]).ToArray(),
            refined.Changed());
    }

    /// <summary>
    /// Refinement adjusts an answer somebody already holds references to, so it
    /// must keep their numbering rather than impose the canonical one.
    /// </summary>
    [Fact]
    public void RefinementKeepsTheCallersNumbering()
    {
        var (graph, _, truth) = Synthetic.Communities();
        int[] permutation = { 2, 0, 3, 1 };
        var renumbered = truth.Select(t => permutation[t]).ToArray();

        var refined = MessagePassing.Refine(renumbered, graph);
        Assert.Equal(renumbered, refined.Labels);
    }

    /// <summary>
    /// A sample with no labelled sample within reach stays unplaced, and says so,
    /// rather than being given a label the data never supported.
    /// </summary>
    [Fact]
    public void SamplesOutOfReachStayUnplaced()
    {
        var graph = WeightedGraph.FromEdges(6, new[] { (0, 1), (1, 2), (3, 4), (4, 5) });
        var refined = MessagePassing.Refine(new[] { 0, -1, 1, -1, -1, -1 }, graph);

        Assert.Equal(new[] { 3, 4, 5 }, refined.Unreached());
        Assert.Equal(0.0, refined.Confidence[4]);
        Assert.Equal(0, refined.Labels[0]);
        Assert.Equal(1, refined.Labels[2]);
    }
}
