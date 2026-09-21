using OtterLogic.MachineLearning.Graphs;
using OtterLogic.Unsupervised.Clustering;
using Xunit;

namespace OtterLogic.Unsupervised.Tests;

public class NeighbourhoodProfileTests
{
    /// <summary>
    /// Two stars that share nothing: each hub has the same profile as the other hub and
    /// each spoke as every other spoke, which no community method could say, since
    /// nothing connects the two stars at all.
    /// </summary>
    [Fact]
    public void SamplesPlayingTheSamePartFarApartGetTheSameProfile()
    {
        var edges = new List<(int, int)>();
        for (int s = 1; s <= 4; s++)
        {
            edges.Add((0, s));
            edges.Add((5, 5 + s));
        }

        var graph = WeightedGraph.FromEdges(10, edges);
        var x = new double[10, 1];
        x[0, 0] = x[5, 0] = 3.0;
        for (int s = 1; s <= 4; s++)
            x[s, 0] = x[5 + s, 0] = -1.0;

        var profile = NeighbourhoodProfile.Embed(x, graph);

        Assert.Equal(3, profile.GetLength(1));
        for (int c = 0; c < 3; c++)
        {
            Assert.Equal(profile[0, c], profile[5, c], 12);
            Assert.Equal(profile[1, c], profile[8, c], 12);
        }

        // A spoke's surroundings are the hub, scaled by the decay; a hub's are its spokes.
        Assert.Equal(0.5 * 3.0, profile[1, 1], 12);
        Assert.Equal(0.5 * -1.0, profile[0, 1], 12);
    }

    /// <summary>Own features are kept as they were, not blended — that is the difference from message passing.</summary>
    [Fact]
    public void OwnFeaturesAreKeptUnblended()
    {
        var graph = WeightedGraph.FromEdges(3, new[] { (0, 1), (1, 2) });
        var x = new double[,] { { 1.0, 2.0 }, { -4.0, 0.5 }, { 7.0, 7.0 } };

        var profile = NeighbourhoodProfile.Embed(x, graph, hops: 1);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
                Assert.Equal(x[i, j], profile[i, j]);
    }

    /// <summary>The profile view is off unless asked for, and votes when it is.</summary>
    [Fact]
    public void TheProfileViewVotesOnlyWhenGivenAWeight()
    {
        var (graph, x) = Ladder(12);

        var without = MultiViewClustering.Fit(graph, x, x, x);
        var with = MultiViewClustering.Fit(graph, x, x, x, new MultiViewClusteringOptions { ProfileWeight = 1.0 });

        Assert.Null(without.ProfileLabels);
        Assert.DoesNotContain(without.Views, view => view.Name == "Profile");
        Assert.NotNull(with.ProfileLabels);
        Assert.Contains(with.Views, view => view.Name == "Profile");
    }

    /// <summary>Two rails joined by rungs: rails are long, rungs are short.</summary>
    private static (WeightedGraph Graph, double[,] X) Ladder(int rungs)
    {
        int n = 3 * rungs;
        var edges = new List<(int, int)>();
        var x = new double[n, 2];

        for (int r = 0; r < rungs; r++)
        {
            int top = 3 * r, bottom = 3 * r + 1, rung = 3 * r + 2;
            (x[top, 0], x[top, 1]) = (2.0, 1.0);
            (x[bottom, 0], x[bottom, 1]) = (2.0, -1.0);
            (x[rung, 0], x[rung, 1]) = (-1.0, 0.0);

            edges.Add((top, rung));
            edges.Add((bottom, rung));
            if (r > 0)
            {
                edges.Add((top, top - 3));
                edges.Add((bottom, bottom - 3));
            }
        }

        return (WeightedGraph.FromEdges(n, edges), x);
    }
}
