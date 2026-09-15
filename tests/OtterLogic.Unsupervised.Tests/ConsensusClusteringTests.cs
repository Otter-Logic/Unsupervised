using OtterLogic.MachineLearning.Graphs;
using OtterLogic.Unsupervised.Clustering;
using Xunit;

namespace OtterLogic.Unsupervised.Tests;

/// <summary>
/// Fusion is tested against labellings whose consensus is known by construction:
/// the truth, damaged differently in each view, so no one view is right but the
/// majority always is.
/// </summary>
public class ConsensusClusteringTests
{
    private const int PerGroup = 40;

    [Fact]
    public void ViewsWrongInDifferentPlaces_RecoverTheTruth()
    {
        var truth = Truth();
        var views = Enumerable.Range(0, 3).Select(v => new ClusterView($"view {v}", Damaged(truth, v))).ToArray();

        var result = ConsensusClustering.Fuse(views);

        Assert.Equal(3, result.Groups);
        Assert.Equal(1.0, ClusterAgreement.AdjustedRand(truth, result.Labels), 12);
        Assert.All(views, view => Assert.True(ClusterAgreement.AdjustedRand(truth, view.Labels) < 1.0));
    }

    /// <summary>
    /// The samples a view got wrong are exactly the ones the consensus is least sure
    /// of. By construction an undamaged sample agrees fully with the 27 other
    /// undamaged samples of its group and two-thirds with the 12 damaged ones —
    /// 35/39 — and a damaged sample reads 24.33/39.
    /// </summary>
    [Fact]
    public void AgreementIsLowestWhereTheViewsSplit()
    {
        var truth = Truth();
        var views = Enumerable.Range(0, 3).Select(v => new ClusterView($"view {v}", Damaged(truth, v))).ToArray();

        var result = ConsensusClustering.Fuse(views);

        for (int i = 0; i < truth.Length; i++)
        {
            double expected = i % 10 < 3 ? (28.0 * 2.0 / 3.0 + 3.0 + 8.0 / 3.0) / 39.0 : 35.0 / 39.0;
            Assert.Equal(expected, result.Agreement[i], 12);
        }
    }

    [Fact]
    public void RenumberingAViewChangesNothing()
    {
        var truth = Truth();
        var views = Enumerable.Range(0, 3).Select(v => new ClusterView($"view {v}", Damaged(truth, v))).ToArray();
        var renumbered = views.Select(v => v with { Labels = v.Labels.Select(l => 2 - l).ToArray() }).ToArray();

        Assert.Equal(ConsensusClustering.Fuse(views).Labels, ConsensusClustering.Fuse(renumbered).Labels);
    }

    [Fact]
    public void AViewWeightedZeroHasNoVote()
    {
        var truth = Truth();
        var noise = Enumerable.Range(0, truth.Length).Select(i => i % 7).ToArray();
        var views = new[]
        {
            new ClusterView("truth", truth),
            new ClusterView("noise", noise, Weight: 0.0),
        };

        var result = ConsensusClustering.Fuse(views);

        Assert.Equal(1.0, ClusterAgreement.AdjustedRand(truth, result.Labels), 12);
        Assert.Equal(0.0, result.Weights[1]);
    }

    /// <summary>
    /// A view that disagrees with every other loses weight when weighting by
    /// agreement, and keeps it when that is switched off.
    /// </summary>
    [Fact]
    public void ADissentingViewIsOutvoted()
    {
        var truth = Truth();
        var dissent = Enumerable.Range(0, truth.Length).Select(i => i % 3).ToArray();
        var views = new[]
        {
            new ClusterView("a", Damaged(truth, 0)),
            new ClusterView("b", Damaged(truth, 1)),
            new ClusterView("dissent", dissent),
        };

        var weighted = ConsensusClustering.Fuse(views);
        var flat = ConsensusClustering.Fuse(views, options: new ConsensusOptions { WeightByAgreement = false });

        Assert.True(weighted.Weights[2] < 0.5 * weighted.Weights[0],
            $"the dissenting view kept {weighted.Weights[2]:0.00} of the vote against {weighted.Weights[0]:0.00}");
        Assert.Equal(1.0 / 3.0, flat.Weights[2], 12);
    }

    /// <summary>An unplaced sample abstains, so leaving samples out of one view does not pull them apart.</summary>
    [Fact]
    public void UnplacedSamplesAbstain()
    {
        var truth = Truth();
        var partial = truth.Select((label, i) => i % 3 == 0 ? -1 : label).ToArray();

        var result = ConsensusClustering.Fuse(new[] { new ClusterView("full", truth), new ClusterView("partial", partial) });

        Assert.Equal(1.0, ClusterAgreement.AdjustedRand(truth, result.Labels), 12);
        Assert.All(result.Agreement, a => Assert.Equal(1.0, a, 12));
    }

    [Fact]
    public void AFixedCountIsKept()
    {
        var truth = Truth();
        var result = ConsensusClustering.Fuse(new[] { new ClusterView("truth", truth) }, options: new ConsensusOptions { Groups = 2 });

        Assert.Equal(2, result.Groups);
        Assert.Single(result.Sweep);
    }

    /// <summary>
    /// A group under the minimum size merges into the group it is connected to,
    /// even when nothing else prefers that group — the graph decides the tie.
    /// </summary>
    [Fact]
    public void SmallGroupsMergeIntoAConnectedGroup()
    {
        // Two groups of forty and one of two, each group entirely apart in every view.
        var labels = Enumerable.Range(0, 82).Select(i => i < 40 ? 0 : i < 80 ? 1 : 2).ToArray();
        var views = new[] { new ClusterView("a", labels), new ClusterView("b", labels) };

        // The pair touches the second group only.
        var edges = Enumerable.Range(0, 79).Where(i => i != 39).Select(i => (i, i + 1)).Append((80, 81)).Append((81, 79));
        var graph = WeightedGraph.FromEdges(82, edges);

        var result = ConsensusClustering.Fuse(views, graph, new ConsensusOptions { Groups = 3, MinimumGroupSize = 3 });

        Assert.Equal(1, result.MergedGroups);
        Assert.Equal(2, result.Groups);
        Assert.Equal(result.Labels[79], result.Labels[80]);
        Assert.NotEqual(result.Labels[0], result.Labels[80]);
    }

    /// <summary>Samples sharing a label in every view are fused once, not once each.</summary>
    [Fact]
    public void WorksOnSignaturesNotSamples()
    {
        var truth = Truth();
        var result = ConsensusClustering.Fuse(new[] { new ClusterView("a", truth), new ClusterView("b", truth) });

        Assert.Equal(3, result.Signatures);
    }

    [Fact]
    public void SameInputTwice_GivesTheSameAnswer()
    {
        var truth = Truth();
        var views = Enumerable.Range(0, 3).Select(v => new ClusterView($"view {v}", Damaged(truth, v))).ToArray();

        var first = ConsensusClustering.Fuse(views);
        var second = ConsensusClustering.Fuse(views);

        Assert.Equal(first.Labels, second.Labels);
        Assert.Equal(first.Agreement, second.Agreement);
    }

    [Fact]
    public void ViewsOfDifferentLengths_FailWithSomethingReadable()
    {
        var error = Assert.Throws<ArgumentException>(() => ConsensusClustering.Fuse(new[]
        {
            new ClusterView("long", new[] { 0, 0, 1 }),
            new ClusterView("short", new[] { 0, 1 }),
        }));

        Assert.Contains("same samples", error.Message);
    }

    private static int[] Truth() => Enumerable.Range(0, 3 * PerGroup).Select(i => i / PerGroup).ToArray();

    /// <summary>The truth with every sample whose index ends in <paramref name="view"/> moved to the next group.</summary>
    private static int[] Damaged(int[] truth, int view)
        => truth.Select((label, i) => i % 10 == view ? (label + 1) % 3 : label).ToArray();
}
