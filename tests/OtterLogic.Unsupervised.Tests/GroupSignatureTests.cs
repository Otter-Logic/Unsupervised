using OtterLogic.Unsupervised.Clustering;
using Xunit;

namespace OtterLogic.Unsupervised.Tests;

public class GroupSignatureTests
{
    private static double[,] Column(params double[] values)
    {
        var data = new double[values.Length, 1];
        for (int i = 0; i < values.Length; i++)
            data[i, 0] = values[i];
        return data;
    }

    /// <summary>
    /// Worked by hand. Group {1, 2, 3} against rest {2, 4, 5, 6}: of the twelve
    /// member-by-non-member pairs, one has the member above (3 over 2), ten below,
    /// and one tie (2 against 2) that counts neither way, so the delta is
    /// (1 − 10) / 12 = −0.75.
    /// </summary>
    [Fact]
    public void SeparationMatchesCliffsDeltaByHand()
    {
        var result = GroupSignature.Describe(Column(1, 2, 3, 2, 4, 5, 6), new[] { 0, 0, 0, 1, 1, 1, 1 });

        Assert.Equal(-0.75, result.Separation[0, 0], 12);
        Assert.Equal(0.75, result.Separation[1, 0], 12);
    }

    /// <summary>
    /// By hand: means 2 and 4.25, variances 1 and 8.75/3, pooled
    /// sqrt((2·1 + 3·8.75/3) / 5) = sqrt(2.15), so d = −2.25 / sqrt(2.15).
    /// </summary>
    [Fact]
    public void EffectMatchesCohensDByHand()
    {
        var result = GroupSignature.Describe(Column(1, 2, 3, 2, 4, 5, 6), new[] { 0, 0, 0, 1, 1, 1, 1 });

        Assert.Equal(-2.25 / Math.Sqrt(2.15), result.Effect[0, 0], 12);
        Assert.Equal(2.0, result.Means[0, 0], 12);
        Assert.Equal(4.25, result.RestMeans[0, 0], 12);
        Assert.Equal(1.0, result.Spread[0, 0], 12);
    }

    /// <summary>
    /// The feature that splits the groups apart ranks first and reads a full
    /// separation; the feature that is noise for both groups ranks last.
    /// </summary>
    [Fact]
    public void TheSeparatingFeatureRanksFirst()
    {
        var rng = new Random(11);
        int n = 60;
        var data = new double[n, 3];
        var labels = new int[n];

        for (int i = 0; i < n; i++)
        {
            labels[i] = i < 20 ? 0 : 1;
            data[i, 0] = rng.NextDouble();                                    // noise
            data[i, 1] = (labels[i] == 0 ? 10.0 : 0.0) + rng.NextDouble();    // separates
            data[i, 2] = (labels[i] == 0 ? 0.6 : 0.4) + rng.NextDouble();     // leans
        }

        var result = GroupSignature.Describe(data, labels);

        Assert.Equal(new[] { 1, 2, 0 }, result.Ranking[0]);
        Assert.Equal(1.0, result.Separation[0, 1], 12);
        Assert.Equal(-1.0, result.Separation[1, 1], 12);
        Assert.True(Math.Abs(result.Separation[0, 0]) < 0.3, $"noise separated by {result.Separation[0, 0]}");
    }

    /// <summary>
    /// Neither measure depends on a column's units: millimetres and metres
    /// describe the same group identically, which is what lets features in
    /// different units be ranked against each other at all.
    /// </summary>
    [Fact]
    public void ScalingAColumnChangesNothing()
    {
        double[] values = { 3.1, 2.7, 4.4, 9.0, 8.2, 7.7, 9.9, 1.2 };
        int[] labels = { 0, 0, 0, 1, 1, 1, 1, 0 };

        var metres = GroupSignature.Describe(Column(values), labels);
        var millimetres = GroupSignature.Describe(Column(values.Select(v => v * 1000.0).ToArray()), labels);

        Assert.Equal(metres.Separation[0, 0], millimetres.Separation[0, 0], 12);
        Assert.Equal(metres.Effect[0, 0], millimetres.Effect[0, 0], 9);
    }

    /// <summary>
    /// A small minority reads as fully separated however many samples sit on the
    /// other side — two identical values against fifty-eight lower ones.
    /// </summary>
    [Fact]
    public void ASmallGroupIsNotSmotheredByALargeOne()
    {
        var values = Enumerable.Range(0, 58).Select(i => i / 58.0).Concat(new[] { 5.0, 5.0 }).ToArray();
        var labels = Enumerable.Repeat(0, 58).Concat(new[] { 1, 1 }).ToArray();

        var result = GroupSignature.Describe(Column(values), labels);

        Assert.Equal(1.0, result.Separation[1, 0], 12);
        Assert.Equal(2, result.Sizes[1]);
    }

    /// <summary>
    /// Each group constant and different: no spread within either side, so the
    /// spread over every sample stands in and the effect is finite.
    /// </summary>
    [Fact]
    public void ConstantGroupsGiveAFiniteEffect()
    {
        var result = GroupSignature.Describe(Column(0, 0, 0, 1, 1, 1), new[] { 0, 0, 0, 1, 1, 1 });

        Assert.True(double.IsFinite(result.Effect[0, 0]));
        Assert.True(result.Effect[0, 0] < 0.0);
        Assert.Equal(-1.0, result.Separation[0, 0], 12);
    }

    [Fact]
    public void AConstantFeatureSeparatesNothing()
    {
        var result = GroupSignature.Describe(Column(4, 4, 4, 4), new[] { 0, 0, 1, 1 });

        Assert.Equal(0.0, result.Separation[0, 0], 12);
        Assert.Equal(0.0, result.Effect[0, 0], 12);
    }

    /// <summary>
    /// Unplaced samples are part of "the rest" for every group, but get no row of
    /// their own: the group's mean is compared against them too.
    /// </summary>
    [Fact]
    public void UnplacedSamplesCountAsRestButGetNoSignature()
    {
        var result = GroupSignature.Describe(Column(1, 1, 5, 5, 100), new[] { 0, 0, 1, 1, -1 });

        Assert.Equal(2, result.Groups);
        Assert.Equal((5 + 5 + 100) / 3.0, result.RestMeans[0, 0], 12);
    }

    /// <summary>A label with no samples keeps its row, so position stays the label.</summary>
    [Fact]
    public void AnEmptyLabelKeepsItsRow()
    {
        var result = GroupSignature.Describe(Column(1, 2, 8, 9), new[] { 0, 0, 2, 2 });

        Assert.Equal(3, result.Groups);
        Assert.Equal(0, result.Sizes[1]);
        Assert.DoesNotContain("Group 1 ", result.Summary());
    }

    [Fact]
    public void OneGroupHoldingEverySampleIsRefused()
    {
        Assert.Throws<ArgumentException>(() => GroupSignature.Describe(Column(1, 2, 3), new[] { 0, 0, 0 }));
    }

    [Fact]
    public void NothingPlacedIsRefused()
    {
        Assert.Throws<ArgumentException>(() => GroupSignature.Describe(Column(1, 2, 3), new[] { -1, -1, -1 }));
    }

    [Fact]
    public void SummaryNamesTheStrongestFeatureWithItsDirection()
    {
        var data = new double[6, 2];
        double[] length = { 9, 9.5, 10, 1, 1.5, 2 };
        double[] depth = { 3, 1, 2, 2, 3, 1 };
        for (int i = 0; i < 6; i++)
        {
            data[i, 0] = depth[i];
            data[i, 1] = length[i];
        }

        string summary = GroupSignature.Describe(data, new[] { 0, 0, 0, 1, 1, 1 })
            .Summary(new[] { "Depth", "Length" }, top: 1);

        Assert.Contains("Group 0 — 3 of 6 samples", summary);
        Assert.Contains("Length  higher", summary);
        Assert.Contains("large", summary);
        Assert.DoesNotContain("Depth", summary);
    }

    [Theory]
    [InlineData(0.10, "negligible")]
    [InlineData(-0.20, "small")]
    [InlineData(0.40, "medium")]
    [InlineData(-0.90, "large")]
    public void MagnitudeFollowsRomanosConventions(double separation, string word)
    {
        Assert.Equal(word, GroupSignatureResult.Magnitude(separation));
    }
}
