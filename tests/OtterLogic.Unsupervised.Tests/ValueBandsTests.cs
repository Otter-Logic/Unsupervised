using OtterLogic.Unsupervised.Clustering;
using Xunit;

namespace OtterLogic.Unsupervised.Tests;

public class ValueBandsTests
{
    private static double[] Repeat(double value, int count) => Enumerable.Repeat(value, count).ToArray();

    /// <summary>
    /// Floor heights: three tight bands, and one value 25 mm off the middle one. The
    /// 25 mm gap is nothing beside the gaps between bands, so it joins the band — and
    /// is then the one value in it not where the rest are.
    /// </summary>
    [Fact]
    public void BandsTheLevelsAndFlagsTheValueOffOne()
    {
        var values = Repeat(0, 12).Concat(Repeat(4000, 30)).Concat(new[] { 4025.0 }).Concat(Repeat(8000, 30)).ToArray();

        var result = ValueBands.Fit(values);

        Assert.True(result.Learned);
        Assert.Equal(new[] { 0.0, 4000.0, 8000.0 }, result.Bands.Select(b => b.Median));
        Assert.Equal(1, result.Band[42]);
        Assert.Equal(new[] { 42 }, result.Outliers);
        Assert.Equal(25.0, result.Deviation[42], 12);
    }

    /// <summary>Two identical values far from fifty-eight others keep a band of their own.</summary>
    [Fact]
    public void ASmallBandIsNotSmothered()
    {
        var values = Enumerable.Range(0, 58).Select(i => i / 58.0).Concat(new[] { 100.0, 100.0 }).ToArray();

        var result = ValueBands.Fit(values);

        Assert.Equal(2, result.Bands.Count);
        Assert.Equal(2, result.Bands[1].Count);
    }

    /// <summary>A ramp — values spread evenly with no gap — is one band, not an invented break.</summary>
    [Fact]
    public void AnEvenSpreadIsOneBand()
    {
        var result = ValueBands.Fit(Enumerable.Range(0, 50).Select(i => i * 10.0).ToArray());
        Assert.Single(result.Bands);
    }

    /// <summary>
    /// Directions of lines, modulo 180: 179.8 and 0.2 are the same direction, and
    /// the band they share straddles the seam.
    /// </summary>
    [Fact]
    public void OnACircleABandWrapsTheSeam()
    {
        var values = new[] { 0.2, 179.8, 0.0, 179.9, 0.1, 0.3, 90.0, 90.1, 89.9, 90.2, 89.8, 90.0, 0.0, 179.95 };

        var result = ValueBands.Fit(values, new ValueBandsOptions { Period = 180 });

        Assert.Equal(2, result.Bands.Count);
        Assert.Equal(result.Band[0], result.Band[1]);
        Assert.NotEqual(result.Band[0], result.Band[6]);
        Assert.Equal(-0.2, result.Deviation[1] - result.Deviation[2], 9);

        var seam = result.Bands[result.Band[0]];
        Assert.True(seam.Low > seam.High, "a band across the seam starts above where it ends");
    }

    [Fact]
    public void OneDirectionOnACircleIsOneBand()
    {
        var result = ValueBands.Fit(Repeat(45, 20), new ValueBandsOptions { Period = 180 });

        Assert.Single(result.Bands);
        Assert.Equal(45.0, result.Bands[0].Median, 12);
    }

    /// <summary>
    /// Too few values to trust a gap: only values further apart than the resolution
    /// split, and the result says the bands were not learned.
    /// </summary>
    [Fact]
    public void ASmallPopulationSplitsOnlyAtTheResolution()
    {
        var result = ValueBands.Fit(new[] { 0.0, 0.0005, 6000.0, 6000.0, 12000.0 }, new ValueBandsOptions { Resolution = 0.001 });

        Assert.False(result.Learned);
        Assert.Equal(new[] { 0.0, 6000.0, 12000.0 }, result.Bands.Select(b => Math.Round(b.Median)));
    }

    [Fact]
    public void GapsWithinTheResolutionNeverSplit()
    {
        var values = Repeat(0, 10).Concat(Repeat(0.0004, 10)).ToArray();
        var result = ValueBands.Fit(values, new ValueBandsOptions { Resolution = 0.001 });

        Assert.Single(result.Bands);
        Assert.Empty(result.Outliers);
    }

    [Fact]
    public void BandsAreNumberedByMedian()
    {
        var values = Repeat(900, 10).Concat(Repeat(100, 10)).Concat(Repeat(500, 10)).ToArray();
        var result = ValueBands.Fit(values);

        Assert.Equal(new[] { 100.0, 500.0, 900.0 }, result.Bands.Select(b => b.Median));
        Assert.Equal(2, result.Band[0]);
    }

    [Fact]
    public void NonFiniteValuesAreRefused()
    {
        Assert.Throws<ArgumentException>(() => ValueBands.Fit(new[] { 1.0, double.NaN }));
    }

    [Fact]
    public void ANegativePeriodIsRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ValueBands.Fit(new[] { 1.0, 2.0 }, new ValueBandsOptions { Period = -1 }));
    }
}
