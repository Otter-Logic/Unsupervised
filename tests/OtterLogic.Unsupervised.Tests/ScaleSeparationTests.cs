using OtterLogic.Unsupervised.Clustering;
using Xunit;

namespace OtterLogic.Unsupervised.Tests;

public class ScaleSeparationTests
{
    /// <summary>
    /// Lengths of exactly 4000, 6000 and 8000, many times over, and one of 3: on a
    /// log scale the ordinary three span 0.3 decades and the sliver sits 3.1 decades
    /// below them — separated. The three ordinary lengths are not, however many
    /// repeats make them look like separate bands.
    /// </summary>
    [Fact]
    public void ASliverIsSeparatedAndOrdinaryRepeatsAreNot()
    {
        var lengths = Enumerable.Repeat(4000.0, 30).Concat(Enumerable.Repeat(6000.0, 40))
            .Concat(Enumerable.Repeat(8000.0, 50)).Concat(new[] { 3.0 }).Select(Math.Log10).ToArray();

        Assert.Equal(new[] { 120 }, ScaleSeparation.Below(lengths));
        Assert.Empty(ScaleSeparation.Above(lengths));
    }

    [Fact]
    public void ContinuousSpreadHasNothingSeparated()
    {
        var values = Enumerable.Range(0, 40).Select(i => i * 0.1).ToArray();

        Assert.Empty(ScaleSeparation.Below(values));
        Assert.Empty(ScaleSeparation.Above(values));
    }

    /// <summary>
    /// Detour ratios: connected pairs between 1 and 3, near misses in the thousands.
    /// Every near miss is flagged, however far apart the near misses are from each
    /// other — only the ordinary side's span has to be beaten.
    /// </summary>
    [Fact]
    public void EveryValueBeyondTheGapIsFlagged()
    {
        var detours = new[] { 1.0, 1.4, 2.0, 2.8, 1.0, 1.2, 800.0, 90000.0 }.Select(Math.Log10).ToArray();

        Assert.Equal(new[] { 6, 7 }, ScaleSeparation.Above(detours));
    }

    [Fact]
    public void AGapNoWiderThanTheEstimatedRangeSeparatesNothing()
    {
        // Three ordinary values spanning 10 estimate a range of 10 x (3 + 1) / (3 - 1) = 20.
        // A value 15 beyond them is within it; 21 beyond is not.
        Assert.Empty(ScaleSeparation.Above(new[] { 0.0, 5.0, 10.0, 25.0 }));
        Assert.Equal(new[] { 3 }, ScaleSeparation.Above(new[] { 0.0, 5.0, 10.0, 31.0 }));
    }

    /// <summary>
    /// A regular frame's length ratios: columns two thirds of the beams they meet, beams
    /// one and a third of theirs — a handful of distinct values. Nothing is on another
    /// scale, and turning the frame (which only adds rounding) must not change that.
    /// </summary>
    [Fact]
    public void AFewDistinctOrdinaryRatiosAreNotSeparated()
    {
        var ratios = Enumerable.Repeat(2.0 / 3.0, 24).Concat(Enumerable.Repeat(1.0, 30))
            .Concat(Enumerable.Repeat(4.0 / 3.0, 20)).Select(Math.Log10).ToArray();

        Assert.Empty(ScaleSeparation.Below(ratios));
        Assert.Empty(ScaleSeparation.Above(ratios));
    }

    /// <summary>
    /// Many ordinary values measure their range well, so a gap only a little wider than
    /// their span is enough — where a fixed doubling would miss it.
    /// </summary>
    [Fact]
    public void ManyOrdinaryValuesNeedLittleMargin()
    {
        var ordinary = Enumerable.Range(0, 51).Select(i => -0.74 + 1.2 * i / 50.0);
        var values = ordinary.Concat(new[] { -2.38, -2.7, -3.08 }).ToArray();

        Assert.Equal(new[] { 51, 52, 53 }, ScaleSeparation.Below(values));
    }

    [Fact]
    public void FewerThanTwoDistinctValuesSeparateNothing()
    {
        Assert.Empty(ScaleSeparation.Above(new[] { 3.0, 3.0, 3.0 }));
        Assert.Empty(ScaleSeparation.Below(Array.Empty<double>()));
    }
}
