namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Settings for <see cref="ValueBands"/>. Every one has a default.
/// </summary>
public sealed record ValueBandsOptions
{
    /// <summary>
    /// Zero for an ordinary number line. Above zero, the values lie on a circle of
    /// this period — 180 for the direction of a line in degrees, 360 for a heading —
    /// so the top of the range sits beside the bottom, and a band may wrap across it.
    /// </summary>
    public double Period { get; init; }

    /// <summary>
    /// Gaps no wider than this never split a band, however few values there are:
    /// values this close are the same value. Zero by default; a caller measuring
    /// geometry passes its document tolerance.
    /// </summary>
    public double Resolution { get; init; }

    /// <summary>
    /// Below this many values the gap test is not trusted, and only values further
    /// apart than <see cref="Resolution"/> are split. A handful of points can throw
    /// up a gap that clears the bar by chance; a dozen is comfortably past that.
    /// </summary>
    public int MinimumPopulation { get; init; } = 12;

    /// <summary>
    /// A gap is a break once it is this many times the width an evenly spread
    /// population of the same size would leave between neighbours. A real
    /// population clusters, which leaves most gaps inside a band well under that
    /// width and only the gaps between bands near or above it, so two is a
    /// comfortable margin over the ordinary noise inside one band.
    /// </summary>
    public double GapMultiple { get; init; } = 2.0;

    /// <summary>
    /// A value is an outlier in its band when it sits further from the band's
    /// median than this many robust standard deviations (1.4826 x the median
    /// absolute deviation) — the Hampel identifier, with its conventional three.
    /// Robust because the median and its deviation ignore the very values being
    /// looked for, up to half the band.
    /// </summary>
    public double OutlierMultiple { get; init; } = 3.0;

    /// <summary>Checks these settings.</summary>
    public void Validate()
    {
        if (!double.IsFinite(Period) || Period < 0.0)
            throw new ArgumentOutOfRangeException(nameof(Period), Period, "Period must be zero, for a number line, or above zero.");
        if (!double.IsFinite(Resolution) || Resolution < 0.0)
            throw new ArgumentOutOfRangeException(nameof(Resolution), Resolution, "Resolution must be zero or above.");
        if (MinimumPopulation < 2)
            throw new ArgumentOutOfRangeException(nameof(MinimumPopulation), MinimumPopulation, "Need at least two values to find a gap.");
        if (!(GapMultiple > 0.0))
            throw new ArgumentOutOfRangeException(nameof(GapMultiple), GapMultiple, "Gap multiple must be above zero.");
        if (!(OutlierMultiple > 0.0))
            throw new ArgumentOutOfRangeException(nameof(OutlierMultiple), OutlierMultiple, "Outlier multiple must be above zero.");
    }
}
