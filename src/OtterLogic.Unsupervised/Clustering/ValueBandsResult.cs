namespace OtterLogic.Unsupervised.Clustering;

/// <summary>One band of values.</summary>
/// <param name="Members">Indices of the values in it, ascending.</param>
/// <param name="Median">
/// Its median — what the band is, unmoved by a stray member. On a circle, within
/// zero and the period.
/// </param>
/// <param name="Low">Its lowest value. On a circle, where the band starts going round.</param>
/// <param name="High">Its highest value. On a circle, where it ends — below <paramref name="Low"/> when it wraps.</param>
/// <param name="Spread">
/// Robust standard deviation: 1.4826 x the median absolute deviation from the
/// median. Zero when more than half the members share one value exactly.
/// </param>
public sealed record ValueBand(int[] Members, double Median, double Low, double High, double Spread)
{
    /// <summary>Number of values in the band.</summary>
    public int Count => Members.Length;
}

/// <summary>
/// The bands a population of values falls into, which band each value is in,
/// and which values sit apart from the rest of their own band.
/// </summary>
public sealed class ValueBandsResult
{
    internal ValueBandsResult(int[] band, IReadOnlyList<ValueBand> bands, double[] deviation, int[] outliers, bool learned)
    {
        Band = band;
        Bands = bands;
        Deviation = deviation;
        Outliers = outliers;
        Learned = learned;
    }

    /// <summary>Band per value, numbered by median ascending.</summary>
    public int[] Band { get; }

    /// <summary>The bands, lowest median first.</summary>
    public IReadOnlyList<ValueBand> Bands { get; }

    /// <summary>
    /// Per value, how far it sits from its band's median, signed — positive above.
    /// On a circle, the shorter way round.
    /// </summary>
    public double[] Deviation { get; }

    /// <summary>
    /// Values further from their band's median than the band's own scatter
    /// explains, largest deviation first.
    /// </summary>
    public int[] Outliers { get; }

    /// <summary>
    /// Whether the bands came from the gap test. False when there were too few
    /// values to trust it, and only values further apart than the resolution
    /// were split.
    /// </summary>
    public bool Learned { get; }

    /// <summary>Number of values.</summary>
    public int Count => Band.Length;
}
