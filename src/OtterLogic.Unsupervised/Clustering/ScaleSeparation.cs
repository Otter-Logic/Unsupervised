namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Finds the values that sit apart from the rest of a population by more than the
/// rest spans — a group on a different scale, not merely a different value.
/// <para>
/// <see cref="ValueBands"/> answers "where does this population gather", and on data
/// full of exact repeats that rightly splits every distinct value into its own band:
/// lengths of exactly 4000, 6000 and 8000 are three bands. That is the wrong answer
/// to "is anything here unusual", where the three are one ordinary spread and only a
/// value a thousand times smaller is not. This asks the second question with one
/// comparison: split the sorted distinct values at the widest gap that is wider than
/// the ordinary side's range. A group further from the rest than the rest ranges is on
/// another scale.
/// </para>
/// <para>
/// The ordinary side's range is estimated, not read off, because the span of the values
/// in hand understates the range they come from — badly when there are few. Ratios drawn
/// from a handful of exactly repeated quantities are three distinct values spanning a
/// tenth of a decade; against that raw span, a ratio of two thirds reads as a separate
/// scale, and in the same data perturbed by rounding, which adds distinct values, it does
/// not. A result that depends on rounding is not a result. So the span of m distinct
/// ordinary values is widened by (m + 1) / (m − 1) — the unbiased estimate of a range
/// from m samples — which is three for two values, one and a half for five, and barely
/// more than one for fifty. Few values earn a wide margin; many earn none. Measured: that
/// ratio of two thirds separates by 0.18 decades against an estimated range of 0.38; a
/// group fifty-one ordinary values sit above by 1.64, against 1.25; a value two thousand
/// times smaller than the rest by 3.1, against 0.6.
/// </para>
/// <para>
/// "The rest" must be the rest: the ordinary side has to hold more of the
/// population, repeats counted, than the side flagged. Without that, a lone value at
/// one end is an ordinary side of span zero, every gap beats it, and an evenly
/// spread population would have its whole remainder flagged.
/// </para>
/// <para>
/// Distinct values, not every value, so a thousand repeats of one length weigh the
/// same as one, and a population of two distinct values can still be read. Take the
/// logarithm first for anything multiplicative — lengths, ratios — so a scale means
/// a factor rather than a difference.
/// </para>
/// </summary>
public static class ScaleSeparation
{
    /// <summary>
    /// Indices of the values separated <em>below</em> the rest — the rest being the
    /// ordinary side whose own span the gap must exceed. Empty when nothing is.
    /// </summary>
    public static int[] Below(IReadOnlyList<double> values) => Separate(values, below: true);

    /// <summary>Indices of the values separated <em>above</em> the rest. Empty when nothing is.</summary>
    public static int[] Above(IReadOnlyList<double> values) => Separate(values, below: false);

    private static int[] Separate(IReadOnlyList<double> values, bool below)
    {
        ArgumentNullException.ThrowIfNull(values);
        for (int i = 0; i < values.Count; i++)
            if (!double.IsFinite(values[i]))
                throw new ArgumentException($"Value {i} is {values[i]}; values must be finite.", nameof(values));

        var distinct = values.Distinct().OrderBy(v => v).ToArray();
        int m = distinct.Length;
        if (m < 2)
            return Array.Empty<int>();

        // How many values sit at or below each distinct value, repeats counted.
        var atOrBelow = new int[m];
        var counts = values.GroupBy(v => v).ToDictionary(g => g.Key, g => g.Count());
        for (int t = 0, running = 0; t < m; t++)
            atOrBelow[t] = running += counts[distinct[t]];
        int n = values.Count;

        // For each split between distinct values t and t + 1, the ordinary side is
        // the one not being flagged; its span is what the gap must beat.
        double bestGap = 0.0;
        double boundary = double.NaN;

        for (int t = 0; t < m - 1; t++)
        {
            double gap = distinct[t + 1] - distinct[t];
            double ordinarySpan = below ? distinct[m - 1] - distinct[t + 1] : distinct[t] - distinct[0];
            int ordinaryValues = below ? m - 1 - t : t + 1;

            // One distinct ordinary value has no range to estimate: every other value is
            // off it, and the majority test alone decides.
            double range = ordinaryValues > 1 ? ordinarySpan * (ordinaryValues + 1) / (ordinaryValues - 1) : 0.0;

            int flagged = below ? atOrBelow[t] : n - atOrBelow[t];
            bool ordinaryIsTheRest = n - flagged > flagged;

            if (ordinaryIsTheRest && gap > range && gap > bestGap)
            {
                bestGap = gap;
                boundary = below ? distinct[t] : distinct[t + 1];
            }
        }

        if (double.IsNaN(boundary))
            return Array.Empty<int>();

        return Enumerable.Range(0, values.Count)
            .Where(i => below ? values[i] <= boundary : values[i] >= boundary)
            .ToArray();
    }
}
