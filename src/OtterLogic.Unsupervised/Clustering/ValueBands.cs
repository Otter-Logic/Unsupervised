namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Splits a population of single values into the bands its own distribution
/// supports, and finds the values that sit apart from the rest of their band.
/// <para>
/// Sorts the values and breaks wherever the gap between neighbours dwarfs the
/// spacing the population would have if spread evenly over its range — a
/// comparison to the population's own spacing rather than to an absolute
/// distance. That is what lets it work the same in millimetres and metres, and
/// what keeps a genuine but tiny minority as a band of its own: the gap either side
/// of two identical values is as wide whether ninety-eight others sit beyond it or
/// two, where a distance weighted by cluster size, such as an agglomerative tree's
/// merge height, can smother a small band into its larger neighbour. A continuous
/// spread with no gap — a ramp — gives one band, not an invented break.
/// </para>
/// <para>
/// On a circle (<see cref="ValueBandsOptions.Period"/> above zero) the evenly
/// spread spacing is the period over the count, and the gap from the last value
/// round to the first counts like any other, so a band can straddle the seam: a
/// line at 179.8 degrees and one at 0.2 are the same direction.
/// </para>
/// <para>
/// Inside each band, a value is an outlier by the Hampel identifier — further from
/// the band's median than its robust scatter explains. Banding says a value
/// belongs with its neighbours; this says it is nonetheless not quite where they
/// are. Neither step is given a distance to use; both read it from the values.
/// </para>
/// </summary>
public static class ValueBands
{
    /// <summary>Scales a median absolute deviation to a standard deviation for normally distributed values.</summary>
    private const double MadScale = 1.4826;

    /// <summary>Bands <paramref name="values"/>.</summary>
    /// <param name="values">The population, at least one value, all finite.</param>
    /// <param name="options">Settings; null for the defaults.</param>
    public static ValueBandsResult Fit(IReadOnlyList<double> values, ValueBandsOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(values);
        options ??= new ValueBandsOptions();
        options.Validate();

        int n = values.Count;
        if (n == 0)
            throw new ArgumentException("Need at least one value to band.", nameof(values));

        double period = options.Period;
        bool circular = period > 0.0;

        var v = new double[n];
        for (int i = 0; i < n; i++)
        {
            if (!double.IsFinite(values[i]))
                throw new ArgumentException($"Value {i} is {values[i]}; values must be finite.", nameof(values));

            v[i] = circular ? Wrap(values[i], period) : values[i];
        }

        var order = Enumerable.Range(0, n).OrderBy(i => v[i]).ThenBy(i => i).ToArray();
        bool learned = n >= options.MinimumPopulation;

        // Gap g[t] is the one after sorted position t; on a circle the last wraps
        // round to the first.
        int gapCount = circular ? n : n - 1;
        var gap = new double[gapCount];
        for (int t = 0; t < n - 1; t++)
            gap[t] = v[order[t + 1]] - v[order[t]];
        if (circular)
            gap[n - 1] = v[order[0]] + period - v[order[n - 1]];

        double threshold;
        if (!learned)
            threshold = 0.0;
        else if (circular)
            threshold = options.GapMultiple * period / n;
        else
        {
            double range = v[order[n - 1]] - v[order[0]];
            threshold = range > 0.0 ? options.GapMultiple * range / (n - 1) : double.PositiveInfinity;
        }

        var breaks = Enumerable.Range(0, gapCount)
            .Where(t => gap[t] >= threshold && gap[t] > options.Resolution)
            .ToList();

        // Where to start walking the sorted values. On a line, the lowest. On a
        // circle, just after a break — the widest gap when there is none, so a
        // single band is unwrapped across the emptiest part of the circle.
        int start = 0;
        if (circular && n > 1)
        {
            int cut = breaks.Count > 0 ? breaks[0] : Enumerable.Range(0, n).OrderByDescending(t => gap[t]).First();
            start = (cut + 1) % n;
        }

        var breakAfter = new HashSet<int>(breaks);
        var groups = new List<List<(int Index, double Unwrapped)>>();
        var current = new List<(int, double)>();
        double offset = 0.0;

        for (int step = 0; step < n; step++)
        {
            int t = (start + step) % n;
            if (circular && step > 0 && t == 0)
                offset = period;

            current.Add((order[t], v[order[t]] + offset));

            bool last = step == n - 1;
            if (last || breakAfter.Contains(t))
            {
                groups.Add(current);
                current = new List<(int, double)>();
            }
        }

        var summaries = groups.Select(group => Summarise(group, circular, period)).ToList();
        var ranked = Enumerable.Range(0, groups.Count).OrderBy(g => summaries[g].Band.Median).ToArray();

        var band = new int[n];
        var deviation = new double[n];
        var bands = new List<ValueBand>(groups.Count);
        var outliers = new List<int>();

        for (int b = 0; b < ranked.Length; b++)
        {
            int g = ranked[b];
            var (summary, unwrappedMedian) = summaries[g];
            bands.Add(summary);

            double limit = Math.Max(options.OutlierMultiple * summary.Spread, options.Resolution);
            foreach (var (index, unwrapped) in groups[g])
            {
                band[index] = b;
                deviation[index] = unwrapped - unwrappedMedian;
                if (Math.Abs(deviation[index]) > limit)
                    outliers.Add(index);
            }
        }

        return new ValueBandsResult(
            band,
            bands,
            deviation,
            outliers.OrderByDescending(i => Math.Abs(deviation[i])).ThenBy(i => i).ToArray(),
            learned);
    }

    /// <summary>A band's summary, and its median before wrapping back into the period.</summary>
    private static (ValueBand Band, double UnwrappedMedian) Summarise(
        List<(int Index, double Unwrapped)> group, bool circular, double period)
    {
        var sorted = group.Select(member => member.Unwrapped).OrderBy(x => x).ToArray();
        double median = Median(sorted);
        double mad = Median(sorted.Select(x => Math.Abs(x - median)).OrderBy(x => x).ToArray());

        double low = sorted[0];
        double high = sorted[^1];
        double reported = median;

        if (circular)
        {
            low = Wrap(low, period);
            high = Wrap(high, period);
            reported = Wrap(median, period);
        }

        var members = group.Select(member => member.Index).OrderBy(i => i).ToArray();
        return (new ValueBand(members, reported, low, high, MadScale * mad), median);
    }

    private static double Median(double[] sorted)
    {
        int n = sorted.Length;
        return n % 2 == 1 ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    }

    private static double Wrap(double value, double period)
    {
        double wrapped = value % period;
        if (wrapped < 0.0)
            wrapped += period;
        // Rounding can land a value just below the period on exactly the period.
        return wrapped >= period ? 0.0 : wrapped;
    }
}
