namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// What sets each group of a labelling apart from everything outside it, feature
/// by feature, in numbers a person can read.
/// <para>
/// Every clustering hands back integers. A group numbered 3 means nothing until
/// somebody can say what its members share that the rest do not — and until they
/// can, they cannot name it, trust it or build on it. This is that step: for each
/// group and each feature, how far the group's values sit from the rest's, in two
/// complementary measures, and the features ranked by how strongly they separate.
/// The groups are found by a method; what they are called is left to whoever reads
/// this.
/// </para>
/// <list type="bullet">
/// <item><b>Separation</b> — Cliff's delta, −1 to 1: the chance a member exceeds a
/// non-member, less the chance it falls below one. Built from ranks alone, so it is
/// untouched by units, skew or a few extreme values, and it is bounded, so a feature
/// that splits the group off perfectly reads 1 rather than something unbounded. This
/// is what the ranking uses.</item>
/// <item><b>Effect size</b> — Cohen's d: the gap between the group's mean and the
/// rest's, in pooled standard deviations. Says <em>how far</em> apart where
/// separation says <em>how cleanly</em>, and so tells apart two groups that separate
/// equally cleanly by very different margins.</item>
/// </list>
/// <para>
/// "The rest" is every sample outside the group, unplaced ones included: the
/// question is what makes the group different from everything else, and samples a
/// method declined to place are part of everything else. They get no signature of
/// their own — to describe them, give them a label.
/// </para>
/// </summary>
public static class GroupSignature
{
    /// <summary>
    /// Describes every group of <paramref name="labels"/> against the rest.
    /// </summary>
    /// <param name="data">
    /// n x d, one row per sample, in whatever units the result should be read in —
    /// usually the raw values rather than a scaled copy, since neither measure is
    /// changed by scaling a column and the means are only readable in real units.
    /// </param>
    /// <param name="labels">Group per sample; negative is unplaced.</param>
    public static GroupSignatureResult Describe(double[,] data, int[] labels)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(labels);

        int n = data.GetLength(0);
        int d = data.GetLength(1);

        if (labels.Length != n)
            throw new ArgumentException($"{labels.Length} labels for {n} samples; give one per sample.", nameof(labels));
        if (d < 1)
            throw new ArgumentException("The data has no columns to describe.", nameof(data));

        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                if (!double.IsFinite(data[i, j]))
                    throw new ArgumentException($"data[{i}, {j}] is {data[i, j]}; values must be finite.", nameof(data));

        var members = ClusterLabels.Members(labels);
        int k = members.Length;

        if (k == 0 || members.All(m => m.Length == 0))
            throw new ArgumentException("No sample is in a group, so there is nothing to describe.", nameof(labels));

        for (int g = 0; g < k; g++)
            if (members[g].Length == n)
                throw new ArgumentException(
                    $"Every sample is in group {g}, so there is nothing outside it to compare against.", nameof(labels));

        var sizes = members.Select(m => m.Length).ToArray();
        var means = new double[k, d];
        var spread = new double[k, d];
        var restMeans = new double[k, d];
        var effect = new double[k, d];
        var separation = new double[k, d];

        var column = new double[n];
        var inGroup = new bool[n];

        for (int j = 0; j < d; j++)
        {
            for (int i = 0; i < n; i++)
                column[i] = data[i, j];

            var ranks = AverageRanks(column);
            var (_, overallSpread) = MeanAndSpread(column, Enumerable.Range(0, n));

            for (int g = 0; g < k; g++)
            {
                int m = members[g].Length;
                if (m == 0)
                    continue;

                Array.Clear(inGroup);
                foreach (int i in members[g])
                    inGroup[i] = true;

                var rest = Enumerable.Range(0, n).Where(i => !inGroup[i]);
                int r = n - m;

                var (meanIn, spreadIn) = MeanAndSpread(column, members[g]);
                var (meanRest, spreadRest) = MeanAndSpread(column, rest);

                means[g, j] = meanIn;
                spread[g, j] = spreadIn;
                restMeans[g, j] = meanRest;

                // Pooled within-side spread, Cohen's own denominator. When neither
                // side varies at all — each group constant, the groups different —
                // it is zero, and the gap is infinite in its units. The feature's
                // spread over every sample stands in there, so a feature that
                // separates perfectly reads as large rather than as infinity, which
                // nothing downstream can sort, plot or colour by.
                double pooled = m + r > 2
                    ? Math.Sqrt(((m - 1) * spreadIn * spreadIn + (r - 1) * spreadRest * spreadRest) / (m + r - 2))
                    : 0.0;
                double scale = pooled > 1e-12 * overallSpread ? pooled : overallSpread;
                effect[g, j] = scale > 0.0 ? (meanIn - meanRest) / scale : 0.0;

                // Cliff's delta through the Mann-Whitney U: the members' rank sum,
                // less the least it could be, over every member-by-non-member pair.
                // One sort per column serves every group, where counting pairs
                // directly would cost m x r comparisons per group.
                double rankSum = 0.0;
                foreach (int i in members[g])
                    rankSum += ranks[i];

                double u = rankSum - m * (m + 1) / 2.0;
                separation[g, j] = 2.0 * u / ((double)m * r) - 1.0;
            }
        }

        var ranking = new int[k][];
        for (int g = 0; g < k; g++)
            ranking[g] = Enumerable.Range(0, d)
                .OrderByDescending(j => Math.Abs(separation[g, j]))
                .ThenByDescending(j => Math.Abs(effect[g, j]))
                .ThenBy(j => j)
                .ToArray();

        return new GroupSignatureResult(n, sizes, means, spread, restMeans, effect, separation, ranking);
    }

    /// <summary>
    /// 1-based ranks, ties sharing the average of the ranks they span — the
    /// convention the Mann-Whitney U is defined with, and what makes tied values
    /// count as half a win each.
    /// </summary>
    private static double[] AverageRanks(double[] values)
    {
        int n = values.Length;
        var order = Enumerable.Range(0, n).OrderBy(i => values[i]).ToArray();
        var ranks = new double[n];

        int start = 0;
        while (start < n)
        {
            int end = start;
            while (end + 1 < n && values[order[end + 1]] == values[order[start]])
                end++;

            double shared = (start + end) / 2.0 + 1.0;
            for (int t = start; t <= end; t++)
                ranks[order[t]] = shared;

            start = end + 1;
        }

        return ranks;
    }

    /// <summary>Mean and sample standard deviation (n − 1); zero spread for a single value.</summary>
    private static (double Mean, double Spread) MeanAndSpread(double[] values, IEnumerable<int> indices)
    {
        double sum = 0.0;
        int count = 0;
        foreach (int i in indices)
        {
            sum += values[i];
            count++;
        }

        if (count == 0)
            return (0.0, 0.0);

        double mean = sum / count;
        if (count == 1)
            return (mean, 0.0);

        double squares = 0.0;
        foreach (int i in indices)
            squares += (values[i] - mean) * (values[i] - mean);

        return (mean, Math.Sqrt(squares / (count - 1)));
    }
}
