namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Groups samples so that, in every column, each member carries at least a set
/// share of the largest value in its group.
/// <para>
/// A different question from every other method here. Those ask which samples
/// are alike — near each other, in the same dense region, strongly connected —
/// and a cluster from any of them may run from small to large as long as it runs
/// smoothly. This asks which samples can share one design sized for the largest
/// of them without wasting it: a connection type, a panel size, a stock length.
/// That answer has to bound the spread inside a group, and nothing statistical
/// does.
/// </para>
/// <para>
/// Values are read as ratios, so 80 against 100 is the same gap as 8,000 against
/// 10,000, and the gap between two samples is the largest in any column,
/// <c>1 − smaller / larger</c>. The grouping is a complete-linkage tree over that
/// gap cut at <c>1 − share</c>: complete linkage because it is the one linkage
/// whose clusters are bounded by the cut, so no two members differ by more than
/// the cut and every member carries at least the share of its group's peak. The
/// number of groups follows from the share rather than being guessed.
/// </para>
/// <para>
/// Lifted out of the beam end plate grouping in StructuralDesign, where it was
/// first written, because nothing in it knows what the columns are. What
/// governs, and how small a value is too small to matter, stays with the caller:
/// raise anything under a floor to the floor before calling, and the small
/// samples form light groups rather than splitting on differences nobody designs
/// for.
/// </para>
/// <para>
/// O(n² d) time and O(n²) memory — the tree's.
/// </para>
/// </summary>
public static class RatioClustering
{
    /// <summary>Groups the rows of <paramref name="x"/>, largest group first.</summary>
    /// <param name="x">n x d, rows are samples: finite magnitudes, zero or more.</param>
    /// <param name="minimumShare">
    /// Least share of its group's peak every member carries in every column,
    /// between 0 and 1. Zero puts everything in one group; one groups only
    /// samples that are identical.
    /// </param>
    public static RatioClusteringResult Fit(double[,] x, double minimumShare)
    {
        ArgumentNullException.ThrowIfNull(x);

        int n = x.GetLength(0);
        int d = x.GetLength(1);

        if (n < 1 || d < 1)
            throw new ArgumentException("Need at least one sample and one column.", nameof(x));
        if (!(minimumShare >= 0.0 && minimumShare <= 1.0))
            throw new ArgumentOutOfRangeException(nameof(minimumShare), minimumShare, "A share is between 0 and 1.");

        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                if (!double.IsFinite(x[i, j]) || x[i, j] < 0.0)
                    throw new ArgumentException(
                        $"Value [{i}, {j}] is {x[i, j]}; ratios need finite values of zero or more, so pass magnitudes.",
                        nameof(x));

        int[] labels;
        if (n == 1)
        {
            labels = new[] { 0 };
        }
        else
        {
            var tree = HierarchicalClustering.Fit(
                n, (a, b) => Gap(x, a, b), new HierarchicalClusteringOptions { Linkage = Linkage.Complete });

            // Cutting at a distance undoes every merge at or above it; the next
            // number up keeps a merge that lands exactly on the limit.
            labels = tree.CutAtDistance(Math.BitIncrement(1.0 - minimumShare));
        }

        int k = labels.Max() + 1;
        var members = ClusterLabels.Members(labels, k);
        var peaks = new double[k, d];
        var leastShare = new double[k];

        for (int c = 0; c < k; c++)
        {
            double least = 1.0;
            for (int j = 0; j < d; j++)
            {
                double peak = 0.0;
                double smallest = double.MaxValue;
                foreach (int i in members[c])
                {
                    peak = Math.Max(peak, x[i, j]);
                    smallest = Math.Min(smallest, x[i, j]);
                }

                peaks[c, j] = peak;
                if (peak > 0.0)
                    least = Math.Min(least, smallest / peak);
            }

            leastShare[c] = least;
        }

        return new RatioClusteringResult(labels, peaks, leastShare, minimumShare);
    }

    /// <summary>
    /// The largest ratio gap between two rows in any column: zero for identical
    /// rows, <c>1 − smaller / larger</c> otherwise — so it is at most
    /// <c>1 − share</c> exactly when the smaller carries at least that share of
    /// the larger. Two zeros are identical; a zero against anything is a gap of one.
    /// <para>
    /// Written as one minus the ratio, not as the difference over the larger,
    /// though the two are equal on paper: the cut is <c>1 − share</c>, and taking
    /// both through the same subtraction is what keeps a pair sitting exactly on
    /// the share — 80 against 100 at 0.8 — on the same side of it.
    /// </para>
    /// </summary>
    private static double Gap(double[,] x, int a, int b)
    {
        double worst = 0.0;
        for (int j = 0; j < x.GetLength(1); j++)
        {
            double larger = Math.Max(x[a, j], x[b, j]);
            if (larger > 0.0)
                worst = Math.Max(worst, 1.0 - Math.Min(x[a, j], x[b, j]) / larger);
        }

        return worst;
    }
}
