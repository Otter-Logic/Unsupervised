namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// How far two labellings of the same samples agree, whatever numbers either of
/// them used.
/// <para>
/// Public because agreement is now something the library decides with, not only
/// something the tests check: fusing several clusterings weights each by how far
/// the others agree with it, and a selection that fits three models can say how
/// alike their answers were. The tests had their own copy; this one also reads
/// unplaced samples, which the tests never needed to.
/// </para>
/// </summary>
public static class ClusterAgreement
{
    /// <summary>
    /// Adjusted Rand index: 1 for the same partition, about 0 for agreement no
    /// better than chance, negative for worse.
    /// <para>
    /// Counts pairs of samples both labellings put together or both put apart, so
    /// renumbering either changes nothing. Adjusted, because the raw share of
    /// agreeing pairs is high for any two partitions into many small clusters —
    /// most pairs are apart in both simply because most pairs are apart.
    /// </para>
    /// <para>
    /// A negative label is a sample left unplaced, and each one is read as a
    /// cluster of its own. That is the only reading that neither rewards nor
    /// punishes a method for declining to place something: two labellings that
    /// both leave a sample out agree about it, and one that files it with others
    /// disagrees about exactly those pairs.
    /// </para>
    /// </summary>
    public static double AdjustedRand(int[] a, int[] b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        if (a.Length != b.Length)
            throw new ArgumentException($"{a.Length} labels against {b.Length}; both must label the same samples.", nameof(b));

        int n = a.Length;
        if (n < 2)
            return 1.0;

        var left = Compact(a);
        var right = Compact(b);

        var cells = new Dictionary<long, int>();
        var rowSums = new int[left.Max() + 1];
        var columnSums = new int[right.Max() + 1];

        for (int i = 0; i < n; i++)
        {
            long key = (long)left[i] * columnSums.Length + right[i];
            cells[key] = cells.TryGetValue(key, out int count) ? count + 1 : 1;
            rowSums[left[i]]++;
            columnSums[right[i]]++;
        }

        double index = cells.Values.Sum(Pairs);
        double rowTotal = rowSums.Sum(Pairs);
        double columnTotal = columnSums.Sum(Pairs);
        double expected = rowTotal * columnTotal / Pairs(n);
        double maximum = 0.5 * (rowTotal + columnTotal);

        return maximum - expected == 0.0 ? 1.0 : (index - expected) / (maximum - expected);
    }

    /// <summary>Labels renumbered from zero, each negative one given a number of its own.</summary>
    private static int[] Compact(int[] labels)
    {
        var numbers = new Dictionary<int, int>();
        var compact = new int[labels.Length];
        int next = 0;

        for (int i = 0; i < labels.Length; i++)
        {
            if (labels[i] < 0)
            {
                compact[i] = next++;
                continue;
            }

            if (!numbers.TryGetValue(labels[i], out int number))
                numbers[labels[i]] = number = next++;

            compact[i] = number;
        }

        return compact;
    }

    private static double Pairs(int count) => count * (count - 1) / 2.0;
}
