namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Canonical cluster numbering: largest cluster first, ties to whichever holds
/// the lowest sample index.
/// <para>
/// Every method here numbers its clusters in whatever order its internals
/// happened to produce — the k-means start that won, the order a tree was cut,
/// the component an eigenvector happened to favour. A small change upstream then
/// permutes the numbers, cluster zero becomes cluster two, and every colour and
/// geometry assignment downstream jumps for no reason a user can see. Numbering by
/// a property of the answer rather than of its history is what stops that, and
/// doing it in one place is what keeps the tie-break the same everywhere.
/// </para>
/// </summary>
internal static class Labelling
{
    /// <summary>
    /// Renumbers <paramref name="labels"/> canonically. Negative labels mean
    /// unplaced and come back as <c>-1</c>.
    /// </summary>
    /// <param name="labels">Any non-negative integers per sample, or negative for unplaced.</param>
    /// <param name="mapping">Old label to new, for callers that must reorder something alongside.</param>
    internal static int[] Canonical(int[] labels, out Dictionary<int, int> mapping)
    {
        var counts = new Dictionary<int, int>();
        var first = new Dictionary<int, int>();

        for (int i = 0; i < labels.Length; i++)
        {
            int label = labels[i];
            if (label < 0)
                continue;

            counts[label] = counts.TryGetValue(label, out int c) ? c + 1 : 1;
            first.TryAdd(label, i);
        }

        mapping = new Dictionary<int, int>(counts.Count);
        int next = 0;
        foreach (int label in counts.Keys.OrderByDescending(l => counts[l]).ThenBy(l => first[l]))
            mapping[label] = next++;

        var result = new int[labels.Length];
        for (int i = 0; i < labels.Length; i++)
            result[i] = labels[i] < 0 ? -1 : mapping[labels[i]];

        return result;
    }

    /// <inheritdoc cref="Canonical(int[], out Dictionary{int, int})"/>
    internal static int[] Canonical(int[] labels) => Canonical(labels, out _);

    /// <summary>Sample indices bucketed by label, unplaced samples excluded.</summary>
    internal static int[][] Buckets(int[] labels, int clusterCount)
    {
        var buckets = new List<int>[clusterCount];
        for (int c = 0; c < clusterCount; c++)
            buckets[c] = new List<int>();

        for (int i = 0; i < labels.Length; i++)
            if (labels[i] >= 0)
                buckets[labels[i]].Add(i);

        return buckets.Select(b => b.ToArray()).ToArray();
    }

    /// <summary>Mean of the rows of <paramref name="x"/> in each cluster, unplaced rows excluded.</summary>
    internal static double[,] Means(double[,] x, int[] labels, int clusterCount)
    {
        int d = x.GetLength(1);
        var means = new double[clusterCount, d];
        var counts = new int[clusterCount];

        for (int i = 0; i < labels.Length; i++)
        {
            int c = labels[i];
            if (c < 0)
                continue;

            counts[c]++;
            for (int j = 0; j < d; j++)
                means[c, j] += x[i, j];
        }

        for (int c = 0; c < clusterCount; c++)
            if (counts[c] > 0)
                for (int j = 0; j < d; j++)
                    means[c, j] /= counts[c];

        return means;
    }
}
