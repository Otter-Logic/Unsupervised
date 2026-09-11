using OtterLogic.MachineLearning.Distances;
using OtterLogic.MachineLearning.Graphs;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Turns "which samples are related" into "how strongly", by how alike their
/// features are.
/// <para>
/// This is the join between connectivity and behaviour. A graph says which
/// samples may influence each other — they touch, they are near, somebody said
/// so — and says nothing about whether they are alike. Features say how alike any
/// two samples are and nothing about whether they are related. Weighting each
/// edge by feature similarity gives a graph where a strong edge means both, and
/// that is the graph the spectral and message-passing methods want: cuts fall
/// where related samples stop being alike.
/// </para>
/// </summary>
public static class Affinity
{
    /// <summary>
    /// Floor on the kernel, as a fraction of the edge's own weight.
    /// <para>
    /// Far-apart neighbours would otherwise underflow to exactly zero and the edge
    /// would vanish, quietly changing the graph's components and isolating
    /// samples the caller connected on purpose. The topology is the caller's
    /// claim; the kernel only says how strongly it holds.
    /// </para>
    /// </summary>
    private const double KernelFloor = 1e-12;

    /// <summary>
    /// Reweights every edge by a self-tuning Gaussian kernel on feature distance:
    /// <c>w * exp(-d^2 / (s_a s_b))</c>, where <c>s_a</c> is a local scale for
    /// sample a.
    /// <para>
    /// A single global bandwidth fails whenever density varies — tight clusters
    /// beside diffuse ones — because any one width is too wide for the first and
    /// too narrow for the second. The local scale follows Zelnik-Manor and Perona
    /// (2004), with one change: it is the median distance to the sample's own
    /// graph neighbours rather than the distance to its seventh-nearest point in
    /// feature space. On a nearest-neighbour graph those are much the same thing;
    /// on a connectivity graph the neighbours in feature space may not be
    /// neighbours at all, and the scale should be set by the ones that are.
    /// </para>
    /// <para>
    /// A sample whose neighbours all coincide with it has a local scale of zero,
    /// and borrows the median over every edge instead. If every edge has zero
    /// length there is no similarity to measure and the graph comes back as given.
    /// </para>
    /// </summary>
    /// <param name="graph">Which samples are related. Existing weights are kept as a multiplier.</param>
    /// <param name="x">n x d features, one row per node, already prepared.</param>
    public static WeightedGraph Gaussian(WeightedGraph graph, double[,] x)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(x);

        int n = graph.NodeCount;
        if (x.GetLength(0) != n)
            throw new ArgumentException(
                $"The graph has {n} nodes but the features have {x.GetLength(0)} rows.", nameof(x));

        double Distance(int a, int b) => Euclidean.Between(x, a, x, b);

        var positive = new List<double>(2 * graph.EdgeCount);
        var local = new double[n];
        var around = new List<double>();

        for (int a = 0; a < n; a++)
        {
            around.Clear();
            foreach (int b in graph.Neighbours(a))
                around.Add(Distance(a, b));

            local[a] = Median(around);
            foreach (double length in around)
                if (length > 0.0)
                    positive.Add(length);
        }

        if (positive.Count == 0)
            return graph;

        double global = Median(positive);
        for (int a = 0; a < n; a++)
            if (local[a] <= 0.0)
                local[a] = global;

        return graph.Reweight((a, b, weight) =>
        {
            double distance = Distance(a, b);
            double kernel = Math.Exp(-distance * distance / (local[a] * local[b]));
            return weight * Math.Max(kernel, KernelFloor);
        });
    }

    private static double Median(List<double> values)
    {
        if (values.Count == 0)
            return 0.0;

        values.Sort();
        int middle = values.Count / 2;
        return values.Count % 2 == 1 ? values[middle] : 0.5 * (values[middle - 1] + values[middle]);
    }
}
