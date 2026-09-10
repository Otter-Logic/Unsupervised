using OtterLogic.MachineLearning.Graphs;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Graph-network-style clustering with nothing to train: features, or labels,
/// spread over a graph by message passing, so each sample's answer takes account
/// of its neighbourhood as well as of itself.
/// <para>
/// A graph neural network layer does two things — it averages each node's
/// features with its neighbours', then multiplies by a learned weight matrix.
/// Simplified Graph Convolution (Wu et al., 2019) showed that most of the benefit
/// on node-level tasks comes from the first half, and the averaging needs no
/// training at all. That half is what is here. It is fitted on every solve from
/// the data on the wire, like a mixture, so it belongs in C# rather than behind an
/// ONNX boundary. A <em>learned</em> graph network — the second half — is a
/// different kind of model, trained in Python and shipped as a frozen graph, and
/// it lives in DeepLearning when it arrives. What this produces is the kind of
/// labelled data that model will be trained on.
/// </para>
/// <para>
/// Two uses, one operator:
/// <see cref="Fit"/> smooths the features and clusters the result — grouping by
/// behaviour, with neighbours pulling each other together; and
/// <see cref="Refine(double[,], WeightedGraph, PropagationOptions?)"/> smooths an
/// existing soft labelling instead, so a clustering from any method can be
/// corrected by what each sample's neighbours were given, and samples it left
/// unplaced can be filled in from theirs.
/// </para>
/// <para>
/// Each round is <c>H = (1 - a) S H + a X</c>, where <c>S</c> is the graph's
/// symmetric normalised adjacency with a self-loop on every node — a graph
/// convolution's renormalised operator — and <c>a</c> is
/// <see cref="PropagationOptions.Retention"/>. The self-loop is not decoration.
/// Without it a sample's next value is purely its neighbours' average, and on a
/// graph that alternates — a grid, a lattice, anything bipartite — the signal
/// flips between the two halves every round rather than settling.
/// </para>
/// </summary>
public static class MessagePassing
{
    /// <summary>
    /// Weight of the self-loop, relative to edge weights. One means a sample
    /// counts itself as much as a neighbour joined at full strength — the graph
    /// convolution convention, and the natural one when edges are affinities
    /// between zero and one.
    /// </summary>
    private const double SelfWeight = 1.0;

    /// <summary>
    /// Features after message passing: each sample's own blended with its
    /// neighbourhood's.
    /// </summary>
    /// <param name="x">n x d, one row per node, already prepared. Not modified.</param>
    /// <param name="graph">Which samples pass messages to which, and how strongly.</param>
    /// <param name="options">Hops and retention; null for the defaults.</param>
    public static double[,] Smooth(double[,] x, WeightedGraph graph, PropagationOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(graph);
        options ??= new PropagationOptions();
        options.Validate();

        if (x.GetLength(0) != graph.NodeCount)
            throw new ArgumentException(
                $"The graph has {graph.NodeCount} nodes but the data has {x.GetLength(0)} rows.", nameof(x));

        int n = x.GetLength(0);
        int d = x.GetLength(1);
        double keep = options.Retention;

        var h = (double[,])x.Clone();
        for (int hop = 0; hop < options.Hops; hop++)
        {
            var next = graph.Propagate(h, SelfWeight);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < d; j++)
                    next[i, j] = (1.0 - keep) * next[i, j] + keep * x[i, j];

            h = next;
        }

        return h;
    }

    /// <summary>
    /// Smooths the features over the graph, then partitions the result by k-means.
    /// </summary>
    /// <param name="x">n x d, one row per node, already prepared.</param>
    /// <param name="graph">Which samples pass messages to which, and how strongly.</param>
    /// <param name="options">Propagation and k-means settings.</param>
    public static MessagePassingResult Fit(double[,] x, WeightedGraph graph, MessagePassingOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        var embedding = Smooth(x, graph, options.Propagation);
        var partition = KMeans.Fit(embedding, options.KMeans);

        var labels = Labelling.Canonical(partition.Labels, out var mapping);
        int k = mapping.Count;
        int d = x.GetLength(1);

        var centroids = new double[k, d];
        foreach (var (from, to) in mapping)
            for (int j = 0; j < d; j++)
                centroids[to, j] = partition.Centroids[from, j];

        return new MessagePassingResult(
            labels, embedding, centroids, Labelling.Means(x, labels, k), partition.Inertia);
    }

    /// <summary>
    /// Refines a soft labelling by passing it over the graph.
    /// <para>
    /// Label propagation in the sense of Zhou et al. (2004), "learning with local
    /// and global consistency": each round a sample's memberships become its
    /// neighbourhood's, with <see cref="PropagationOptions.Retention"/> of its
    /// starting memberships restored — so retention is how far the original
    /// labelling is trusted over the neighbours. An all-zero row is a sample with
    /// no label yet; it starts with nothing to restore and takes whatever its
    /// neighbourhood holds.
    /// </para>
    /// </summary>
    /// <param name="responsibilities">
    /// n x k, non-negative. A mixture's responsibilities fit directly; so does any
    /// scoring where a larger value means more belonging.
    /// </param>
    /// <param name="graph">Which samples pass messages to which, and how strongly.</param>
    /// <param name="options">Hops and retention; null for the defaults.</param>
    public static RefinementResult Refine(
        double[,] responsibilities, WeightedGraph graph, PropagationOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(responsibilities);

        int n = responsibilities.GetLength(0);
        int k = responsibilities.GetLength(1);
        if (k < 1)
            throw new ArgumentException("Need at least one cluster column.", nameof(responsibilities));

        for (int i = 0; i < n; i++)
            for (int c = 0; c < k; c++)
                if (double.IsNaN(responsibilities[i, c]) || double.IsInfinity(responsibilities[i, c])
                    || responsibilities[i, c] < 0.0)
                    throw new ArgumentException(
                        $"Membership [{i}, {c}] is {responsibilities[i, c]}; memberships must be finite and non-negative.",
                        nameof(responsibilities));

        var initial = ArgMaxOrNone(responsibilities);
        var smoothed = Smooth(responsibilities, graph, options);

        // The symmetric normalisation does not preserve row sums — a well-connected
        // sample gathers more mass than a sparse one — so rows are renormalised to
        // distributions at the end rather than at every round, which would weight
        // the rounds differently for no reason.
        var refined = new double[n, k];
        var confidence = new double[n];

        for (int i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (int c = 0; c < k; c++)
                sum += smoothed[i, c];

            if (sum <= 0.0)
                continue;

            for (int c = 0; c < k; c++)
            {
                refined[i, c] = smoothed[i, c] / sum;
                confidence[i] = Math.Max(confidence[i], refined[i, c]);
            }
        }

        return new RefinementResult(refined, ArgMaxOrNone(refined), confidence, initial);
    }

    /// <summary>
    /// Refines a hard labelling by passing it over the graph. Negative labels are
    /// unlabelled — HDBSCAN's noise, say — and are filled in from their
    /// neighbourhood where it holds anything.
    /// </summary>
    /// <param name="labels">Cluster per sample, or negative for none.</param>
    /// <param name="graph">Which samples pass messages to which, and how strongly.</param>
    /// <param name="options">Hops and retention; null for the defaults.</param>
    public static RefinementResult Refine(int[] labels, WeightedGraph graph, PropagationOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(labels);

        int k = labels.Length == 0 ? 0 : labels.Max() + 1;
        if (k < 1)
            throw new ArgumentException("No sample carries a label, so there is nothing to propagate.", nameof(labels));

        var oneHot = new double[labels.Length, k];
        for (int i = 0; i < labels.Length; i++)
            if (labels[i] >= 0)
                oneHot[i, labels[i]] = 1.0;

        return Refine(oneHot, graph, options);
    }

    /// <summary>Largest column per row, first on a tie, or -1 for a row with nothing in it.</summary>
    private static int[] ArgMaxOrNone(double[,] memberships)
    {
        int n = memberships.GetLength(0);
        int k = memberships.GetLength(1);
        var labels = new int[n];

        for (int i = 0; i < n; i++)
        {
            int best = -1;
            double bestValue = 0.0;

            for (int c = 0; c < k; c++)
            {
                if (memberships[i, c] > bestValue)
                {
                    bestValue = memberships[i, c];
                    best = c;
                }
            }

            labels[i] = best;
        }

        return labels;
    }
}
