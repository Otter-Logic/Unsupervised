using OtterLogic.Graphs;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Describes every sample by what it is like <em>and what it is surrounded by</em>,
/// so that samples playing the same part in different places come out alike.
/// <para>
/// This is the opposite question to the one <see cref="MessagePassing"/> and
/// <see cref="SpectralClustering"/> answer. Those find <b>communities</b>: they pull
/// connected samples together, so their groups are regions of the graph. This finds
/// <b>roles</b>: two samples are alike when their own features match and their
/// neighbours' features match and their neighbours' neighbours' do, whether or not
/// anything connects the two of them — structural equivalence, in the sense of
/// struc2vec and GraphSAGE's mean aggregator, with nothing to train.
/// </para>
/// <para>
/// The difference is in one operation. Message passing <em>blends</em> each sample
/// with its neighbourhood, which is what makes neighbours alike. Here the
/// neighbourhood's mean is <em>appended</em> beside the sample's own features, one
/// block per hop, so a sample and the quite different thing it is attached to each
/// keep what they are and gain a record of the other. Clustering those rows groups a
/// hub with every other hub and a spoke with every other spoke, however far apart.
/// </para>
/// </summary>
public static class NeighbourhoodProfile
{
    /// <summary>
    /// Each sample's features, followed by the mean of its neighbours', followed by
    /// the mean of theirs, out to <paramref name="hops"/>.
    /// </summary>
    /// <param name="x">n x d, one row per node, already prepared. Not modified.</param>
    /// <param name="graph">Which samples neighbour which; a heavier edge counts for more of the mean.</param>
    /// <param name="hops">How far out the surroundings are recorded, one block of d columns each. At least one.</param>
    /// <param name="decay">
    /// Each hop's block is scaled by this over the one before, between 0 and 1. What a
    /// sample is matters more than what it touches, and that more than what that
    /// touches; without the scaling two hops of surroundings would outvote the sample
    /// itself two to one in any distance taken over the row.
    /// </param>
    /// <returns>n x d(hops + 1). A sample with no neighbours has zeros in every block after its own.</returns>
    public static double[,] Embed(double[,] x, WeightedGraph graph, int hops = 2, double decay = 0.5)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(graph);

        if (x.GetLength(0) != graph.NodeCount)
            throw new ArgumentException(
                $"The graph has {graph.NodeCount} nodes but the data has {x.GetLength(0)} rows.", nameof(x));
        if (hops < 1)
            throw new ArgumentOutOfRangeException(nameof(hops), hops, "Need at least one hop; with none this is the data unchanged.");
        if (!(decay > 0.0) || decay > 1.0)
            throw new ArgumentOutOfRangeException(nameof(decay), decay, "Decay must be above 0 and at most 1.");

        int n = x.GetLength(0);
        int d = x.GetLength(1);
        var profile = new double[n, d * (hops + 1)];

        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                profile[i, j] = x[i, j];

        var current = x;
        double scale = 1.0;

        for (int hop = 1; hop <= hops; hop++)
        {
            var mean = new double[n, d];
            for (int i = 0; i < n; i++)
            {
                var neighbours = graph.Neighbours(i);
                var weights = graph.EdgeWeights(i);
                double total = graph.Degree(i);
                if (total <= 0.0)
                    continue;

                for (int e = 0; e < neighbours.Length; e++)
                {
                    double share = weights[e] / total;
                    for (int j = 0; j < d; j++)
                        mean[i, j] += share * current[neighbours[e], j];
                }
            }

            scale *= decay;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < d; j++)
                    profile[i, hop * d + j] = scale * mean[i, j];

            current = mean;
        }

        return profile;
    }
}
