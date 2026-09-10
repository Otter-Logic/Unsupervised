using OtterLogic.MachineLearning.Graphs;

namespace OtterLogic.Unsupervised.Tests;

/// <summary>
/// Generated graphs with a known answer, for the graph methods' behaviour tests.
/// <para>
/// Built in C# rather than as fixtures because nothing is being compared with a
/// reference implementation here — the question is whether a method recovers
/// structure that was put there on purpose, and a seeded generator states that
/// structure more plainly than a JSON file would.
/// </para>
/// </summary>
internal static class Synthetic
{
    /// <summary>
    /// A planted-partition graph: communities densely connected inside and
    /// sparsely between, with features that are each community's centre plus heavy
    /// noise — so heavy that the features alone barely separate the communities
    /// and the graph is needed to do it.
    /// </summary>
    internal static (WeightedGraph Graph, double[,] X, int[] Truth) Communities(
        int communities = 4, int size = 60, double inside = 0.15, double between = 0.004,
        double noise = 1.6, int dimensions = 4, int seed = 17)
    {
        var rng = new Random(seed);
        int n = communities * size;
        var truth = Enumerable.Range(0, n).Select(i => i / size).ToArray();

        var edges = new List<(int, int)>();
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (rng.NextDouble() < (truth[i] == truth[j] ? inside : between))
                    edges.Add((i, j));

        var centres = new double[communities, dimensions];
        for (int c = 0; c < communities; c++)
            centres[c, c % dimensions] = 1.0;

        var x = new double[n, dimensions];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < dimensions; j++)
                x[i, j] = centres[truth[i], j] + noise * Gaussian(rng);

        return (WeightedGraph.FromEdges(n, edges), x, truth);
    }

    /// <summary>
    /// A rectangular lattice, each node joined to its four neighbours, whose
    /// features change abruptly at one column — the boundary sits off-centre on
    /// purpose — with noise on top.
    /// <para>
    /// Built so that each kind of evidence alone points somewhere wrong. The
    /// lattice's own weakest cut is across its middle, which is where connectivity
    /// alone splits it. The features alone see the boundary, but the noise leaves
    /// stray samples on the wrong side. Only both together put every sample on
    /// the correct side of the true boundary.
    /// </para>
    /// </summary>
    internal static (WeightedGraph Graph, double[,] X, int[] Truth) Lattice(
        int columns = 30, int rows = 8, int boundary = 9, double noise = 0.45, int seed = 23)
    {
        var rng = new Random(seed);
        int n = columns * rows;
        int Node(int column, int row) => column * rows + row;

        var edges = new List<(int, int)>();
        for (int c = 0; c < columns; c++)
        {
            for (int r = 0; r < rows; r++)
            {
                if (c + 1 < columns)
                    edges.Add((Node(c, r), Node(c + 1, r)));
                if (r + 1 < rows)
                    edges.Add((Node(c, r), Node(c, r + 1)));
            }
        }

        var truth = new int[n];
        var x = new double[n, 2];
        for (int c = 0; c < columns; c++)
        {
            for (int r = 0; r < rows; r++)
            {
                int i = Node(c, r);
                truth[i] = c < boundary ? 0 : 1;
                x[i, 0] = (truth[i] == 0 ? -1.0 : 1.0) + noise * Gaussian(rng);
                x[i, 1] = noise * Gaussian(rng);
            }
        }

        return (WeightedGraph.FromEdges(n, edges), x, truth);
    }

    /// <summary>Whether every cluster in <paramref name="labels"/> is one connected piece of the graph.</summary>
    internal static bool EveryClusterIsConnected(WeightedGraph graph, int[] labels)
    {
        foreach (int cluster in labels.Where(l => l >= 0).Distinct())
        {
            var members = Enumerable.Range(0, labels.Length).Where(i => labels[i] == cluster).ToArray();
            var seen = new HashSet<int> { members[0] };
            var stack = new Stack<int>();
            stack.Push(members[0]);

            while (stack.Count > 0)
                foreach (int next in graph.Neighbours(stack.Pop()))
                    if (labels[next] == cluster && seen.Add(next))
                        stack.Push(next);

            if (seen.Count != members.Length)
                return false;
        }

        return true;
    }

    /// <summary>Box-Muller, so the generator depends on nothing but <see cref="Random"/>.</summary>
    private static double Gaussian(Random rng)
    {
        double u = 1.0 - rng.NextDouble();
        double v = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u)) * Math.Cos(2.0 * Math.PI * v);
    }
}
