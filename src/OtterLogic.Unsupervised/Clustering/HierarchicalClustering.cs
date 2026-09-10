using OtterLogic.MachineLearning.Graphs;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Agglomerative hierarchical clustering: start with every sample on its own and
/// repeatedly merge the closest pair of clusters, recording the whole tree.
/// <para>
/// What it offers that the flat methods do not is nesting. Cut the tree at eight
/// clusters and at three, and every one of the eight sits wholly inside one of the
/// three. Fine-grained groups for the people acting on them and coarse ones for an
/// overview come from one fit and are guaranteed to agree — two separate k-means
/// runs at those counts promise nothing of the kind.
/// </para>
/// <para>
/// With a connectivity graph, only clusters joined by an edge may merge, so every
/// cluster at every level is a connected piece of the graph. That is the version
/// for grouping things that must stay contiguous: regions, runs, anything where a
/// group split in two by something that is not in it is not a group.
/// </para>
/// <para>
/// Unconstrained, this is the nearest-neighbour chain algorithm (Müllner 2011) —
/// O(n^2) time, and SciPy's exact route, so the tree matches <c>linkage</c> merge
/// for merge. Constrained, it is a priority queue over the graph's edges, which is
/// scikit-learn's route and costs time and memory in proportion to the edges
/// rather than to n^2.
/// </para>
/// </summary>
public static class HierarchicalClustering
{
    /// <summary>
    /// Pairwise distances the unconstrained fit will hold, at most. The chain
    /// algorithm needs every one, n(n-1)/2 doubles; this is a gigabyte, reached
    /// at about sixteen thousand samples. Past it, the answer is a connectivity
    /// graph, which needs memory only per edge.
    /// </summary>
    private const long MaximumPairs = 1L << 27;

    /// <summary>
    /// Builds the full tree over the samples, any cluster free to merge with any
    /// other.
    /// </summary>
    /// <param name="x">n x d data, rows are samples. Expected to be standardised already.</param>
    /// <param name="options">Fit settings; null for Ward.</param>
    public static HierarchicalClusteringResult Fit(double[,] x, HierarchicalClusteringOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        options ??= new HierarchicalClusteringOptions();

        int n = x.GetLength(0);
        if (n < 2)
            throw new ArgumentException("Need at least two samples to build a hierarchy.", nameof(x));

        long pairs = (long)n * (n - 1) / 2;
        if (pairs > MaximumPairs)
            throw new ArgumentException(
                $"{n} samples need {pairs:N0} pairwise distances, more than an unconstrained tree "
                + "should hold in memory. Pass a connectivity graph — a nearest-neighbour graph "
                + "will do — and the fit needs memory only per edge.", nameof(x));

        var merges = NearestNeighbourChain(x, options.Linkage);
        return new HierarchicalClusteringResult(merges, n, options.Linkage, 1);
    }

    /// <summary>
    /// Builds the full tree with merges restricted to clusters joined by an edge
    /// of <paramref name="connectivity"/>.
    /// </summary>
    /// <param name="x">n x d data, one row per node. Expected to be standardised already.</param>
    /// <param name="connectivity">Which samples may be grouped together. Weights are ignored.</param>
    /// <param name="options">Fit settings; null for Ward.</param>
    public static HierarchicalClusteringResult Fit(
        double[,] x, WeightedGraph connectivity, HierarchicalClusteringOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(connectivity);
        options ??= new HierarchicalClusteringOptions();

        int n = x.GetLength(0);
        if (n < 2)
            throw new ArgumentException("Need at least two samples to build a hierarchy.", nameof(x));
        if (connectivity.NodeCount != n)
            throw new ArgumentException(
                $"The graph has {connectivity.NodeCount} nodes but the data has {n} rows.", nameof(connectivity));

        connectivity.ConnectedComponents(out int components);
        var merges = new ConstrainedTree(x, options.Linkage).Build(connectivity);
        return new HierarchicalClusteringResult(merges, n, options.Linkage, components);
    }

    /// <summary>
    /// The nearest-neighbour chain, following SciPy's <c>nn_chain</c> step for
    /// step: grow a chain in which each cluster's nearest neighbour is the next,
    /// until two clusters are each other's nearest; merge those; carry on from
    /// what is left of the chain.
    /// <para>
    /// It works because these four linkages are reducible — merging two clusters
    /// never brings the result closer to a third than the nearer of the two was —
    /// so a mutual nearest pair found anywhere is safe to merge now, and the rest
    /// of the chain stays valid. Merges come out of order and are sorted by
    /// distance at the end, then renumbered into the dendrogram layout.
    /// </para>
    /// </summary>
    private static ClusterMerge[] NearestNeighbourChain(double[,] x, Linkage linkage)
    {
        int n = x.GetLength(0);
        int d = x.GetLength(1);

        // Condensed upper triangle, SciPy's layout.
        var distances = new double[(long)n * (n - 1) / 2];
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                distances[Condensed(n, i, j)] = Math.Sqrt(KMeans.SquaredDistance(x, i, x, j, d));

        // A slot holds a cluster, named after one of its samples; zero size means
        // the slot has been merged away.
        var size = new int[n];
        Array.Fill(size, 1);

        var chain = new int[n];
        int length = 0;
        var raw = new (int X, int Y, double Distance)[n - 1];

        for (int k = 0; k < n - 1; k++)
        {
            if (length == 0)
            {
                length = 1;
                for (int i = 0; i < n; i++)
                {
                    if (size[i] > 0)
                    {
                        chain[0] = i;
                        break;
                    }
                }
            }

            int a;
            int b;
            double nearest;

            while (true)
            {
                a = chain[length - 1];

                // Prefer the previous link on a tie. Without that, two equidistant
                // clusters can hand the chain back and forth forever.
                if (length > 1)
                {
                    b = chain[length - 2];
                    nearest = distances[Condensed(n, a, b)];
                }
                else
                {
                    b = -1;
                    nearest = double.PositiveInfinity;
                }

                for (int i = 0; i < n; i++)
                {
                    if (size[i] == 0 || i == a)
                        continue;

                    double distance = distances[Condensed(n, a, i)];
                    if (distance < nearest)
                    {
                        nearest = distance;
                        b = i;
                    }
                }

                if (length > 1 && b == chain[length - 2])
                    break;

                chain[length++] = b;
            }

            length -= 2;

            if (a > b)
                (a, b) = (b, a);

            int sizeA = size[a];
            int sizeB = size[b];
            raw[k] = (a, b, nearest);

            // The merged cluster takes over slot b; slot a is retired.
            size[a] = 0;
            size[b] = sizeA + sizeB;

            for (int i = 0; i < n; i++)
            {
                if (size[i] == 0 || i == b)
                    continue;

                long toB = Condensed(n, i, b);
                distances[toB] = LanceWilliams(
                    linkage, distances[Condensed(n, i, a)], distances[toB], nearest, sizeA, sizeB, size[i]);
            }
        }

        // Stable, so merges at an equal distance keep the order they were made in.
        var ordered = raw.OrderBy(m => m.Distance).ToArray();
        return Renumber(ordered, n);
    }

    /// <summary>
    /// Turns merges of sample-named slots into merges of dendrogram nodes, by
    /// union-find in merge order — SciPy's <c>label</c>. The lower node goes on
    /// the left.
    /// </summary>
    private static ClusterMerge[] Renumber((int X, int Y, double Distance)[] ordered, int n)
    {
        var parent = new int[2 * n - 1];
        var size = new int[2 * n - 1];
        for (int i = 0; i < parent.Length; i++)
            parent[i] = i;
        for (int i = 0; i < n; i++)
            size[i] = 1;

        int Find(int node)
        {
            int root = node;
            while (parent[root] != root)
                root = parent[root];

            while (parent[node] != root)
            {
                int next = parent[node];
                parent[node] = root;
                node = next;
            }

            return root;
        }

        var merges = new ClusterMerge[ordered.Length];
        for (int t = 0; t < ordered.Length; t++)
        {
            int rootX = Find(ordered[t].X);
            int rootY = Find(ordered[t].Y);
            int created = n + t;

            size[created] = size[rootX] + size[rootY];
            parent[rootX] = created;
            parent[rootY] = created;

            merges[t] = new ClusterMerge(
                Math.Min(rootX, rootY), Math.Max(rootX, rootY), ordered[t].Distance, size[created]);
        }

        return merges;
    }

    /// <summary>
    /// The Lance-Williams update: the distance from cluster i to the union of a
    /// and b, from the distances to a and b alone. Written as SciPy writes it, so
    /// the rounding agrees too.
    /// </summary>
    private static double LanceWilliams(
        Linkage linkage, double toA, double toB, double between, int sizeA, int sizeB, int sizeI)
    {
        switch (linkage)
        {
            case Linkage.Single:
                return Math.Min(toA, toB);
            case Linkage.Complete:
                return Math.Max(toA, toB);
            case Linkage.Average:
                return (sizeA * toA + sizeB * toB) / (sizeA + sizeB);
            case Linkage.Ward:
                double t = 1.0 / (sizeA + sizeB + sizeI);
                return Math.Sqrt((sizeI + sizeA) * t * toA * toA
                                 + (sizeI + sizeB) * t * toB * toB
                                 - sizeI * t * between * between);
            default:
                throw new ArgumentOutOfRangeException(nameof(linkage), linkage, null);
        }
    }

    private static long Condensed(int n, int i, int j)
    {
        if (i > j)
            (i, j) = (j, i);

        return (long)n * i - (long)i * (i + 1) / 2 + (j - i - 1);
    }

    /// <summary>
    /// Merges over a connectivity graph, cheapest edge first, by a priority queue
    /// with lazy deletion — scikit-learn's structured tree.
    /// <para>
    /// Every merge creates a fresh node, so an entry in the queue that names a
    /// node already merged away is stale by construction and is skipped when it
    /// surfaces, rather than searched for and removed.
    /// </para>
    /// <para>
    /// Ward distances come from cluster sizes and sums, needing nothing per pair.
    /// The other linkages keep a distance per pair of adjacent clusters and update
    /// it by Lance-Williams where both halves are known; where a neighbour touched
    /// only one of the two clusters just merged, its distance to the other was
    /// never needed before, and is computed from the samples directly.
    /// </para>
    /// </summary>
    private sealed class ConstrainedTree
    {
        private readonly double[,] _x;
        private readonly Linkage _linkage;
        private readonly int _n;
        private readonly int _d;

        private readonly int[] _size;
        private readonly double[,] _sums;
        private readonly List<int>?[] _members;
        private readonly HashSet<int>?[] _adjacent;
        private readonly Dictionary<long, double> _pairDistance = new();
        private readonly PriorityQueue<(int A, int B), (double Distance, int Newer, int Older)> _queue = new();

        public ConstrainedTree(double[,] x, Linkage linkage)
        {
            _x = x;
            _linkage = linkage;
            _n = x.GetLength(0);
            _d = x.GetLength(1);

            int nodes = 2 * _n - 1;
            _size = new int[nodes];
            _sums = new double[nodes, _d];
            _members = new List<int>?[nodes];
            _adjacent = new HashSet<int>?[nodes];

            for (int i = 0; i < _n; i++)
            {
                _size[i] = 1;
                for (int j = 0; j < _d; j++)
                    _sums[i, j] = x[i, j];

                _members[i] = new List<int> { i };
                _adjacent[i] = new HashSet<int>();
            }
        }

        public ClusterMerge[] Build(WeightedGraph connectivity)
        {
            foreach (var (a, b, _) in connectivity.Edges())
                Connect(a, b);

            var merges = new ClusterMerge[_n - 1];
            int created = _n;

            while (created < 2 * _n - 1)
            {
                if (_queue.Count == 0)
                {
                    // Every component has become one cluster and nothing else may
                    // merge under the constraint. Let the remaining roots merge
                    // freely so the tree is complete — GraphComponents on the
                    // result says how many of the top merges these are.
                    var roots = Enumerable.Range(0, created).Where(c => _adjacent[c] is not null).ToArray();
                    for (int r = 0; r < roots.Length; r++)
                        for (int s = r + 1; s < roots.Length; s++)
                            Connect(roots[r], roots[s]);

                    continue;
                }

                _queue.TryDequeue(out var pair, out var priority);
                var (a, b) = pair;

                if (_adjacent[a] is null || _adjacent[b] is null)
                    continue;

                int u = created++;
                merges[u - _n] = new ClusterMerge(Math.Min(a, b), Math.Max(a, b), priority.Distance, _size[a] + _size[b]);
                Merge(a, b, u);
            }

            return merges;
        }

        /// <summary>Makes two live clusters adjacent and queues the pair.</summary>
        private void Connect(int a, int b)
        {
            if (!_adjacent[a]!.Add(b))
                return;

            _adjacent[b]!.Add(a);

            double distance = Distance(a, b);
            if (_linkage != Linkage.Ward)
                _pairDistance[Key(a, b)] = distance;

            _queue.Enqueue((a, b), (distance, Math.Max(a, b), Math.Min(a, b)));
        }

        private void Merge(int a, int b, int u)
        {
            _size[u] = _size[a] + _size[b];
            for (int j = 0; j < _d; j++)
                _sums[u, j] = _sums[a, j] + _sums[b, j];

            if (_linkage != Linkage.Ward)
            {
                var larger = _members[a]!.Count >= _members[b]!.Count ? _members[a]! : _members[b]!;
                var smaller = ReferenceEquals(larger, _members[a]) ? _members[b]! : _members[a]!;
                larger.AddRange(smaller);
                _members[u] = larger;
            }

            _members[a] = null;
            _members[b] = null;

            // Ascending, so the queue sees the same insertions in the same order
            // whatever order the hash sets happened to hold them in.
            var neighbours = _adjacent[a]!.Union(_adjacent[b]!).Where(c => c != a && c != b).OrderBy(c => c).ToArray();

            var aroundU = new HashSet<int>();
            _adjacent[u] = aroundU;

            foreach (int c in neighbours)
            {
                double distance;
                if (_linkage == Linkage.Ward)
                {
                    distance = WardDistance(c, u);
                }
                else if (_pairDistance.TryGetValue(Key(c, a), out double toA)
                         && _pairDistance.TryGetValue(Key(c, b), out double toB))
                {
                    distance = LanceWilliams(_linkage, toA, toB, 0.0, _size[a], _size[b], _size[c]);
                }
                else
                {
                    distance = Direct(c, u);
                }

                var aroundC = _adjacent[c]!;
                aroundC.Remove(a);
                aroundC.Remove(b);
                aroundC.Add(u);
                aroundU.Add(c);

                if (_linkage != Linkage.Ward)
                {
                    _pairDistance.Remove(Key(c, a));
                    _pairDistance.Remove(Key(c, b));
                    _pairDistance[Key(c, u)] = distance;
                }

                _queue.Enqueue((u, c), (distance, u, c));
            }

            _adjacent[a] = null;
            _adjacent[b] = null;
            if (_linkage != Linkage.Ward)
                _pairDistance.Remove(Key(a, b));
        }

        private double Distance(int a, int b)
            => _linkage == Linkage.Ward ? WardDistance(a, b) : Direct(a, b);

        /// <summary>
        /// Ward's distance from sizes and sums: <c>sqrt(2 n_a n_b / (n_a + n_b)) * |m_a - m_b|</c>,
        /// the square root of twice the increase in within-cluster sum of squares.
        /// The same scale as SciPy's and scikit-learn's, so a threshold means the
        /// same thing to all three.
        /// </summary>
        private double WardDistance(int a, int b)
        {
            double na = _size[a];
            double nb = _size[b];
            double squares = 0.0;

            for (int j = 0; j < _d; j++)
            {
                double delta = _sums[a, j] / na - _sums[b, j] / nb;
                squares += delta * delta;
            }

            return Math.Sqrt(2.0 * (na * nb / (na + nb)) * squares);
        }

        /// <summary>Single, complete or average linkage computed from every pair of samples.</summary>
        private double Direct(int a, int b)
        {
            double best = _linkage == Linkage.Single ? double.PositiveInfinity : 0.0;
            double total = 0.0;

            foreach (int i in _members[a]!)
            {
                foreach (int j in _members[b]!)
                {
                    double distance = Math.Sqrt(KMeans.SquaredDistance(_x, i, _x, j, _d));
                    switch (_linkage)
                    {
                        case Linkage.Single:
                            best = Math.Min(best, distance);
                            break;
                        case Linkage.Complete:
                            best = Math.Max(best, distance);
                            break;
                        default:
                            total += distance;
                            break;
                    }
                }
            }

            return _linkage == Linkage.Average
                ? total / ((double)_members[a]!.Count * _members[b]!.Count)
                : best;
        }

        private long Key(int a, int b) => (long)Math.Min(a, b) * (2 * _n) + Math.Max(a, b);
    }
}
