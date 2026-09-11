namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// One merge in a dendrogram, in SciPy's <c>linkage</c> layout: samples are nodes
/// <c>0..n-1</c>, and merge t creates node <c>n + t</c>.
/// </summary>
/// <param name="Left">The lower-numbered of the two nodes merged.</param>
/// <param name="Right">The higher-numbered of the two nodes merged.</param>
/// <param name="Distance">The linkage distance at which they merged.</param>
/// <param name="Size">Samples under the new node.</param>
public readonly record struct ClusterMerge(int Left, int Right, double Distance, int Size);

/// <summary>
/// A fitted hierarchy: the whole dendrogram, from every sample on its own to one
/// cluster holding everything, and the means to cut it anywhere.
/// <para>
/// The tree is the answer, not any one cut of it. A flat method commits to a
/// granularity before it sees the data; this defers the choice, so the same fit
/// can be cut coarse for an overview and fine for the groups somebody will
/// actually act on — and the fine groups are guaranteed to nest inside the
/// coarse ones, which no pair of separate flat fits promises.
/// </para>
/// </summary>
public sealed class HierarchicalClusteringResult
{
    internal HierarchicalClusteringResult(
        ClusterMerge[] merges, int sampleCount, Linkage linkage, int graphComponents)
    {
        Merges = merges;
        SampleCount = sampleCount;
        Linkage = linkage;
        GraphComponents = graphComponents;
    }

    /// <summary>
    /// Every merge, n - 1 of them, in the order they were made.
    /// <para>
    /// Without a connectivity graph the distances never decrease, and this is
    /// exactly SciPy's linkage matrix. With one they can: joining two clusters can
    /// bring a third into reach that was closer than the merge just made, but was
    /// not allowed to join until its neighbour did.
    /// </para>
    /// </summary>
    public IReadOnlyList<ClusterMerge> Merges { get; }

    /// <summary>Number of samples fitted.</summary>
    public int SampleCount { get; }

    /// <summary>The linkage the tree was built with.</summary>
    public Linkage Linkage { get; }

    /// <summary>
    /// Connected components of the connectivity graph, or one when there was none.
    /// <para>
    /// A constrained tree cannot merge across a disconnection, so once every
    /// component has become one cluster the remaining roots are merged without
    /// the constraint, to finish the tree. The last <c>GraphComponents - 1</c>
    /// merges are those — so any cut into fewer clusters than there are
    /// components groups samples the graph never related.
    /// </para>
    /// </summary>
    public int GraphComponents { get; }

    /// <summary>
    /// Cuts the tree into exactly <paramref name="clusters"/> clusters by undoing
    /// the last merges. Largest cluster first.
    /// </summary>
    /// <param name="clusters">Between 1 and <see cref="SampleCount"/>.</param>
    public int[] Cut(int clusters)
    {
        if (clusters < 1 || clusters > SampleCount)
            throw new ArgumentOutOfRangeException(nameof(clusters), clusters,
                $"Can cut between 1 and {SampleCount} clusters.");

        int n = SampleCount;
        var parent = new int[2 * n - 1];
        Array.Fill(parent, -1);

        for (int t = 0; t < n - clusters; t++)
        {
            parent[Merges[t].Left] = n + t;
            parent[Merges[t].Right] = n + t;
        }

        var roots = new int[n];
        for (int i = 0; i < n; i++)
        {
            int node = i;
            while (parent[node] >= 0)
                node = parent[node];

            roots[i] = node;
        }

        return ClusterLabels.Canonical(roots);
    }

    /// <summary>
    /// Cuts the tree by distance: every merge at or above
    /// <paramref name="threshold"/> is undone. Largest cluster first.
    /// <para>
    /// Counted rather than traced, as scikit-learn does — the number of merges at
    /// or above the threshold, plus one, is the cluster count. For an unconstrained
    /// tree that is the ordinary horizontal cut. For a constrained tree with a
    /// merge below one made earlier, it still returns a nested partition, where
    /// tracing the tree would not.
    /// </para>
    /// </summary>
    public int[] CutAtDistance(double threshold) => Cut(ClustersAtDistance(threshold));

    /// <summary>How many clusters <see cref="CutAtDistance"/> would return.</summary>
    public int ClustersAtDistance(double threshold)
        => Merges.Count(m => m.Distance >= threshold) + 1;
}
