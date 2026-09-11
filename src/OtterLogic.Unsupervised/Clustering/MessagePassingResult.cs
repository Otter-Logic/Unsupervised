namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// A message-passing clustering: the partition, the smoothed features it was
/// drawn in, and what each cluster looks like before and after smoothing.
/// </summary>
public sealed class MessagePassingResult
{
    internal MessagePassingResult(
        int[] labels, double[,] embedding, double[,] centroids, double[,] inputMeans, double inertia)
    {
        Labels = labels;
        Embedding = embedding;
        Centroids = centroids;
        InputMeans = inputMeans;
        Inertia = inertia;
    }

    /// <summary>Cluster index per sample, largest cluster first.</summary>
    public int[] Labels { get; }

    /// <summary>
    /// The features after message passing, n x d — each sample's own blended with
    /// its neighbourhood's. Exposed because it is worth clustering by other means
    /// too: a mixture over this gives soft assignments that already know about the
    /// graph.
    /// </summary>
    public double[,] Embedding { get; }

    /// <summary>Cluster centres, k x d, in the smoothed space the partition was drawn in.</summary>
    public double[,] Centroids { get; }

    /// <summary>
    /// Mean of each cluster's <em>original</em> features, k x d.
    /// <para>
    /// The one to read when naming a cluster. Smoothing drew every centre towards
    /// its neighbours', so <see cref="Centroids"/> understates how distinct the
    /// clusters are; this reports what the samples in each actually were.
    /// </para>
    /// </summary>
    public double[,] InputMeans { get; }

    /// <summary>Total squared distance from every smoothed sample to its cluster's centre.</summary>
    public double Inertia { get; }

    /// <summary>Number of clusters.</summary>
    public int ClusterCount => Centroids.GetLength(0);

    /// <summary>Number of samples fitted.</summary>
    public int SampleCount => Labels.Length;

    /// <summary>Sample indices bucketed by cluster.</summary>
    public int[][] Clusters() => ClusterLabels.Members(Labels, ClusterCount);
}
