namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// The three clustering models <see cref="ClusterSelector"/> chooses between.
/// <para>
/// They are not three implementations of one idea. Each assumes something
/// different about what a cluster looks like, and the whole point of running all
/// three is that the data decides which assumption holds.
/// </para>
/// </summary>
public enum ClusteringModel
{
    /// <summary>
    /// k-means. Assumes round clusters of roughly equal size, and puts every
    /// sample in one. Fastest and steadiest when that is true.
    /// </summary>
    KMeans,

    /// <summary>
    /// Gaussian mixture. Allows clusters to be elongated, unequal, and to
    /// overlap, and reports how strongly each sample belongs. The one to use
    /// when samples sit between two clusters.
    /// </summary>
    GaussianMixture,

    /// <summary>
    /// HDBSCAN. Assumes clusters are dense regions of any shape, and refuses to
    /// place samples that belong to none. The one to use when there are genuine
    /// one-off samples or the clusters are irregular.
    /// </summary>
    Hdbscan,
}
