namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// How the distance between two clusters is measured, which decides what shape of
/// cluster a hierarchy prefers to build.
/// </summary>
public enum Linkage
{
    /// <summary>
    /// Merge whichever pair adds least to the total within-cluster variance.
    /// Builds compact clusters of similar size — the tree-shaped cousin of
    /// k-means, and the default for that reason.
    /// </summary>
    Ward,

    /// <summary>
    /// Distance between the two farthest samples. Every cluster ends up with a
    /// bounded diameter, so nothing in a group is far from anything else in it.
    /// </summary>
    Complete,

    /// <summary>
    /// Mean distance over every pair across the two clusters. A compromise
    /// between the extremes, and less swayed by any single outlying sample.
    /// </summary>
    Average,

    /// <summary>
    /// Distance between the two nearest samples. Follows chains of close samples
    /// however long, so it finds elongated clusters — and will join two genuine
    /// clusters through a thin bridge of samples. HDBSCAN is this made robust to
    /// exactly that.
    /// </summary>
    Single,
}
