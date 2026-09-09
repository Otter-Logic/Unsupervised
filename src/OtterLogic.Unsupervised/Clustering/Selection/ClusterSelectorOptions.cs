namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Settings for <see cref="ClusterSelector"/>. Every one has a default, and the
/// intended use is to pass none of them.
/// <para>
/// They exist so the behaviour is inspectable and testable, not because a caller
/// is expected to reach for them. Anyone who wants to drive the individual
/// models is better served by <see cref="KMeans"/>, <see cref="GaussianMixture"/>
/// and <see cref="Hdbscan"/> directly, which expose everything.
/// </para>
/// <para>
/// The thresholds below are about the shape of a point cloud — how much of it a
/// density model leaves unplaced, how well a round partition scores, how many
/// samples land on a boundary. That is a property of the data and not of any one
/// discipline, which is why they live here rather than in the toolkit that calls
/// this. A caller whose data behaves differently overrides them.
/// </para>
/// </summary>
public sealed record ClusterSelectorOptions
{
    /// <summary>Lowest number of clusters to consider.</summary>
    public int MinimumGroups { get; init; } = 2;

    /// <summary>
    /// Highest number of clusters to consider.
    /// <para>
    /// Ten, because every cluster is something somebody has to interpret and act
    /// on. Data that genuinely wants more than ten is data where a grouping is
    /// not the useful abstraction.
    /// </para>
    /// </summary>
    public int MaximumGroups { get; init; } = 10;

    /// <summary>
    /// Smallest group HDBSCAN will call a cluster. Null derives it from the
    /// sample count — see <see cref="HdbscanOptions.DefaultMinimumClusterSize"/>.
    /// </summary>
    public int? MinimumClusterSize { get; init; }

    /// <summary>
    /// Forces a model instead of choosing one. Null runs the comparison, which
    /// is the point of this class.
    /// </summary>
    public ClusteringModel? Model { get; init; }

    /// <summary>
    /// Seed for every model that starts randomly. Fixed, so a Grasshopper
    /// re-solve returns the same clusters.
    /// </summary>
    public int Seed { get; init; } = 1;

    /// <summary>
    /// Noise fraction at or above which the data is treated as messy enough to
    /// want HDBSCAN, provided it found clusters at all.
    /// </summary>
    public double MessyNoiseFloor { get; init; } = 0.05;

    /// <summary>
    /// Noise fraction above which HDBSCAN is judged to have found nothing rather
    /// than to have found outliers, and is passed over.
    /// <para>
    /// A third is already generous. Genuine one-off samples are a minority of any
    /// real dataset; when a density model cannot place more than that, the honest
    /// reading is that the data has no density structure to find, not that most
    /// of it is exceptional. Measured on clusters overlapping so heavily they are
    /// one diffuse cloud, HDBSCAN left 47 to 77 per cent unplaced — which is the
    /// failure this ceiling exists to catch.
    /// </para>
    /// </summary>
    public double MessyNoiseCeiling { get; init; } = 0.35;

    /// <summary>
    /// Silhouette below which no round partition of this data is any good, which
    /// is itself evidence the clusters are not clean.
    /// </summary>
    public double CleanSilhouetteFloor { get; init; } = 0.25;

    /// <summary>
    /// Share of samples sitting between two clusters, above which the clusters
    /// are judged to overlap and the mixture wins.
    /// <para>
    /// One sample in ten being a genuine boundary case is enough to matter: a
    /// hard partition would file all of them silently, and the whole value of a
    /// mixture is that it says which ones they are. See
    /// <see cref="ClusterCandidate.AmbiguousFraction"/> for why this is a tail
    /// measure rather than a mean.
    /// </para>
    /// </summary>
    public double OverlapAmbiguousShare { get; init; } = 0.10;

    /// <summary>
    /// Checks these settings against the sample count they will be used on.
    /// Public because a caller that prepares its own data wants the complaint
    /// before it spends time on the preprocessing.
    /// </summary>
    public void Validate(int sampleCount)
    {
        if (MinimumGroups < 2)
            throw new ArgumentOutOfRangeException(nameof(MinimumGroups), MinimumGroups,
                "Comparing partitions needs at least two clusters.");
        if (MaximumGroups < MinimumGroups)
            throw new ArgumentOutOfRangeException(nameof(MaximumGroups), MaximumGroups,
                $"Maximum groups ({MaximumGroups}) is below minimum ({MinimumGroups}).");
        if (MinimumGroups > sampleCount)
            throw new ArgumentOutOfRangeException(nameof(MinimumGroups), MinimumGroups,
                $"Cannot ask for {MinimumGroups} clusters from {sampleCount} samples.");
    }
}
