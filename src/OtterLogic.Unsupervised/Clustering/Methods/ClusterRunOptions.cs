namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// What <see cref="ClusterRun"/> does around the method: the preparation before it
/// and the readings after it.
/// </summary>
public sealed record ClusterRunOptions
{
    /// <summary>
    /// Bring every column to the same scale before fitting. On by default.
    /// <para>
    /// Not an opinion about the data: every method here measures distances, so
    /// without this a length in millimetres beside an angle in radians is a fit
    /// that has only looked at the length. Off is for columns that already share
    /// a scale that means something.
    /// </para>
    /// </summary>
    public bool Standardise { get; init; } = true;

    /// <summary>
    /// Dimensions of the map laid out for looking at the clusters: 2 or 3, or 0 for
    /// none. Zero by default because the map costs a full distance matrix and is
    /// only ever for looking at.
    /// </summary>
    public int MapDimensions { get; init; }

    /// <summary>
    /// Whether to describe what sets each cluster apart, feature by feature, for the
    /// report. On by default: a grouping nobody can name is a grouping nobody acts on.
    /// </summary>
    public bool Explain { get; init; } = true;

    /// <summary>How many features to name per cluster in the explanation.</summary>
    public int TopFeatures { get; init; } = 3;

    public void Validate()
    {
        if (MapDimensions is not (0 or 2 or 3))
            throw new ArgumentOutOfRangeException(nameof(MapDimensions), MapDimensions,
                "A map is for looking at, so it has 2 or 3 dimensions; 0 for none.");
        if (TopFeatures < 1)
            throw new ArgumentOutOfRangeException(nameof(TopFeatures), TopFeatures, "Name at least one feature.");
    }
}
