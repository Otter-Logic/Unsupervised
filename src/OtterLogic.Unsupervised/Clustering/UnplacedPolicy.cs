namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// What to do with the samples a clustering declined to place — HDBSCAN's noise,
/// spectral clustering's isolated samples. See <see cref="ClusterLabels.ResolveUnplaced"/>.
/// <para>
/// A choice for the caller, not a default to hide, because the right answer
/// depends entirely on what happens next. Reading the data, a sample that fits
/// nowhere is a finding and should stay visible. Acting on it, every sample still
/// needs a group — and whether it should get one of its own or join the nearest
/// depends on whether being unusual is a reason to treat it separately.
/// </para>
/// </summary>
public enum UnplacedPolicy
{
    /// <summary>Keep them unplaced, labelled <c>-1</c>. The honest reading of the data.</summary>
    Leave,

    /// <summary>
    /// Give each one a group of its own, numbered after every existing group. The
    /// cautious choice when every sample must be acted on and an unusual one should
    /// not be treated like its neighbours.
    /// </summary>
    OwnGroup,

    /// <summary>
    /// File each one with the group whose mean it is nearest. For when every
    /// sample must be placed and being unusual is not a reason to single it out.
    /// </summary>
    Nearest,
}
