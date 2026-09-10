namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// How far and how hard <see cref="MessagePassing"/> spreads a signal over a
/// graph.
/// <para>
/// The defaults are Simplified Graph Convolution's (Wu et al., 2019): two hops of
/// plain averaging. On a planted four-community graph whose features alone
/// barely beat chance (k-means ARI 0.04), that recovered the communities at ARI
/// 0.97 — see <c>MessagePassingTests</c>, which prints the whole sweep.
/// </para>
/// </summary>
public sealed record PropagationOptions
{
    /// <summary>
    /// Rounds of message passing — how many edges away a sample's signal can reach.
    /// Zero returns the input untouched.
    /// <para>
    /// Two reads naturally as "a sample's neighbourhood": its neighbours and
    /// theirs. A few more can help where communities are large and loosely knit —
    /// four hops reached ARI 1.00 on the planted graph above — but past that,
    /// plain averaging starts to erase what it found. See <see cref="Retention"/>.
    /// </para>
    /// </summary>
    public int Hops { get; init; } = 2;

    /// <summary>
    /// Share of each sample's own original signal restored after every round,
    /// between 0 and 1. Zero by default.
    /// <para>
    /// Without it, repeated averaging converges on the same value for every sample
    /// in a connected graph — oversmoothing, the classic failure of deep message
    /// passing. Measured on the planted graph, zero retention scored ARI 0.99 at
    /// sixteen hops, 0.48 at thirty-two and 0.00 at sixty-four. Restoring a share
    /// each round (APPNP, Klicpera et al., 2019) makes the limit a personalised
    /// average instead of one global mean: at 0.1 the same sweep held at 0.73 from
    /// thirty-two hops to sixty-four, and would hold at any number.
    /// </para>
    /// <para>
    /// The cost is that a sample's own features keep a fixed say however many
    /// hops run, which is the point when they are informative and a drag when they
    /// are mostly noise — at two hops, 0.1 scored 0.72 on that graph against 0.97
    /// for none. So: leave it at zero for a few hops, and raise it when running
    /// many, or when each sample's own behaviour must not be averaged away.
    /// </para>
    /// </summary>
    public double Retention { get; init; }

    internal void Validate()
    {
        if (Hops < 0)
            throw new ArgumentOutOfRangeException(nameof(Hops), Hops, "Hops cannot be negative.");
        if (double.IsNaN(Retention) || Retention < 0.0 || Retention > 1.0)
            throw new ArgumentOutOfRangeException(nameof(Retention), Retention, "Retention is a share, between 0 and 1.");
    }
}
