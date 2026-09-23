namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// A clustering algorithm with its settings chosen, ready to be handed samples.
/// <para>
/// The mechanism behind one wire. A Grasshopper user picks a method component —
/// K-Means, HDBSCAN — sets its two or three settings, and wires the result into
/// the one component that holds the data. The method knows nothing about where the
/// samples come from; the data component knows nothing about how any method
/// works. Adding an algorithm is one record here and one small component there,
/// and the data component never changes.
/// </para>
/// <para>
/// A record rather than an interface so that a method is a value: two with the
/// same settings are equal, and one can be copied with one setting changed. Each
/// concrete record wraps the algorithm's own options record rather than replacing
/// it, so the algorithm's entry point stays the one every test and toolkit calls.
/// </para>
/// </summary>
public abstract record ClusteringMethod
{
    /// <summary>The algorithm's name as a user knows it: "K-Means", "HDBSCAN".</summary>
    public abstract string Name { get; }

    /// <summary>The name and the settings that matter, in one line for a report.</summary>
    public abstract string Describe();

    /// <summary>
    /// Complains about anything the fit would refuse, before any data is prepared for
    /// it. The sample count is the one thing every method's limits depend on.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">A setting is outside what the method accepts.</exception>
    /// <exception cref="ArgumentException">The method cannot run on this many samples.</exception>
    public abstract void Validate(int sampleCount);

    /// <summary>
    /// Fits to <paramref name="x"/> and returns the labelling with whatever the
    /// method can honestly say about it.
    /// </summary>
    /// <param name="x">
    /// n x d samples, rows are samples. Expected to be on a common scale already;
    /// <see cref="ClusterRun"/> standardises before calling this, and a caller that
    /// does not must know why.
    /// </param>
    public abstract ClusteringOutcome Fit(double[,] x);

    public sealed override string ToString() => Describe();
}
