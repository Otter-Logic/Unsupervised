namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// One labelling of the samples handed to <see cref="ConsensusClustering"/>: a
/// name to report it under, a label per sample, and how much its vote counts.
/// </summary>
/// <param name="Name">What produced it, as a user should read it — "Spectral", "Hierarchical".</param>
/// <param name="Labels">
/// One label per sample. Negative is unplaced, and an unplaced sample abstains:
/// the view says nothing about which samples it belongs with, rather than saying
/// it belongs with none.
/// </param>
/// <param name="Weight">
/// How much this view's vote counts relative to the others. Zero leaves it out;
/// the default treats every view alike.
/// </param>
public sealed record ClusterView(string Name, int[] Labels, double Weight = 1.0);
