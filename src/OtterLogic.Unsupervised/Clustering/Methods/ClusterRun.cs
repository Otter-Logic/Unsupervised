using OtterLogic.Core;
using OtterLogic.MachineLearning.Decomposition;
using OtterLogic.MachineLearning.Preprocessing;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Samples and a method in, a clustering a person can act on out: standardise, fit,
/// then read the result back in the units the samples arrived in.
/// <para>
/// The one entry point the data component calls, so every decision between a
/// tree of numbers and a coloured model lives here rather than in an adaptor —
/// which columns were dropped, where the centres are in real units, how well
/// separated the clusters are, what sets each one apart. Any method goes through
/// the same call, and a toolkit that wants exactly what the component does calls
/// this and gets it.
/// </para>
/// </summary>
public static class ClusterRun
{
    /// <summary>
    /// Clusters <paramref name="data"/> with <paramref name="method"/>, or chooses a
    /// method when none is given.
    /// </summary>
    /// <param name="data">n x d raw values, one row per sample, in whatever units they have.</param>
    /// <param name="method">The method, or null to fit K-Means, a mixture and HDBSCAN and keep the one the data supports.</param>
    /// <param name="options">What to do around the method; null for the defaults.</param>
    /// <exception cref="ArgumentException">The data or the method cannot be used as given; the message says why.</exception>
    public static ClusterRunResult Fit(double[,] data, ClusteringMethod? method = null, ClusterRunOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(data);
        method ??= new AutoMethod();
        options ??= new ClusterRunOptions();
        options.Validate();

        int n = data.GetLength(0);
        int d = data.GetLength(1);
        if (n < 2)
            throw new ArgumentException($"Need at least two samples to cluster; got {n}.", nameof(data));
        if (d < 1)
            throw new ArgumentException("The samples have no values.", nameof(data));

        // Before any data is prepared, so a bad setting costs nothing but the
        // message. Standardising first would spend the whole pipeline to say the
        // same thing later.
        method.Validate(n);

        var notes = new List<Note>();
        double[,] fitted = data;
        int[] kept = Enumerable.Range(0, d).ToArray();

        if (options.Standardise)
        {
            FeaturePipeline pipeline;
            try
            {
                pipeline = FeaturePipeline.Fit(data);
            }
            catch (InvalidOperationException)
            {
                // The pipeline refuses to fit with no column left; from here that
                // is a complaint about the data handed in, and it is said in the
                // words a user of this call sees.
                throw new ArgumentException(
                    "Every column has the same value in every sample, so there is nothing to cluster on.", nameof(data));
            }

            kept = pipeline.KeptColumns;
            if (kept.Length < d)
            {
                var dropped = Enumerable.Range(0, d).Except(kept).Select(j => j.ToString());
                notes.Add(Note.Remark(
                    $"Column(s) {string.Join(", ", dropped)} never change across the samples and were left out of the fit."));
            }

            fitted = pipeline.Transform(data);
        }

        var outcome = method.Fit(fitted);
        notes.AddRange(outcome.Notes);

        // In the units the samples arrived in, over every column: a centre nobody
        // can read is a centre nobody acts on, and a dropped column still has a
        // value there — the same one for every cluster.
        var centres = ClusterLabels.Means(data, outcome.Labels, outcome.ClusterCount);

        int placedClusters = outcome.Members().Count(m => m.Length > 0);
        double silhouette = placedClusters >= 2 ? ClusterQuality.Silhouette(fitted, outcome.Labels) : double.NaN;
        double daviesBouldin = placedClusters >= 2 ? ClusterQuality.DaviesBouldin(fitted, outcome.Labels) : double.NaN;

        if (placedClusters < 2)
            notes.Add(Note.Remark(
                placedClusters == 0
                    ? "No cluster holds a sample, so there is no separation to score and nothing to explain."
                    : "Everything placed is in one cluster, so there is no separation to score and nothing "
                      + "to explain. Ask for more clusters, or a method that finds its own count."));

        if (outcome.UnplacedCount > 0)
            notes.Add(Note.Remark(
                $"{outcome.UnplacedCount} sample(s) are in no cluster, labelled -1. Unplaced lists them; "
                + "they are the outliers, and often the interesting ones."));

        double[,]? map = null;
        if (options.MapDimensions > 0)
        {
            var scaling = MultidimensionalScaling.FromFeatures(
                fitted, new MultidimensionalScalingOptions { Dimensions = options.MapDimensions });
            map = scaling.Coordinates;
            if (scaling.Stress > 0.2)
                notes.Add(Note.Remark(
                    $"The map is a rough one (stress {scaling.Stress:0.00}): the samples need more than "
                    + $"{options.MapDimensions} dimensions to lay out faithfully. Distances on it are indicative."));
        }

        // Two placed clusters is exactly what Describe needs — it refuses an empty
        // labelling and one cluster holding everything, and neither can happen past
        // the guard — so a failure here is a real mismatch and is left to surface.
        GroupSignatureResult? signature = options.Explain && placedClusters >= 2
            ? GroupSignature.Describe(data, outcome.Labels)
            : null;

        return new ClusterRunResult(outcome, centres, silhouette, daviesBouldin, map, signature,
            options.Standardise, kept, notes, options.TopFeatures);
    }
}
