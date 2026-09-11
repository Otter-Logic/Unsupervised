using OtterLogic.MachineLearning.Distances;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Fits all three clustering models to the same data and picks the one the
/// evidence supports, then says which and why.
/// <para>
/// The mechanism, and only the mechanism. It takes a matrix that is already
/// prepared — scaled, projected, whatever the caller decided — and returns a
/// choice between k-means, a Gaussian mixture and HDBSCAN. It knows nothing
/// about what the columns mean, which is what lets a structural toolkit and a
/// fabrication toolkit both use it with their own preprocessing and their own
/// vocabulary for the answer.
/// </para>
/// <para>
/// The judgement about *which preprocessing suits what data* is deliberately not
/// here. That is a property of the discipline, and it belongs in the toolkit
/// that has one.
/// </para>
/// </summary>
public static class ClusterSelector
{
    /// <summary>
    /// Clusters n prepared samples, choosing the model rather than being told
    /// one.
    /// </summary>
    /// <param name="data">n x d, one row per sample, already prepared.</param>
    /// <param name="options">Settings. The intended call passes none.</param>
    public static ClusterSelection Select(double[,] data, ClusterSelectorOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(data);
        options ??= new ClusterSelectorOptions();

        int n = data.GetLength(0);

        if (n < 4)
            throw new ArgumentException(
                $"Need at least four samples to compare clusterings; got {n}.", nameof(data));

        options.Validate(n);

        int maximumGroups = Math.Min(options.MaximumGroups, n - 1);
        int minimumGroups = Math.Min(options.MinimumGroups, maximumGroups);

        var kMeans = FitKMeans(data, minimumGroups, maximumGroups, options.Seed);
        var mixture = FitMixture(data, minimumGroups, maximumGroups, options.Seed);
        var density = FitHdbscan(data, options.MinimumClusterSize);

        var candidates = new[] { kMeans, mixture, density };

        var (chosen, rationale) = Choose(options, kMeans, mixture, density);
        var winner = candidates.First(c => c.Model == chosen);

        return new ClusterSelection(chosen, rationale, winner, candidates, data);
    }

    /// <summary>
    /// k-means across the group range, keeping the count with the best
    /// silhouette.
    /// <para>
    /// Silhouette rather than inertia, because inertia falls monotonically as
    /// clusters are added and so cannot choose between counts. This is the one
    /// place a silhouette is the right referee: it is comparing k-means against
    /// itself, where its preference for round clusters is a constant.
    /// </para>
    /// </summary>
    private static ClusterCandidate FitKMeans(double[,] x, int minimum, int maximum, int seed)
    {
        int[]? bestLabels = null;
        double[,]? bestCentroids = null;
        double bestScore = double.NegativeInfinity;

        for (int k = minimum; k <= maximum; k++)
        {
            var fit = KMeans.Fit(x, new KMeansOptions { Clusters = k, Seed = seed });
            double score = ClusterQuality.Silhouette(x, fit.Labels);

            if (score > bestScore)
            {
                bestScore = score;
                bestLabels = fit.Labels;
                bestCentroids = fit.Centroids;
            }
        }

        var labels = bestLabels!;
        var confidence = CentroidMargin(x, labels, bestCentroids!);

        return new ClusterCandidate(
            ClusteringModel.KMeans,
            Groups(labels),
            labels,
            confidence,
            bestScore,
            ClusterQuality.DaviesBouldin(x, labels),
            0.0);
    }

    /// <summary>
    /// A Gaussian mixture across the group range, keeping the count with the
    /// lowest BIC.
    /// <para>
    /// BIC rather than silhouette, because the mixture has a likelihood and BIC
    /// uses it — and because scoring the mixture by a compactness measure would
    /// throw away exactly the elongated, overlapping solutions it exists to find.
    /// Every fit here is over the same data, so the BIC values are comparable to
    /// each other.
    /// </para>
    /// <para>
    /// Full covariance, not diagonal. Decorrelating the data as a whole says
    /// nothing about correlation inside a single cluster, and letting each
    /// component take its own orientation is the mixture's advantage over
    /// k-means. At three dimensions that costs six parameters per cluster.
    /// </para>
    /// </summary>
    private static ClusterCandidate FitMixture(double[,] x, int minimum, int maximum, int seed)
    {
        GaussianMixtureResult? best = null;
        double bestBic = double.PositiveInfinity;

        for (int k = minimum; k <= maximum; k++)
        {
            var fit = GaussianMixture.Fit(x, new GaussianMixtureOptions
            {
                Components = k,
                Covariance = CovarianceType.Full,
                Seed = seed,
            });

            if (fit.Bic < bestBic)
            {
                bestBic = fit.Bic;
                best = fit;
            }
        }

        // Largest first, so a small change upstream does not permute the
        // clusters and shuffle every colour downstream.
        var mixture = best!.OrderedByWeight();
        var labels = mixture.Labels();

        return new ClusterCandidate(
            ClusteringModel.GaussianMixture,
            Groups(labels),
            labels,
            mixture.Confidence(),
            ClusterQuality.Silhouette(x, labels),
            ClusterQuality.DaviesBouldin(x, labels),
            0.0);
    }

    /// <summary>
    /// HDBSCAN once. It is not swept, because it is not told how many clusters
    /// to find — the count is something it reports.
    /// </summary>
    private static ClusterCandidate FitHdbscan(double[,] x, int? minimumClusterSize)
    {
        int n = x.GetLength(0);
        var fit = Hdbscan.Fit(x, new HdbscanOptions
        {
            MinimumClusterSize = minimumClusterSize ?? HdbscanOptions.DefaultMinimumClusterSize(n),
        });

        return new ClusterCandidate(
            ClusteringModel.Hdbscan,
            fit.ClusterCount,
            fit.Labels,
            fit.Probabilities,
            ClusterQuality.Silhouette(x, fit.Labels),
            ClusterQuality.DaviesBouldin(x, fit.Labels),
            fit.NoiseFraction);
    }

    /// <summary>
    /// Picks a model, testing each hypothesis with the measure that can answer it
    /// rather than scoring all three on one number.
    /// <para>
    /// This ordering is deliberate and the reason it is not a leaderboard.
    /// Silhouette and Davies-Bouldin both reward compact, round, well-separated
    /// clusters — which is what k-means optimises — so ranking the three on them
    /// would hand k-means the result almost regardless of the data. Measured on
    /// two interleaved crescents, where HDBSCAN recovers the truth exactly and
    /// k-means fails badly, the silhouette still prefers k-means by 0.49 to 0.33.
    /// A referee that agrees with one player is not a referee.
    /// </para>
    /// <para>
    /// So: messiness is tested by what only HDBSCAN can measure, overlap by what
    /// only the mixture can measure, and cleanliness is what is left when neither
    /// fires — which is also the only question a silhouette is trustworthy on.
    /// </para>
    /// </summary>
    private static (ClusteringModel Model, string Rationale) Choose(
        ClusterSelectorOptions options,
        ClusterCandidate kMeans,
        ClusterCandidate mixture,
        ClusterCandidate density)
    {
        if (options.Model is { } forced)
            return (forced, "the model was specified rather than chosen.");

        // Messy: a real share of samples belong to no dense region. Above the
        // ceiling HDBSCAN has not found outliers, it has found nothing, and its
        // own noise fraction is what says so.
        bool outliers = density.Groups >= 2
            && density.NoiseFraction >= options.MessyNoiseFloor
            && density.NoiseFraction <= options.MessyNoiseCeiling;

        if (outliers)
            return (ClusteringModel.Hdbscan,
                $"{density.NoiseFraction:P0} of samples sit outside every dense cluster, so a model "
                + "that can leave one unplaced describes this better than one that must file "
                + "everything.");

        // Irregular: no round partition of this data scores well at any count,
        // which is evidence about the shape of the clusters rather than about
        // how many there are.
        bool irregular = density.Groups >= 2 && kMeans.Silhouette < options.CleanSilhouetteFloor;

        if (irregular)
            return (ClusteringModel.Hdbscan,
                $"no round partition scores well at any count (best silhouette {kMeans.Silhouette:0.00}), "
                + "so the clusters are not the shape k-means and a mixture assume.");

        // Overlap: samples sitting between two clusters. Only the mixture
        // measures this, because only the mixture assigns softly — and it is the
        // share of boundary samples that says so, not the average confidence,
        // which stays high even when clusters genuinely intermingle.
        if (mixture.AmbiguousFraction >= options.OverlapAmbiguousShare)
            return (ClusteringModel.GaussianMixture,
                $"{mixture.AmbiguousFraction:P0} of samples sit between two clusters rather than "
                + "inside one, so a soft assignment reports the structure where a hard split would "
                + "file them silently.");

        return (ClusteringModel.KMeans,
            $"clusters are clean and well separated (silhouette {kMeans.Silhouette:0.00}, "
            + $"{mixture.AmbiguousFraction:P0} of samples on a boundary), so the simplest model is "
            + "the honest one.");
    }

    /// <summary>
    /// Confidence for a hard partition: how much closer a sample is to its own
    /// centre than to the next nearest, scaled to 0..1.
    /// <para>
    /// k-means has no probability to report, but it does know whether a sample
    /// was a close call. Zero means the two nearest centres are equidistant.
    /// </para>
    /// </summary>
    private static double[] CentroidMargin(double[,] x, int[] labels, double[,] centroids)
    {
        int n = x.GetLength(0);
        int k = centroids.GetLength(0);

        var margin = new double[n];

        for (int i = 0; i < n; i++)
        {
            double own = double.MaxValue;
            double other = double.MaxValue;

            for (int c = 0; c < k; c++)
            {
                double distance = Euclidean.Between(x, i, centroids, c);

                if (c == labels[i])
                    own = distance;
                else if (distance < other)
                    other = distance;
            }

            margin[i] = other is double.MaxValue or 0.0 ? 1.0 : Math.Clamp((other - own) / other, 0.0, 1.0);
        }

        return margin;
    }

    private static int Groups(int[] labels)
    {
        int highest = -1;
        foreach (int label in labels)
            if (label > highest)
                highest = label;

        return highest + 1;
    }
}
