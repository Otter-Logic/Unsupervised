using OtterLogic.MachineLearning.Distances;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// k-means by Lloyd's algorithm from a greedy k-means++ start.
/// <para>
/// This lived inside <see cref="GaussianMixture"/> first, as the seeder EM
/// starts from, and it is public here because it is worth having on its own —
/// not copied. There is one implementation, and the mixture still calls it.
/// A second copy would be a second thing to keep in step with the parity tests.
/// </para>
/// <para>
/// Hard assignment is the whole point of it and the whole limitation. Every
/// sample belongs to exactly one cluster with no notion of how well it fits, and
/// the implicit model is round clusters of roughly equal size. When that
/// describes the data it is faster and steadier than a mixture; when it does
/// not, it will still return an answer and the answer will be confidently wrong.
/// <see cref="GaussianMixture"/> is the one to reach for when clusters overlap,
/// and <see cref="Hdbscan"/> when they are irregular or there is noise.
/// </para>
/// </summary>
public static class KMeans
{
    /// <summary>
    /// Fits a partition, restarting from several k-means++ starts and keeping
    /// whichever reached the lowest inertia. Clusters come back largest first.
    /// </summary>
    /// <param name="x">n x d data, rows are samples. Expected to be standardised already.</param>
    /// <param name="options">Fit settings. <see cref="KMeansOptions.Seed"/> makes this deterministic.</param>
    public static KMeansResult Fit(double[,] x, KMeansOptions options)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(options);

        int n = x.GetLength(0);
        options.Validate(n);

        var seeds = new Random(options.Seed);
        KMeansResult? best = null;

        for (int restart = 0; restart < options.Restarts; restart++)
        {
            var rng = new Random(seeds.Next());
            var candidate = FitOnce(x, options.Clusters, rng, options.MaxIterations);

            // Strictly less, so the earliest restart wins a tie. With a fixed
            // seed that makes the whole thing reproducible down to which start
            // produced the answer.
            if (best is null || candidate.Inertia < best.Inertia)
                best = candidate;
        }

        return Canonical(best!);
    }

    /// <summary>
    /// Renumbers clusters largest first, ties to the one holding the lowest
    /// sample, and reorders the centres to match.
    /// <para>
    /// Lloyd's algorithm numbers a cluster by whichever seed it grew from, so a
    /// small change upstream permutes the numbers and every colour downstream jumps
    /// for no reason a user can see. Measured through the 6DOF classifier: of the
    /// six orderings of three families sized 20, 30 and 45, four came back
    /// numbered other than largest first. Done here, once, so every caller —
    /// the selector, spectral clustering, message passing, the component — gets
    /// the same numbering without each re-deriving it.
    /// </para>
    /// </summary>
    private static KMeansResult Canonical(KMeansResult fit)
    {
        var labels = ClusterLabels.Canonical(fit.Labels, out var mapping);

        int k = fit.ClusterCount;
        int d = fit.Centroids.GetLength(1);

        // A centre that ended with no samples has no size to rank by; it goes
        // after the populated ones, keeping k centres as asked for.
        var order = new int[k];
        foreach (var (from, to) in mapping)
            order[to] = from;

        int next = mapping.Count;
        for (int c = 0; c < k; c++)
            if (!mapping.ContainsKey(c))
                order[next++] = c;

        var centroids = new double[k, d];
        for (int position = 0; position < k; position++)
            for (int j = 0; j < d; j++)
                centroids[position, j] = fit.Centroids[order[position], j];

        return new KMeansResult(labels, centroids, fit.Inertia, fit.Iterations, fit.Converged);
    }

    private static KMeansResult FitOnce(double[,] x, int k, Random rng, int maxIterations)
    {
        var centres = PlusPlusSeeds(x, k, rng);
        var (labels, iterations, converged) = Lloyd(x, centres, k, maxIterations);

        double inertia = 0.0;
        for (int i = 0; i < x.GetLength(0); i++)
            inertia += Euclidean.Squared(x, i, centres, labels[i]);

        return new KMeansResult(labels, centres, inertia, iterations, converged);
    }

    /// <summary>
    /// Lloyd's algorithm: assign every sample to its nearest centre, move each
    /// centre to the mean of what it holds, repeat until nothing moves.
    /// <para>
    /// Centres are updated in place, so the caller reads the fitted centres from
    /// the array it passed in.
    /// </para>
    /// </summary>
    internal static (int[] Labels, int Iterations, bool Converged) Lloyd(
        double[,] x, double[,] centres, int k, int maxIterations)
    {
        int n = x.GetLength(0);

        var labels = new int[n];
        int iteration = 0;
        bool converged = false;

        for (iteration = 1; iteration <= maxIterations; iteration++)
        {
            bool moved = false;

            for (int i = 0; i < n; i++)
            {
                int best = 0;
                double bestDistance = double.MaxValue;

                for (int c = 0; c < k; c++)
                {
                    double distance = Euclidean.Squared(x, i, centres, c);
                    if (distance < bestDistance)
                    {
                        bestDistance = distance;
                        best = c;
                    }
                }

                if (labels[i] != best)
                {
                    labels[i] = best;
                    moved = true;
                }
            }

            // The first pass always "moves", because every label starts at zero.
            if (iteration > 1 && !moved)
            {
                converged = true;
                break;
            }

            RecomputeCentres(x, labels, centres, k);
        }

        return (labels, Math.Min(iteration, maxIterations), converged);
    }

    /// <summary>
    /// Greedy k-means++ seeding.
    /// <para>
    /// Plain k-means++ draws one candidate per seed, with probability
    /// proportional to its squared distance from the nearest seed already
    /// chosen. The greedy variant draws several, evaluates what each would do to
    /// the total squared distance, and keeps the best — which costs a few
    /// hundred extra operations and measurably improves the optimum a subsequent
    /// EM climbs to.
    /// </para>
    /// <para>
    /// This is not a micro-optimisation. Fitting five components to data with
    /// four real families, plain sampling landed on a local optimum roughly two
    /// per cent worse by BIC than scikit-learn reached, consistently. The number
    /// of trials below is scikit-learn's, and it closes that gap.
    /// </para>
    /// </summary>
    internal static double[,] PlusPlusSeeds(double[,] x, int k, Random rng)
    {
        int n = x.GetLength(0);
        int d = x.GetLength(1);

        int trials = 2 + (int)Math.Log(k);

        var centres = new double[k, d];
        var nearest = new double[n];

        int first = rng.Next(n);
        for (int j = 0; j < d; j++)
            centres[0, j] = x[first, j];

        double potential = 0.0;
        for (int i = 0; i < n; i++)
        {
            nearest[i] = Euclidean.Squared(x, i, centres, 0);
            potential += nearest[i];
        }

        for (int c = 1; c < k; c++)
        {
            int best = -1;
            double bestPotential = double.MaxValue;

            for (int trial = 0; trial < trials; trial++)
            {
                int index = SampleProportionally(nearest, potential, rng, n);

                double trialPotential = 0.0;
                for (int i = 0; i < n; i++)
                    trialPotential += Math.Min(nearest[i], Euclidean.Squared(x, i, x, index));

                if (trialPotential < bestPotential)
                {
                    bestPotential = trialPotential;
                    best = index;
                }
            }

            for (int j = 0; j < d; j++)
                centres[c, j] = x[best, j];

            potential = 0.0;
            for (int i = 0; i < n; i++)
            {
                nearest[i] = Math.Min(nearest[i], Euclidean.Squared(x, i, x, best));
                potential += nearest[i];
            }
        }

        return centres;
    }

    /// <summary>
    /// Draws an index with probability proportional to <paramref name="weights"/>.
    /// Falls back to uniform when every weight is zero, which happens when every
    /// point coincides with a seed already chosen — rare in general, and routine
    /// when the data repeats the same row hundreds of times.
    /// </summary>
    private static int SampleProportionally(double[] weights, double total, Random rng, int n)
    {
        if (total <= 0.0 || double.IsNaN(total))
            return rng.Next(n);

        double target = rng.NextDouble() * total;
        double cumulative = 0.0;

        for (int i = 0; i < n; i++)
        {
            cumulative += weights[i];
            if (cumulative >= target)
                return i;
        }

        return n - 1;
    }

    private static void RecomputeCentres(double[,] x, int[] labels, double[,] centres, int k)
    {
        int n = x.GetLength(0);
        int d = x.GetLength(1);

        var counts = new int[k];
        var sums = new double[k, d];

        for (int i = 0; i < n; i++)
        {
            counts[labels[i]]++;
            for (int j = 0; j < d; j++)
                sums[labels[i], j] += x[i, j];
        }

        for (int c = 0; c < k; c++)
        {
            if (counts[c] == 0)
            {
                // An empty cluster leaves a component with no data and a
                // degenerate covariance. Move it onto whichever point is worst
                // served by its current centre — the same repair k-means
                // implementations generally make, and it keeps k honest.
                MoveToWorstServedPoint(x, labels, centres, c);
                continue;
            }

            for (int j = 0; j < d; j++)
                centres[c, j] = sums[c, j] / counts[c];
        }
    }

    private static void MoveToWorstServedPoint(double[,] x, int[] labels, double[,] centres, int empty)
    {
        int n = x.GetLength(0);
        int d = x.GetLength(1);

        int worst = 0;
        double worstDistance = -1.0;

        for (int i = 0; i < n; i++)
        {
            double distance = Euclidean.Squared(x, i, centres, labels[i]);

            if (distance > worstDistance)
            {
                worstDistance = distance;
                worst = i;
            }
        }

        for (int j = 0; j < d; j++)
            centres[empty, j] = x[worst, j];

        labels[worst] = empty;
    }
}
