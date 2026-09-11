namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// A fitted mixture: the parameters, the soft assignments, and enough
/// diagnostics to judge whether the fit is worth believing.
/// </summary>
public sealed class GaussianMixtureResult
{
    internal GaussianMixtureResult(
        double[] mixingWeights,
        double[,] means,
        double[][,] covariances,
        double[,] responsibilities,
        double logLikelihood,
        int iterations,
        bool converged,
        int parameterCount)
    {
        MixingWeights = mixingWeights;
        Means = means;
        Covariances = covariances;
        Responsibilities = responsibilities;
        LogLikelihood = logLikelihood;
        Iterations = iterations;
        Converged = converged;
        ParameterCount = parameterCount;
    }

    /// <summary>Mixing proportion of each component, summing to one.</summary>
    public double[] MixingWeights { get; }

    /// <summary>Component means, k x d, in whatever space the fit was run in.</summary>
    public double[,] Means { get; }

    /// <summary>
    /// Component covariances, k matrices of d x d. Diagonal and spherical fits
    /// return diagonal matrices rather than a compressed form, so callers do not
    /// have to branch on <see cref="CovarianceType"/>.
    /// </summary>
    public double[][,] Covariances { get; }

    /// <summary>
    /// Soft assignments, n x k. Row i sums to one and holds the posterior
    /// probability that sample i came from each component. This is the output
    /// that makes a mixture worth more than k-means.
    /// </summary>
    public double[,] Responsibilities { get; }

    /// <summary>Total log-likelihood of the data under the fitted model.</summary>
    public double LogLikelihood { get; }

    /// <summary>Mean log-likelihood per sample — scikit-learn's <c>score</c>.</summary>
    public double MeanLogLikelihood => LogLikelihood / SampleCount;

    /// <summary>EM iterations taken by the winning restart.</summary>
    public int Iterations { get; }

    /// <summary>Whether the winning restart met the tolerance before the iteration cap.</summary>
    public bool Converged { get; }

    /// <summary>Free parameters in the model, used by <see cref="Bic"/> and <see cref="Aic"/>.</summary>
    public int ParameterCount { get; }

    /// <summary>Number of samples fitted.</summary>
    public int SampleCount => Responsibilities.GetLength(0);

    /// <summary>Number of components.</summary>
    public int ComponentCount => MixingWeights.Length;

    /// <summary>
    /// Bayesian information criterion, lower is better. Penalises parameters by
    /// ln(n), so it prefers fewer components than AIC does — which is usually
    /// what you want when the components are meant to become real groups
    /// somebody has to act on.
    /// </summary>
    public double Bic => -2.0 * LogLikelihood + ParameterCount * Math.Log(SampleCount);

    /// <summary>Akaike information criterion, lower is better.</summary>
    public double Aic => -2.0 * LogLikelihood + 2.0 * ParameterCount;

    /// <summary>Hard assignment of each sample to its most probable component.</summary>
    public int[] Labels() => Posterior.ArgMax(Responsibilities);

    /// <summary>Sample indices in each component, by <see cref="Labels"/>.</summary>
    public int[][] Clusters() => ClusterLabels.Members(Labels(), ComponentCount);

    /// <summary>
    /// The highest responsibility for each sample, between 1/k and 1.
    /// <para>
    /// Read this as confidence. A sample sitting at 0.95 belongs where it was
    /// put; one at 0.4 is on a boundary between two components and is exactly the
    /// case somebody should look at rather than take on trust.
    /// </para>
    /// </summary>
    public double[] Confidence() => Posterior.RowMax(Responsibilities);

    /// <summary>
    /// The same fit with its components renumbered largest mixing weight first,
    /// ties to the lower original number. Weights, means, covariances and
    /// responsibilities move together; the likelihood and every diagnostic are
    /// unchanged, because it is the same model.
    /// <para>
    /// <see cref="GaussianMixture.Fit"/> leaves components in the order EM
    /// started them, which is the order the parity tests hold against
    /// scikit-learn, so it stays. Anything that puts the numbers in front of a
    /// user wants them stable instead — a small change upstream should not turn
    /// component zero into component two and recolour everything downstream —
    /// and the pipeline, the model selector and the Grasshopper component had
    /// each written this renumbering out for themselves.
    /// </para>
    /// </summary>
    public GaussianMixtureResult OrderedByWeight()
    {
        int k = ComponentCount;
        int n = SampleCount;
        int d = Means.GetLength(1);

        var order = Enumerable.Range(0, k)
            .OrderByDescending(c => MixingWeights[c])
            .ThenBy(c => c)
            .ToArray();

        var means = new double[k, d];
        for (int c = 0; c < k; c++)
            for (int j = 0; j < d; j++)
                means[c, j] = Means[order[c], j];

        var responsibilities = new double[n, k];
        for (int i = 0; i < n; i++)
            for (int c = 0; c < k; c++)
                responsibilities[i, c] = Responsibilities[i, order[c]];

        return new GaussianMixtureResult(
            order.Select(c => MixingWeights[c]).ToArray(),
            means,
            order.Select(c => Covariances[c]).ToArray(),
            responsibilities,
            LogLikelihood,
            Iterations,
            Converged,
            ParameterCount);
    }
}
