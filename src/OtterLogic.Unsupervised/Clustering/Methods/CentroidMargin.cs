using OtterLogic.MachineLearning.Distances;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// A confidence for a hard partition: how much nearer each sample is to its own
/// centre than to the next one.
/// <para>
/// Not a probability, and not comparable with a mixture's posterior. It is here
/// because a partition with no confidence at all leaves a user unable to find the
/// samples worth a second look, and the margin is the honest thing a centre-based
/// method can say about that. One for k-means and one for the k-means step inside
/// spectral clustering, and the selector's comparison uses the same number.
/// </para>
/// </summary>
internal static class CentroidMargin
{
    /// <param name="x">n x d samples in the space the centres are in.</param>
    /// <param name="labels">Cluster per sample; a negative label gets a margin of zero.</param>
    /// <param name="centroids">k x d centres, one per cluster.</param>
    /// <returns>Per sample, (distance to the nearest other centre - distance to own) / nearest other, clamped to [0, 1].</returns>
    public static double[] Of(double[,] x, int[] labels, double[,] centroids)
    {
        int n = x.GetLength(0);
        int k = centroids.GetLength(0);
        var margin = new double[n];

        for (int i = 0; i < n; i++)
        {
            if (labels[i] < 0)
                continue;

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

            // One cluster, or a sample sitting on another centre: nothing to measure
            // against, so it is called certain rather than divided by zero.
            margin[i] = other is double.MaxValue or 0.0 ? 1.0 : Math.Clamp((other - own) / other, 0.0, 1.0);
        }

        return margin;
    }
}
