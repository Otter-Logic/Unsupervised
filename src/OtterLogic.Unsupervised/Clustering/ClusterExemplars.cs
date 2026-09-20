using OtterLogic.MachineLearning.Distances;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// The most typical member of each cluster — its medoid — and how far every
/// sample sits from its own.
/// <para>
/// A medoid rather than a mean because it is a real sample: the one to inspect,
/// draw or label on behalf of the rest. A mean is somewhere no sample is, and for
/// counts or categories often somewhere no sample could be.
/// </para>
/// </summary>
public static class ClusterExemplars
{
    /// <summary>
    /// Per cluster, the member with the least total distance to the other members;
    /// −1 for a label with no members. Ties go to the lower sample index.
    /// </summary>
    /// <param name="x">n x d, rows are samples, in the space the clustering was made in.</param>
    /// <param name="labels">Cluster per sample; negative is unplaced and never an exemplar.</param>
    public static int[] Medoids(double[,] x, int[] labels)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(labels);
        if (labels.Length != x.GetLength(0))
            throw new ArgumentException($"{labels.Length} labels for {x.GetLength(0)} samples.", nameof(labels));

        var members = ClusterLabels.Members(labels);
        var medoids = new int[members.Length];

        for (int c = 0; c < members.Length; c++)
        {
            medoids[c] = -1;
            double best = double.PositiveInfinity;

            foreach (int i in members[c])
            {
                double total = 0.0;
                foreach (int j in members[c])
                    total += Euclidean.Between(x, i, x, j);

                if (total < best)
                {
                    best = total;
                    medoids[c] = i;
                }
            }
        }

        return medoids;
    }

    /// <summary>
    /// Per sample, its distance from its own cluster's medoid; zero for the medoid
    /// itself and NaN for an unplaced sample. Zero across a whole cluster means its
    /// members are identical; small but not zero is a near-identical variant.
    /// </summary>
    public static double[] DistanceFromMedoid(double[,] x, int[] labels, int[] medoids)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(labels);
        ArgumentNullException.ThrowIfNull(medoids);

        var distance = new double[labels.Length];
        for (int i = 0; i < labels.Length; i++)
            distance[i] = labels[i] < 0 || labels[i] >= medoids.Length || medoids[labels[i]] < 0
                ? double.NaN
                : Euclidean.Between(x, i, x, medoids[labels[i]]);

        return distance;
    }
}
