using System.Globalization;
using System.Text;

namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Each group of a labelling against the rest of the samples, feature by
/// feature. Every matrix is groups x features, row g being group g, so it lines up
/// with a method's centroids and cluster buckets.
/// <para>
/// A label with no samples keeps its row, filled with zeros and a size of zero,
/// so position is always the label.
/// </para>
/// </summary>
public sealed class GroupSignatureResult
{
    internal GroupSignatureResult(
        int sampleCount,
        int[] sizes,
        double[,] means,
        double[,] spread,
        double[,] restMeans,
        double[,] effect,
        double[,] separation,
        int[][] ranking)
    {
        SampleCount = sampleCount;
        Sizes = sizes;
        Means = means;
        Spread = spread;
        RestMeans = restMeans;
        Effect = effect;
        Separation = separation;
        Ranking = ranking;
    }

    /// <summary>Number of samples described, unplaced ones included.</summary>
    public int SampleCount { get; }

    /// <summary>Number of groups — one more than the largest label.</summary>
    public int Groups => Sizes.Length;

    /// <summary>Number of features.</summary>
    public int Features => Means.GetLength(1);

    /// <summary>Samples in each group.</summary>
    public int[] Sizes { get; }

    /// <summary>Mean of each feature within each group, in the units the data arrived in.</summary>
    public double[,] Means { get; }

    /// <summary>Sample standard deviation of each feature within each group; zero for a group of one.</summary>
    public double[,] Spread { get; }

    /// <summary>Mean of each feature over every sample outside each group.</summary>
    public double[,] RestMeans { get; }

    /// <summary>
    /// Cohen's d: the group's mean less the rest's, in pooled standard deviations.
    /// Positive is higher than the rest. Around 0.2 is conventionally small, 0.5
    /// medium and 0.8 large — but read it beside <see cref="Separation"/>, since a
    /// few extreme values can inflate a mean gap that most members do not share.
    /// </summary>
    public double[,] Effect { get; }

    /// <summary>
    /// Cliff's delta, −1 to 1: the probability a member exceeds a non-member, less
    /// the probability it falls below one. 1 means every member is above every
    /// non-member, −1 every one below, 0 the two sides overlap completely.
    /// </summary>
    public double[,] Separation { get; }

    /// <summary>
    /// Per group, feature indices from most to least separating — by the size of
    /// <see cref="Separation"/>, then of <see cref="Effect"/>, then by index.
    /// </summary>
    public int[][] Ranking { get; }

    /// <summary>
    /// A conventional word for the size of a separation, after Romano and others
    /// (2006): below 0.147 negligible, below 0.33 small, below 0.474 medium, beyond
    /// that large.
    /// <para>
    /// Naming only. Nothing here filters or ranks by these words — the ranking is
    /// by the number itself, and a "negligible" feature still appears in it. They
    /// exist so a sentence can say how strongly without a reader having to know
    /// what a Cliff's delta of 0.4 feels like.
    /// </para>
    /// </summary>
    public static string Magnitude(double separation)
    {
        double size = Math.Abs(separation);
        return size < 0.147 ? "negligible"
            : size < 0.33 ? "small"
            : size < 0.474 ? "medium"
            : "large";
    }

    /// <summary>
    /// The strongest separating features of every group, one block per group, in
    /// plain lines — the part of a report about the groups rather than about what
    /// the samples are, for a caller to head with its own lines.
    /// </summary>
    /// <param name="featureNames">A name per feature; null or short falls back to "Feature j".</param>
    /// <param name="top">Features listed per group, clamped to how many there are.</param>
    public string Summary(IReadOnlyList<string>? featureNames = null, int top = 3)
    {
        if (top < 1)
            throw new ArgumentOutOfRangeException(nameof(top), top, "List at least one feature per group.");

        var invariant = CultureInfo.InvariantCulture;
        int shown = Math.Min(top, Features);
        var names = Enumerable.Range(0, Features).Select(j => FeatureName(featureNames, j)).ToArray();
        int width = names.Max(name => name.Length);

        var text = new StringBuilder();
        for (int g = 0; g < Groups; g++)
        {
            if (Sizes[g] == 0)
                continue;

            if (text.Length > 0)
                text.AppendLine();

            text.AppendLine($"Group {g} — {Sizes[g]} of {SampleCount} samples");

            foreach (int j in Ranking[g].Take(shown))
            {
                string direction = Separation[g, j] > 0.0 ? "higher" : Separation[g, j] < 0.0 ? "lower " : "same  ";
                text.AppendLine(
                    $"  {names[j].PadRight(width)}  {direction}  "
                    + $"sep {Separation[g, j].ToString("+0.00;-0.00;0.00", invariant)}  "
                    + $"d {Effect[g, j].ToString("+0.00;-0.00;0.00", invariant),6}  "
                    + $"{Magnitude(Separation[g, j]),-10}  "
                    + $"mean {Means[g, j].ToString("G4", invariant)} against {RestMeans[g, j].ToString("G4", invariant)}");
            }
        }

        return text.ToString().TrimEnd();
    }

    /// <summary>The name for a feature, falling back to its index.</summary>
    public static string FeatureName(IReadOnlyList<string>? featureNames, int feature)
        => featureNames is not null && feature < featureNames.Count && !string.IsNullOrWhiteSpace(featureNames[feature])
            ? featureNames[feature]
            : $"Feature {feature}";
}
