using OtterLogic.Unsupervised.Clustering;
using Xunit;

namespace OtterLogic.Unsupervised.Tests;

/// <summary>
/// The promise ratio clustering makes — every member carries at least the share
/// of its group's peak in every column — and the edges of it.
/// </summary>
public sealed class RatioClusteringTests
{
    [Theory]
    [InlineData(0.5)]
    [InlineData(0.6)]
    [InlineData(0.8)]
    [InlineData(0.95)]
    public void EveryMemberCarriesTheShareOfItsGroupsPeak(double share)
    {
        var x = Magnitudes(n: 150, d: 3, seed: 5);

        var result = RatioClustering.Fit(x, share);

        var clusters = result.Clusters();
        for (int c = 0; c < result.ClusterCount; c++)
        {
            for (int j = 0; j < x.GetLength(1); j++)
            {
                double peak = clusters[c].Max(i => x[i, j]);
                Assert.Equal(peak, result.Peaks[c, j]);
                foreach (int i in clusters[c])
                    Assert.True(x[i, j] >= share * peak * (1 - 1e-12),
                        $"Sample {i} carries {x[i, j]} of a peak of {peak} in column {j}, under {share:P0}.");
            }

            Assert.True(result.LeastShare[c] >= share * (1 - 1e-12));
        }
    }

    [Fact]
    public void AHigherShareNeverGivesFewerGroups()
    {
        var x = Magnitudes(n: 120, d: 2, seed: 9);

        var counts = new[] { 0.3, 0.5, 0.7, 0.9 }.Select(s => RatioClustering.Fit(x, s).ClusterCount).ToArray();

        for (int t = 1; t < counts.Length; t++)
            Assert.True(counts[t] >= counts[t - 1], $"Counts by share: {string.Join(", ", counts)}.");
    }

    [Fact]
    public void ReadsRatiosNotDifferences()
    {
        // 80 against 100 is the same gap as 8,000 against 10,000.
        var x = new double[,] { { 80.0 }, { 100.0 }, { 8000.0 }, { 10000.0 } };

        var labels = RatioClustering.Fit(x, 0.8).Labels;

        Assert.Equal(labels[0], labels[1]);
        Assert.Equal(labels[2], labels[3]);
        Assert.NotEqual(labels[0], labels[2]);
    }

    [Fact]
    public void ZeroNeverSharesWithSomething()
    {
        var x = new double[,] { { 0.0, 5.0 }, { 0.0, 5.0 }, { 1.0, 5.0 } };

        var labels = RatioClustering.Fit(x, 0.1).Labels;

        Assert.Equal(labels[0], labels[1]);
        Assert.NotEqual(labels[0], labels[2]);
    }

    [Fact]
    public void ShareOfZeroIsOneGroupAndShareOfOneIsIdenticalOnly()
    {
        var x = new double[,] { { 1.0 }, { 2.0 }, { 2.0 }, { 50.0 } };

        Assert.Equal(1, RatioClustering.Fit(x, 0.0).ClusterCount);
        Assert.Equal(new[] { 1, 0, 0, 2 }, RatioClustering.Fit(x, 1.0).Labels);
    }

    [Fact]
    public void NumbersLargestGroupFirst()
    {
        var x = new double[,] { { 100.0 }, { 1.0 }, { 1.0 }, { 1.0 } };

        Assert.Equal(new[] { 1, 0, 0, 0 }, RatioClustering.Fit(x, 0.9).Labels);
    }

    [Fact]
    public void OneSampleIsOneGroup()
    {
        var result = RatioClustering.Fit(new double[,] { { 3.0, 4.0 } }, 0.6);

        Assert.Equal(new[] { 0 }, result.Labels);
        Assert.Equal(1.0, result.LeastShare[0]);
    }

    [Fact]
    public void RefusesNegativeValuesAndAShareOutsideZeroToOne()
    {
        Assert.Throws<ArgumentException>(() => RatioClustering.Fit(new double[,] { { 1.0 }, { -1.0 } }, 0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => RatioClustering.Fit(new double[,] { { 1.0 } }, 1.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => RatioClustering.Fit(new double[,] { { 1.0 } }, double.NaN));
    }

    /// <summary>Log-uniform magnitudes over three decades, the spread real demands have.</summary>
    private static double[,] Magnitudes(int n, int d, int seed)
    {
        var rng = new Random(seed);
        var x = new double[n, d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                x[i, j] = Math.Pow(10.0, 3.0 * rng.NextDouble());

        return x;
    }
}
