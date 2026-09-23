using OtterLogic.Core;
using OtterLogic.Unsupervised.Clustering;
using Xunit;
using Xunit.Abstractions;

namespace OtterLogic.Unsupervised.Tests;

/// <summary>
/// The promise every method on a wire keeps, checked on each of them the same way.
/// <para>
/// The algorithms underneath have their own parity tests; nothing here re-checks
/// arithmetic. What is checked is the contract the data component relies on
/// without knowing which method it was handed: planted clusters come back,
/// numbered largest first, with a confidence only where one is honest, and the
/// same samples give the same answer twice.
/// </para>
/// </summary>
public sealed class ClusteringMethodTests
{
    private readonly ITestOutputHelper _output;

    public ClusteringMethodTests(ITestOutputHelper output) => _output = output;

    private static readonly (double[,] X, int[] Truth) Planted = Synthetic.Blobs();

    /// <summary>
    /// Every concrete method, told three clusters where it needs telling. HDBSCAN
    /// and Auto find the count themselves, which is what the planted count checks.
    /// </summary>
    public static IEnumerable<object[]> EveryMethod()
    {
        yield return new object[] { new KMeansMethod { Clusters = 3 } };
        yield return new object[] { new GaussianMixtureMethod { Clusters = 3 } };
        yield return new object[] { new HdbscanMethod() };
        yield return new object[] { new SpectralMethod { Clusters = 3 } };
        yield return new object[] { new HierarchicalMethod { Clusters = 3 } };
        yield return new object[] { new AutoMethod() };
    }

    [Theory]
    [MemberData(nameof(EveryMethod))]
    public void RecoversThePlantedClusters(ClusteringMethod method)
    {
        var (x, truth) = Planted;
        method.Validate(x.GetLength(0));

        var outcome = method.Fit(x);

        // The library's index reads an unplaced sample as a group of its own, so
        // HDBSCAN is not rewarded for leaving stragglers out.
        double agreement = ClusterAgreement.AdjustedRand(truth, outcome.Labels);
        _output.WriteLine($"{method.Describe()}: {outcome.ClusterCount} clusters, ARI {agreement:F3}, "
                          + $"{outcome.UnplacedCount} unplaced");

        Assert.Equal(3, outcome.ClusterCount);
        Assert.True(agreement > 0.95, $"{method.Name} recovered the planted clusters at ARI {agreement:F3}");
        Assert.Equal(x.GetLength(0), outcome.SampleCount);
    }

    [Theory]
    [MemberData(nameof(EveryMethod))]
    public void NumbersClustersLargestFirstFromZero(ClusteringMethod method)
    {
        var outcome = method.Fit(Planted.X);
        var sizes = outcome.Members().Select(m => m.Length).ToArray();

        Assert.Equal(outcome.ClusterCount, sizes.Length);
        Assert.All(outcome.Labels, l => Assert.InRange(l, -1, outcome.ClusterCount - 1));
        Assert.Contains(0, outcome.Labels);
        for (int c = 1; c < sizes.Length; c++)
            Assert.True(sizes[c] <= sizes[c - 1], $"cluster {c} ({sizes[c]}) is larger than cluster {c - 1} ({sizes[c - 1]})");

        // The planted sizes are 50, 40, 30: a method that recovered them must
        // number the fifty first, whatever its internals called it.
        Assert.Equal(new[] { 50, 40, 30 }, sizes.Take(3).ToArray());
    }

    /// <summary>
    /// A cut tree says which side of a merge a sample fell and nothing about how
    /// nearly it fell the other way, so Hierarchical is the one method with no
    /// number to give. Every other method has one per sample, in [0, 1].
    /// </summary>
    [Theory]
    [MemberData(nameof(EveryMethod))]
    public void ConfidenceIsNullOnlyWhereNoHonestNumberExists(ClusteringMethod method)
    {
        var outcome = method.Fit(Planted.X);

        if (method is HierarchicalMethod)
        {
            Assert.Null(outcome.Confidence);
            return;
        }

        Assert.NotNull(outcome.Confidence);
        Assert.Equal(outcome.SampleCount, outcome.Confidence!.Length);
        Assert.All(outcome.Confidence, c => Assert.InRange(c, 0.0, 1.0));
    }

    /// <summary>
    /// Grasshopper re-solves on any upstream change; a method that returned a
    /// different labelling from identical samples would be unusable on a wire.
    /// </summary>
    [Theory]
    [MemberData(nameof(EveryMethod))]
    public void FittingTheSameSamplesTwiceGivesTheSameLabels(ClusteringMethod method)
    {
        var first = method.Fit(Planted.X);
        var second = method.Fit(Planted.X);

        Assert.Equal(first.Labels, second.Labels);
        Assert.Equal(first.ClusterCount, second.ClusterCount);
        Assert.Equal(first.Method, second.Method);
        if (first.Confidence is not null)
            Numeric.Close(first.Confidence, second.Confidence!, 0.0, "confidence");
    }

    [Theory]
    [MemberData(nameof(EveryMethod))]
    public void DescribeNamesTheMethod(ClusteringMethod method)
    {
        Assert.Contains(method.Name, method.Describe());
        Assert.Equal(method.Describe(), method.ToString());
    }

    [Theory]
    [MemberData(nameof(EveryMethod))]
    public void NotesCarryALevel(ClusteringMethod method)
    {
        var outcome = method.Fit(Planted.X);

        Assert.All(outcome.Notes, note =>
        {
            Assert.True(Enum.IsDefined(note.Level), $"note level {note.Level} is not a NoteLevel");
            Assert.False(string.IsNullOrWhiteSpace(note.Text));
        });
    }

    /// <summary>
    /// A method is a value: the same settings are the same method, so a component
    /// can tell whether its input changed without comparing field by field.
    /// </summary>
    [Fact]
    public void MethodsWithTheSameSettingsAreEqual()
    {
        Assert.Equal(new KMeansMethod { Clusters = 3 }, new KMeansMethod { Clusters = 3 });
        Assert.Equal(new GaussianMixtureMethod { Covariance = CovarianceType.Full }, new GaussianMixtureMethod { Covariance = CovarianceType.Full });
        Assert.Equal(new HdbscanMethod { MinimumClusterSize = 8 }, new HdbscanMethod { MinimumClusterSize = 8 });
        Assert.Equal(new SpectralMethod { Neighbours = 6 }, new SpectralMethod { Neighbours = 6 });
        Assert.Equal(new HierarchicalMethod { Linkage = Linkage.Average }, new HierarchicalMethod { Linkage = Linkage.Average });
        Assert.Equal(new AutoMethod { MaximumClusters = 6 }, new AutoMethod { MaximumClusters = 6 });

        Assert.NotEqual(new KMeansMethod { Clusters = 3 }, new KMeansMethod { Clusters = 4 });
        Assert.NotEqual<ClusteringMethod>(new KMeansMethod { Clusters = 3 }, new HierarchicalMethod { Clusters = 3 });

        var copy = new KMeansMethod { Clusters = 3 } with { Seed = 7 };
        Assert.Equal(3, copy.Clusters);
        Assert.NotEqual(new KMeansMethod { Clusters = 3 }, copy);
    }

    [Fact]
    public void ValidateRefusesZeroClusters()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new KMeansMethod { Clusters = 0 }.Validate(100));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GaussianMixtureMethod { Clusters = 0 }.Validate(100));
        Assert.Throws<ArgumentOutOfRangeException>(() => new HierarchicalMethod { Clusters = 0 }.Validate(100));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SpectralMethod { Clusters = 1 }.Validate(100));
    }

    [Fact]
    public void ValidateRefusesMoreClustersThanSamples()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new KMeansMethod { Clusters = 11 }.Validate(10));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GaussianMixtureMethod { Clusters = 11 }.Validate(10));
        Assert.Throws<ArgumentOutOfRangeException>(() => new HierarchicalMethod { Clusters = 11 }.Validate(10));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SpectralMethod { Clusters = 11 }.Validate(10));
        Assert.Throws<ArgumentOutOfRangeException>(() => new AutoMethod { MinimumClusters = 11 }.Validate(10));
    }

    /// <summary>A cluster of one is every sample on its own, which is no clustering.</summary>
    [Fact]
    public void ValidateRefusesAnHdbscanClusterSizeOfOne()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new HdbscanMethod { MinimumClusterSize = 1 }.Validate(100));
        Assert.Throws<ArgumentOutOfRangeException>(() => new HdbscanMethod { MinimumClusterSize = 101 }.Validate(100));
    }

    /// <summary>
    /// Choosing between methods needs something to compare on. Three samples is
    /// not it, and the complaint says to wire a method instead.
    /// </summary>
    [Fact]
    public void ValidateRefusesAutoOnThreeSamples()
    {
        var complaint = Assert.Throws<ArgumentException>(() => new AutoMethod().Validate(3));
        Assert.Contains("Wire a method", complaint.Message);

        new AutoMethod().Validate(AutoMethod.FewestSamples);
    }

    [Fact]
    public void ValidateAcceptsSensibleSettings()
    {
        int n = Planted.X.GetLength(0);
        foreach (var row in EveryMethod())
            ((ClusteringMethod)row[0]).Validate(n);
    }

    /// <summary>
    /// Auto is the selector made into a method: the outcome is named for the
    /// method it chose, says why, and carries the score table for the report.
    /// </summary>
    [Fact]
    public void AutoNamesTheChosenMethodAndSaysWhy()
    {
        var outcome = new AutoMethod().Fit(Planted.X);

        _output.WriteLine(outcome.Method + " — " + outcome.Rationale);
        foreach (var line in outcome.Details)
            _output.WriteLine(line);

        Assert.Contains(outcome.Method, new[] { "K-Means", "Gaussian Mixture", "HDBSCAN" });
        Assert.False(string.IsNullOrWhiteSpace(outcome.Rationale));

        Assert.Contains("How each method did:", outcome.Details);
        Assert.Contains(outcome.Details, l => l.Contains("Silhouette") && l.Contains("Davies-Bouldin"));
        Assert.Contains(outcome.Details, l => l.Contains("K-Means"));
        Assert.Contains(outcome.Details, l => l.Contains("Gaussian Mixture"));
        Assert.Contains(outcome.Details, l => l.Contains("HDBSCAN"));
        Assert.Contains(outcome.Details, l => l.StartsWith("> ") && l.Contains(outcome.Method));
    }

    /// <summary>A method asked for directly has no rationale: nothing was chosen.</summary>
    [Fact]
    public void ADirectMethodHasNoRationale()
    {
        Assert.Null(new KMeansMethod { Clusters = 3 }.Fit(Planted.X).Rationale);
        Assert.Null(new HdbscanMethod().Fit(Planted.X).Rationale);
    }

    /// <summary>
    /// The derived minimum cluster size is a decision made on the user's behalf,
    /// and one they can override — so it is said, and it stops being said once
    /// they have.
    /// </summary>
    [Fact]
    public void HdbscanRemarksOnADerivedMinimumSizeOnly()
    {
        var derived = new HdbscanMethod().Fit(Planted.X);
        var given = new HdbscanMethod { MinimumClusterSize = 5 }.Fit(Planted.X);

        Assert.Contains(derived.Notes, n => n.Level == NoteLevel.Remark && n.Text.Contains("derived from"));
        Assert.DoesNotContain(given.Notes, n => n.Text.Contains("derived from"));
    }

    /// <summary>
    /// A far outlier is exactly what HDBSCAN exists to leave out. It comes back as
    /// -1, and the placed clusters are still numbered from zero without a gap.
    /// </summary>
    [Fact]
    public void HdbscanLeavesAnOutlierUnplaced()
    {
        var x = WithOutlier(Planted.X);
        var outcome = new HdbscanMethod().Fit(x);

        int last = x.GetLength(0) - 1;
        Assert.Equal(-1, outcome.Labels[last]);
        Assert.Contains(last, outcome.Unplaced());
        Assert.Equal(outcome.Unplaced().Length, outcome.UnplacedCount);
        Assert.Equal(3, outcome.ClusterCount);
        Assert.Equal(0.0, outcome.Confidence![last]);
    }

    /// <summary>The outcome refuses a labelling that contradicts its own count.</summary>
    [Fact]
    public void OutcomeRefusesALabelBeyondTheCount()
    {
        Assert.Throws<ArgumentException>(() => new ClusteringOutcome("Test", new[] { 0, 1, 2 }, 2, null));
        Assert.Throws<ArgumentException>(() => new ClusteringOutcome("Test", new[] { 0, 1 }, 2, new[] { 1.0 }));
        Assert.Throws<ArgumentException>(() => new ClusteringOutcome(" ", new[] { 0, 1 }, 2, null));
    }

    /// <summary>The planted blobs with one sample far from all of them appended.</summary>
    internal static double[,] WithOutlier(double[,] x, double distance = 40.0)
    {
        int n = x.GetLength(0);
        int d = x.GetLength(1);
        var result = new double[n + 1, d];
        Array.Copy(x, result, x.Length);
        for (int j = 0; j < d; j++)
            result[n, j] = distance;
        return result;
    }
}
