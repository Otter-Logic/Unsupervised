using OtterLogic.Core;
using OtterLogic.Unsupervised.Clustering;
using Xunit;
using Xunit.Abstractions;

namespace OtterLogic.Unsupervised.Tests;

/// <summary>
/// What happens around a method: the preparation before it and the readings
/// after it. The method itself is checked in <see cref="ClusteringMethodTests"/>;
/// here the question is whether the one call a data component makes prepares the
/// samples honestly and reads the answer back in units a person can act on.
/// </summary>
public sealed class ClusterRunTests
{
    private readonly ITestOutputHelper _output;

    public ClusterRunTests(ITestOutputHelper output) => _output = output;

    private static readonly (double[,] X, int[] Truth) Planted = Synthetic.Blobs();

    private static readonly KMeansMethod ThreeMeans = new() { Clusters = 3 };

    /// <summary>
    /// Two clusters told apart by a column between zero and one, beside a column
    /// of noise ten thousand times larger that says nothing. Every method measures
    /// distances, so the raw fit sees only the noise; standardised, the structure
    /// is the only thing left to see.
    /// </summary>
    private static (double[,] X, int[] Truth) WildlyDifferentScales()
    {
        var rng = new Random(5);
        int n = 80;
        var x = new double[n, 2];
        var truth = new int[n];
        for (int i = 0; i < n; i++)
        {
            truth[i] = i < 45 ? 0 : 1;
            x[i, 0] = truth[i] + 0.05 * (rng.NextDouble() - 0.5);
            x[i, 1] = 10_000.0 * (rng.NextDouble() - 0.5);
        }

        return (x, truth);
    }

    [Fact]
    public void StandardisationIsWhatLetsSmallColumnsCount()
    {
        var (x, truth) = WildlyDifferentScales();
        var method = new KMeansMethod { Clusters = 2 };

        var standardised = ClusterRun.Fit(x, method, new ClusterRunOptions { Standardise = true });
        var raw = ClusterRun.Fit(x, method, new ClusterRunOptions { Standardise = false });

        double withScaling = ClusterAgreement.AdjustedRand(truth, standardised.Labels);
        double without = ClusterAgreement.AdjustedRand(truth, raw.Labels);
        _output.WriteLine($"ARI standardised {withScaling:F3}, raw {without:F3}");

        Assert.True(withScaling > 0.95, $"standardised fit reached only ARI {withScaling:F3}");
        Assert.True(without < 0.3, $"the raw fit should have followed the noise, but reached ARI {without:F3}");
        Assert.True(standardised.Standardised);
        Assert.False(raw.Standardised);
        Assert.Contains(standardised.Report(), l => l.Contains("common scale"));
        Assert.Contains(raw.Report(), l => l.Contains("as they arrived"));
    }

    /// <summary>
    /// A constant column has nothing to cluster on and would give a standardiser a
    /// zero to divide by. It is left out of the fit and the user is told — but the
    /// centres still have a value for it, because the column is still part of what
    /// the samples are.
    /// </summary>
    [Fact]
    public void AConstantColumnIsRemarkedOnAndKeptInTheCentres()
    {
        var (x, _) = Planted;
        int n = x.GetLength(0);
        var withConstant = new double[n, 3];
        for (int i = 0; i < n; i++)
        {
            withConstant[i, 0] = x[i, 0];
            withConstant[i, 1] = 7.0;
            withConstant[i, 2] = x[i, 1];
        }

        var result = ClusterRun.Fit(withConstant, ThreeMeans);

        Assert.Equal(new[] { 0, 2 }, result.KeptColumns);
        Assert.Contains(result.Notes, n => n.Level == NoteLevel.Remark && n.Text.Contains("Column(s) 1"));
        Assert.Equal(3, result.Centres.GetLength(1));
        for (int c = 0; c < result.ClusterCount; c++)
            Assert.Equal(7.0, result.Centres[c, 1], 12);

        Assert.Equal(1.0, ClusterAgreement.AdjustedRand(Planted.Truth, result.Labels), 6);
    }

    [Fact]
    public void EveryColumnConstantIsRefusedAsAnArgument()
    {
        var flat = new double[10, 2];
        for (int i = 0; i < 10; i++)
        {
            flat[i, 0] = 3.0;
            flat[i, 1] = -1.0;
        }

        var complaint = Assert.Throws<ArgumentException>(() => ClusterRun.Fit(flat, ThreeMeans));
        Assert.Contains("nothing to cluster on", complaint.Message);
    }

    /// <summary>
    /// The fit was made in standardised space; the centres are not. A centre a
    /// person can read is the mean of the raw rows in the cluster, nothing else.
    /// </summary>
    [Fact]
    public void CentresAreTheRawMeansOfEachCluster()
    {
        var (x, _) = Planted;
        var result = ClusterRun.Fit(x, ThreeMeans);

        int d = x.GetLength(1);
        var members = result.Members();
        Assert.Equal(result.ClusterCount, result.Centres.GetLength(0));
        Assert.Equal(d, result.Centres.GetLength(1));

        for (int c = 0; c < result.ClusterCount; c++)
        {
            for (int j = 0; j < d; j++)
            {
                double mean = members[c].Average(i => x[i, j]);
                Numeric.Close(mean, result.Centres[c, j], 1e-12, $"centre[{c},{j}]");
            }
        }

        // Input units: the planted centres are 2 to 8 apart, which a standardised
        // centre could never be.
        Assert.Contains(Enumerable.Range(0, result.ClusterCount), c => result.Centres[c, 0] > 3.0);
    }

    [Fact]
    public void SeparationScoresAreNaNWithOneClusterAndFiniteOtherwise()
    {
        var one = ClusterRun.Fit(Planted.X, new KMeansMethod { Clusters = 1 });
        var three = ClusterRun.Fit(Planted.X, ThreeMeans);

        Assert.Equal(1, one.ClusterCount);
        Assert.True(double.IsNaN(one.Silhouette));
        Assert.True(double.IsNaN(one.DaviesBouldin));
        Assert.Null(one.Signature);
        Assert.DoesNotContain(one.Report(), l => l.StartsWith("Silhouette"));

        Assert.True(double.IsFinite(three.Silhouette));
        Assert.True(double.IsFinite(three.DaviesBouldin));
        Assert.True(three.Silhouette > 0.5, $"planted blobs should be well separated, silhouette {three.Silhouette:F3}");
        Assert.Contains(three.Report(), l => l.StartsWith("Silhouette") && l.Contains("well separated"));
    }

    [Fact]
    public void AMapHasOneRowPerSampleAndTheDimensionsAskedFor()
    {
        var none = ClusterRun.Fit(Planted.X, ThreeMeans);
        var flat = ClusterRun.Fit(Planted.X, ThreeMeans, new ClusterRunOptions { MapDimensions = 2 });

        Assert.Null(none.Map);
        Assert.NotNull(flat.Map);
        Assert.Equal(Planted.X.GetLength(0), flat.Map!.GetLength(0));
        Assert.Equal(2, flat.Map.GetLength(1));

        // The map is drawn from the same fit, so the labels do not change with it.
        Assert.Equal(none.Labels, flat.Labels);
    }

    [Fact]
    public void AMapIsForLookingAtSoFourDimensionsIsRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => ClusterRun.Fit(Planted.X, ThreeMeans, new ClusterRunOptions { MapDimensions = 4 }));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => ClusterRun.Fit(Planted.X, ThreeMeans, new ClusterRunOptions { MapDimensions = 1 }));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => ClusterRun.Fit(Planted.X, ThreeMeans, new ClusterRunOptions { TopFeatures = 0 }));
    }

    [Fact]
    public void ReportOpensWithTheMethodAndTheCount()
    {
        var result = ClusterRun.Fit(Planted.X, ThreeMeans);
        var report = result.Report(new[] { "Width", "Height" });
        foreach (var line in report)
            _output.WriteLine(line);

        Assert.StartsWith("K-Means: 3 clusters from 120 samples", report[0]);
        Assert.Equal("Sizes: 50, 40, 30.", report[1]);
        Assert.DoesNotContain(report, l => l.Contains("chosen because"));
    }

    [Fact]
    public void ReportCarriesTheRationaleForAnAutomaticChoice()
    {
        var result = ClusterRun.Fit(Planted.X, new AutoMethod());
        var report = result.Report();
        foreach (var line in report)
            _output.WriteLine(line);

        Assert.StartsWith(result.Method + ": 3 clusters", report[0]);
        Assert.Contains("chosen because", report[0]);
        Assert.Contains(result.Outcome.Rationale!, report[0]);
        Assert.Contains("How each method did:", report);
    }

    /// <summary>
    /// With two or more clusters there is something to set apart, and the report
    /// says what, feature by feature and by the names the caller gave them.
    /// </summary>
    [Fact]
    public void ReportExplainsEachClusterWhenThereAreTwoOrMore()
    {
        var result = ClusterRun.Fit(Planted.X, ThreeMeans, new ClusterRunOptions { TopFeatures = 1 });
        var report = result.Report(new[] { "Width", "Height" });

        Assert.NotNull(result.Signature);
        Assert.Contains("What sets each cluster apart:", report);
        Assert.Contains(report, l => l.StartsWith("Group 0 — 50 of 120 samples"));
        Assert.Contains(report, l => l.StartsWith("Group 2 — 30 of 120 samples"));
        Assert.Contains(report, l => l.Contains("Width") || l.Contains("Height"));

        // One feature per cluster was asked for: three clusters, three feature lines.
        Assert.Equal(3, report.Count(l => l.TrimStart().StartsWith("Width") || l.TrimStart().StartsWith("Height")));

        var silent = ClusterRun.Fit(Planted.X, ThreeMeans, new ClusterRunOptions { Explain = false });
        Assert.Null(silent.Signature);
        Assert.DoesNotContain("What sets each cluster apart:", silent.Report());
    }

    /// <summary>
    /// An outlier HDBSCAN leaves out is the interesting sample, so the run says it
    /// is there rather than letting a -1 pass silently into a colour lookup.
    /// </summary>
    [Fact]
    public void AnUnplacedSampleIsRemarkedOnAndListed()
    {
        var x = ClusteringMethodTests.WithOutlier(Planted.X);
        var result = ClusterRun.Fit(x, new HdbscanMethod());

        int last = x.GetLength(0) - 1;
        Assert.Equal(-1, result.Labels[last]);
        Assert.Contains(last, result.Unplaced());
        Assert.Contains(result.Notes, n => n.Level == NoteLevel.Remark && n.Text.Contains("in no cluster"));
        Assert.Contains(result.Report(), l => l.StartsWith("Sizes:") && l.Contains("unplaced"));

        // Unplaced samples have no cluster to pull a centre toward.
        Assert.All(Enumerable.Range(0, result.ClusterCount), c => Assert.True(result.Centres[c, 0] < 20.0));
    }

    [Fact]
    public void FewerThanTwoSamplesIsRefused()
    {
        Assert.Throws<ArgumentException>(() => ClusterRun.Fit(new double[1, 2], ThreeMeans));
        Assert.Throws<ArgumentException>(() => ClusterRun.Fit(new double[0, 2], ThreeMeans));
        Assert.Throws<ArgumentException>(() => ClusterRun.Fit(new double[5, 0], ThreeMeans));
    }

    /// <summary>
    /// A method's own complaint reaches the caller before any fitting: asking for
    /// more clusters than samples is refused where the method says so.
    /// </summary>
    [Fact]
    public void TheMethodsValidationIsRunFirst()
    {
        var few = new double[4, 2];
        for (int i = 0; i < 4; i++)
        {
            few[i, 0] = i;
            few[i, 1] = i * i;
        }

        Assert.Throws<ArgumentOutOfRangeException>(() => ClusterRun.Fit(few, new KMeansMethod { Clusters = 5 }));
    }

    [Fact]
    public void NoMethodMeansAuto()
    {
        var chosen = ClusterRun.Fit(Planted.X);
        var explicitly = ClusterRun.Fit(Planted.X, new AutoMethod());

        Assert.NotNull(chosen.Outcome.Rationale);
        Assert.Equal(explicitly.Method, chosen.Method);
        Assert.Equal(explicitly.Labels, chosen.Labels);
        Assert.Contains("chosen because", chosen.Report()[0]);
    }

    /// <summary>The notes the method raised come through, alongside the run's own.</summary>
    [Fact]
    public void TheMethodsNotesAreCarriedThrough()
    {
        var result = ClusterRun.Fit(Planted.X, new HdbscanMethod());

        foreach (var note in result.Outcome.Notes)
            Assert.Contains(note, result.Notes);
    }
}
