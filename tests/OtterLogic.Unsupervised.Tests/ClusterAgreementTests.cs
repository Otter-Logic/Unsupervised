using OtterLogic.Unsupervised.Clustering;
using Xunit;

namespace OtterLogic.Unsupervised.Tests;

public class ClusterAgreementTests
{
    [Fact]
    public void RenumberingChangesNothing()
    {
        Assert.Equal(1.0, ClusterAgreement.AdjustedRand(new[] { 0, 0, 1, 1, 2 }, new[] { 2, 2, 0, 0, 1 }), 12);
    }

    /// <summary>scikit-learn's documented example: <c>adjusted_rand_score([0, 0, 1, 2], [0, 0, 1, 1])</c> is 4/7.</summary>
    [Fact]
    public void MatchesScikitLearnsDocumentedExample()
    {
        Assert.Equal(4.0 / 7.0, ClusterAgreement.AdjustedRand(new[] { 0, 0, 1, 2 }, new[] { 0, 0, 1, 1 }), 12);
    }

    [Fact]
    public void OneGroupAgainstAllApartIsChance()
    {
        Assert.Equal(0.0, ClusterAgreement.AdjustedRand(new[] { 0, 0, 0, 0 }, new[] { 0, 1, 2, 3 }), 12);
    }

    /// <summary>
    /// Unplaced samples are groups of one: leaving the same samples out agrees,
    /// and filing them with others disagrees about exactly those pairs.
    /// </summary>
    [Fact]
    public void UnplacedSamplesAreGroupsOfOne()
    {
        Assert.Equal(1.0, ClusterAgreement.AdjustedRand(new[] { -1, -1, 0, 0 }, new[] { 5, 6, 7, 7 }), 12);
        Assert.True(ClusterAgreement.AdjustedRand(new[] { -1, -1, 0, 0 }, new[] { 1, 1, 0, 0 }) < 1.0);
    }

    /// <summary>The same number the tests' own helper computes, on labellings with no unplaced samples.</summary>
    [Fact]
    public void AgreesWithTheTestHelper()
    {
        var rng = new Random(3);
        var a = Enumerable.Range(0, 80).Select(_ => rng.Next(4)).ToArray();
        var b = Enumerable.Range(0, 80).Select(i => i % 5 == 0 ? rng.Next(4) : a[i]).ToArray();

        Assert.Equal(Numeric.AdjustedRandIndex(a, b), ClusterAgreement.AdjustedRand(a, b), 12);
    }
}
