using OtterLogic.Unsupervised.Clustering;
using Xunit;

namespace OtterLogic.Unsupervised.Tests;

public class ClusterExemplarsTests
{
    /// <summary>
    /// Points at 0, 1, 2 and 10 have total distances to the others of 13, 11, 11 and
    /// 27. The two at 11 tie, and the tie goes to the lower index — the point at 1.
    /// The far point at 10 is never picked, where a mean would sit pulled toward it.
    /// </summary>
    [Fact]
    public void TheMedoidIsTheMostCentralRealSample()
    {
        var x = new double[,] { { 0 }, { 1 }, { 2 }, { 10 }, { 50 }, { 52 } };
        var labels = new[] { 0, 0, 0, 0, 1, 1 };

        var medoids = ClusterExemplars.Medoids(x, labels);

        Assert.Equal(new[] { 1, 4 }, medoids);
    }

    [Fact]
    public void UnplacedSamplesAreNeverExemplarsAndHaveNoDistance()
    {
        var x = new double[,] { { 0 }, { 0 }, { 99 } };
        var labels = new[] { 0, 0, -1 };

        var medoids = ClusterExemplars.Medoids(x, labels);
        var distance = ClusterExemplars.DistanceFromMedoid(x, labels, medoids);

        Assert.Equal(new[] { 0 }, medoids);
        Assert.Equal(0.0, distance[1]);
        Assert.True(double.IsNaN(distance[2]));
    }

    [Fact]
    public void AnEmptyLabelHasNoExemplar()
    {
        var medoids = ClusterExemplars.Medoids(new double[,] { { 0 }, { 1 } }, new[] { 0, 2 });
        Assert.Equal(new[] { 0, -1, 1 }, medoids);
    }

    [Fact]
    public void IdenticalMembersAreAllZeroFromTheirExemplar()
    {
        var x = new double[,] { { 3, 3 }, { 3, 3 }, { 3, 3 }, { 7, 1 } };
        var labels = new[] { 0, 0, 0, 1 };

        var distance = ClusterExemplars.DistanceFromMedoid(x, labels, ClusterExemplars.Medoids(x, labels));

        Assert.All(distance, d => Assert.Equal(0.0, d));
    }
}
