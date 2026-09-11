using OtterLogic.Unsupervised.Clustering;
using Xunit;

namespace OtterLogic.Unsupervised.Tests;

/// <summary>
/// What every result does with its labels: the numbering, the buckets, the
/// inverse, and the means — one convention for unplaced samples throughout.
/// </summary>
public sealed class ClusterLabelsTests
{
    [Fact]
    public void CanonicalNumbersLargestFirstAndTiesByFirstSample()
    {
        var labels = new[] { 5, 9, 9, -3, 7, 7, 9, 5 };

        var canonical = ClusterLabels.Canonical(labels, out var mapping);

        // 9 has three samples; 5 and 7 have two each, and 5 appears first.
        Assert.Equal(new[] { 1, 0, 0, -1, 2, 2, 0, 1 }, canonical);
        Assert.Equal(0, mapping[9]);
        Assert.Equal(1, mapping[5]);
        Assert.Equal(2, mapping[7]);
    }

    [Fact]
    public void MembersKeepsPositionAsTheLabelAndLeavesUnplacedOut()
    {
        var labels = new[] { 2, -1, 0, 2, -1, 0 };

        var members = ClusterLabels.Members(labels, 4);

        Assert.Equal(4, members.Length);
        Assert.Equal(new[] { 2, 5 }, members[0]);
        Assert.Empty(members[1]);
        Assert.Equal(new[] { 0, 3 }, members[2]);
        Assert.Empty(members[3]);
        Assert.Equal(new[] { 1, 4 }, ClusterLabels.Unplaced(labels));
        Assert.Equal(3, ClusterLabels.Members(labels).Length);
    }

    [Fact]
    public void MembersRefusesALabelBeyondTheCount()
        => Assert.Throws<ArgumentOutOfRangeException>(() => ClusterLabels.Members(new[] { 0, 3 }, 3));

    [Fact]
    public void FromMembersInvertsMembers()
    {
        var labels = new[] { 1, -1, 0, 1, 2, -1, 0 };

        var members = ClusterLabels.Members(labels, 3);

        Assert.Equal(labels, ClusterLabels.FromMembers(members, labels.Length));
    }

    [Fact]
    public void FromMembersRefusesASampleInTwoClusters()
    {
        var members = new[] { new[] { 0, 1 }, new[] { 1, 2 } };
        Assert.Throws<ArgumentException>(() => ClusterLabels.FromMembers(members, 3));
    }

    [Fact]
    public void MeansSkipUnplacedRowsAndLeaveAnEmptyClusterAtZero()
    {
        var x = new double[,] { { 1.0, 10.0 }, { 3.0, 30.0 }, { 100.0, 100.0 }, { 5.0, 50.0 } };
        var labels = new[] { 0, 0, -1, 2 };

        var means = ClusterLabels.Means(x, labels, 3);

        Assert.Equal(2.0, means[0, 0]);
        Assert.Equal(20.0, means[0, 1]);
        Assert.Equal(0.0, means[1, 0]);
        Assert.Equal(5.0, means[2, 0]);
    }
}
