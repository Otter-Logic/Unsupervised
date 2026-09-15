using OtterLogic.Unsupervised.Clustering;
using Xunit;

namespace OtterLogic.Unsupervised.Tests;

public class ResolveUnplacedTests
{
    private static readonly double[,] Line = { { 0.0 }, { 10.0 }, { 1.0 }, { 9.0 } };

    [Fact]
    public void LeaveKeepsThemUnplaced()
    {
        var labels = new[] { 0, 1, -1, -1 };
        var resolved = ClusterLabels.ResolveUnplaced(labels, null, UnplacedPolicy.Leave);

        Assert.Equal(labels, resolved);
        Assert.NotSame(labels, resolved);
    }

    [Fact]
    public void OwnGroupNumbersThemAfterEveryExistingGroup()
    {
        Assert.Equal(new[] { 0, 2, 1, 3 },
            ClusterLabels.ResolveUnplaced(new[] { 0, -1, 1, -1 }, null, UnplacedPolicy.OwnGroup));
    }

    [Fact]
    public void NearestFilesThemWithTheClosestGroupMean()
    {
        Assert.Equal(new[] { 0, 1, 0, 1 },
            ClusterLabels.ResolveUnplaced(new[] { 0, 1, -1, -1 }, Line, UnplacedPolicy.Nearest));
    }

    [Fact]
    public void NearestWithNoGroupsGivesEachItsOwn()
    {
        Assert.Equal(new[] { 0, 1, 2, 3 },
            ClusterLabels.ResolveUnplaced(new[] { -1, -1, -1, -1 }, Line, UnplacedPolicy.Nearest));
    }

    [Fact]
    public void NearestNeedsTheData()
    {
        Assert.Throws<ArgumentNullException>(
            () => ClusterLabels.ResolveUnplaced(new[] { 0, -1 }, null, UnplacedPolicy.Nearest));
    }
}
