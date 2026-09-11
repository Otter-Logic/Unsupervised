namespace OtterLogic.Unsupervised.Clustering;

/// <summary>
/// Reductions over a responsibility matrix.
/// <para>
/// Shared because every soft method ends the same way — a mixture, message
/// passing, a refinement — and two copies of an argmax is two places for the
/// tie-breaking to drift apart.
/// </para>
/// </summary>
internal static class Posterior
{
    /// <summary>Index of the largest responsibility in each row, first one winning a tie.</summary>
    internal static int[] ArgMax(double[,] responsibilities)
    {
        int n = responsibilities.GetLength(0);
        int k = responsibilities.GetLength(1);
        var labels = new int[n];

        for (int i = 0; i < n; i++)
        {
            int best = 0;
            for (int c = 1; c < k; c++)
                if (responsibilities[i, c] > responsibilities[i, best])
                    best = c;

            labels[i] = best;
        }

        return labels;
    }

    /// <summary>The largest responsibility in each row — how sure the model is.</summary>
    internal static double[] RowMax(double[,] responsibilities)
    {
        int n = responsibilities.GetLength(0);
        int k = responsibilities.GetLength(1);
        var maxima = new double[n];

        for (int i = 0; i < n; i++)
        {
            double best = responsibilities[i, 0];
            for (int c = 1; c < k; c++)
                if (responsibilities[i, c] > best)
                    best = responsibilities[i, c];

            maxima[i] = best;
        }

        return maxima;
    }

    /// <summary>Largest column per row, first on a tie, or -1 for a row with nothing in it.</summary>
    internal static int[] ArgMaxOrNone(double[,] memberships)
    {
        int n = memberships.GetLength(0);
        int k = memberships.GetLength(1);
        var labels = new int[n];

        for (int i = 0; i < n; i++)
        {
            int best = -1;
            double bestValue = 0.0;

            for (int c = 0; c < k; c++)
            {
                if (memberships[i, c] > bestValue)
                {
                    bestValue = memberships[i, c];
                    best = c;
                }
            }

            labels[i] = best;
        }

        return labels;
    }
}
