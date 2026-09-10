# Unsupervised

Clustering and other unsupervised learning for the OtterLogic Rhino tools.
Finding structure in data nobody has labelled.

A paradigm repo. It sits above
[MachineLearning](https://github.com/Otter-Logic/MachineLearning), which holds
the preprocessing and decomposition every paradigm shares, and below the domain
toolkits that put it to work.

```
Core  ->  MachineLearning  ->  Unsupervised  ->  StructuralDesign
          (features, PCA,      (this repo:       (what the numbers
           datasets, ONNX)      the methods)      actually mean)
```

It references no sibling paradigm — not Supervised, not Reinforcement, not
DeepLearning — and no toolkit. If a sibling ever needs something here, that
something moves *down* into MachineLearning rather than sideways. Same rule Core
has, one layer up.

## What is here

Six clustering methods, and a seventh thing that chooses between three of them.

| | Told how many? | Every sample placed? | Clusters on | Uses a graph? |
|---|---|---|---|---|
| `KMeans` | yes | yes | closeness — round, similar size | no |
| `GaussianMixture` | yes | yes, softly | closeness — any ellipsoid, may overlap | no |
| `Hdbscan` | **no** | **no** | density — any shape | no |
| `SpectralClustering` | yes | all but isolated samples | **connectivity** — paths of strong edges | optional |
| `HierarchicalClustering` | **no — cut afterwards** | yes | a whole nested tree | optional |
| `MessagePassing` | yes | yes | features smoothed over neighbours | **yes** |

They are not six implementations of one idea. Each assumes something different
about what a cluster is, and which assumption holds is a property of the data.

The three graph methods take a `WeightedGraph` from MachineLearning, which says
which samples are related. Where the edges come from is the caller's business — a
toolkit that knows which of its elements touch builds the graph from that; a
caller with only a point cloud uses `WeightedGraph.NearestNeighbours`. Every
graph method has an overload taking both a graph and features, and that is the
**connectivity plus behaviour** case: `Affinity.Gaussian` weights each edge by
how alike its two ends are, so a strong edge means *related and alike*, and cuts
fall where related samples stop behaving alike.

- **`SpectralClustering`** embeds the graph by its Laplacian's leading
  eigenvectors and runs k-means there. Samples joined by a path of strong edges
  land together whatever shape they made in feature space. Reports the
  eigenvalues and the eigengap, which say whether k was a natural number to ask for.
- **`HierarchicalClustering`** builds the full dendrogram — Ward, complete,
  average or single linkage — and cuts it by count or by distance afterwards. Fine
  cuts are guaranteed to nest inside coarse ones. With a graph, every cluster at
  every level is one connected piece.
- **`MessagePassing`** is the training-free half of a graph neural network:
  `Smooth` averages each sample's features with its neighbourhood's, `Fit`
  clusters the result, and `Refine` passes an *existing* labelling over the graph
  — correcting what its neighbours disagree with, filling in what another method
  left unplaced, and reporting exactly which samples changed. A *learned* graph
  network is a trained model and belongs in DeepLearning; what this produces is
  the labelled data it will be trained on.

**`ClusterSelector`** fits k-means, the mixture and HDBSCAN, scores them, and picks the one the evidence
supports, saying which and why. It takes a matrix that is *already prepared* and
knows nothing about what the columns mean. That is deliberate: a structural
toolkit grouping members by demand and a fabrication toolkit grouping panels by
shape want the same mechanism and different judgement, and this is the half they
share.

**`ClusterPipeline`** is the other shape of the same idea: preprocess, decompose,
fit one named method, map the answer back. For a caller that already knows which
method it wants.

`ClusterQuality` holds silhouette and Davies-Bouldin. Both exclude noise rather
than scoring it as a cluster — noise is not a group and has no centre, and
counting it as one would punish HDBSCAN for the thing it exists to do. Both are
also documented as what they are: compactness measures, useful for comparing
partitions and actively misleading for comparing *algorithms*.

`KMeans` is public in its own right but is also EM's seeder, and the partition
step inside spectral clustering and message passing — there is one copy, and they
all call it. A second would be a second thing to keep in step with the
scikit-learn parity fixtures.

Every method numbers its clusters largest first, ties to the lowest sample index,
so a small change upstream does not permute them and shuffle every colour
downstream. The one exception is `MessagePassing.Refine`, which keeps the numbers
it was given — it adjusts an answer somebody already holds references to.

## Parity with scikit-learn and SciPy

scikit-learn and SciPy are reference implementations to test against, never
dependencies. `python/fixtures/make_fixtures.py` writes the JSON the tests assert
against.

| Method | Checked against | How closely |
|---|---|---|
| Gaussian mixture | `GaussianMixture` from a pinned start | parameters to 1e-8 |
| Pipeline | scikit-learn's own pipeline | BIC within 2%, ARI 1.00 |
| Spectral | `SpectralClustering`, and a dense Laplacian `eigh` | eigenvalues to 1e-8, ARI 1.00 |
| Hierarchical | SciPy `linkage`, all four linkages | every merge, distances to 1e-10 |
| Constrained Ward | scikit-learn `ward_tree` with connectivity | every merge, distances to 1e-10 |

Message passing has no reference to compare with, so it is tested against its
definition written out densely, and against planted structure: on a graph whose
communities the features alone cannot find (k-means ARI 0.04), two hops recover
them at ARI 0.97, and refining a labelling with a quarter of it damaged restores
it completely.

## What is deliberately not here

**Any opinion about what the data means.** No log transform chosen for a
particular discipline, no "three components because demand is correlated", no
calling a cluster a "behaviour family", and no rule for which elements of a model
count as connected. Those are claims about a domain, and they live in the toolkit
that has one — see [StructuralDesign](https://github.com/Otter-Logic/StructuralDesign)
for the worked example. Every preprocessing switch defaults off for the same
reason.

The line: if changing it would require knowing what a bending moment is, it does
not belong here.

**Combined tools.** A pipeline that runs spectral clustering for one purpose,
refines it by message passing and cuts a hierarchy inside each group is a
judgement about a discipline, built *from* these methods. It belongs in the
toolkit whose question it answers.

## Layout

```
src/OtterLogic.Unsupervised/
  Clustering/            k-means, Gaussian mixture, HDBSCAN, spectral,
                         hierarchical, message passing, affinity, quality measures
  Clustering/Selection/  fitting k-means, mixture and HDBSCAN and choosing between them
python/                  development only - never ships, never installed by a user
  fixtures/              scikit-learn reference fixtures for the C# tests
tests/                   xunit; runs anywhere, no Rhino needed
```

Nothing here touches a Rhino or Grasshopper API. The components live in
[Rhino3D](https://github.com/Otter-Logic/Rhino3D), under the **Machine Learning**
section.

## Dependencies

None, deliberately. An EM loop with log-sum-exp, a mutual-reachability MST, a
nearest-neighbour chain and a propagation over sparse edges; MathNet would be
carried for arithmetic that is already written here. The graph type and the
leading-eigenvector solver spectral clustering needs live one layer down in
MachineLearning, because a trained graph network will want the same ones. `Microsoft.ML.OnnxRuntime` arrives in MachineLearning when
inference does, and does not belong up here.
