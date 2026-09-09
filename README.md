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

Three clustering methods, and a fourth thing that chooses between them.

| | Told how many? | Every sample placed? | Cluster shape |
|---|---|---|---|
| `KMeans` | yes | yes | round, similar size |
| `GaussianMixture` | yes | yes, softly | any ellipsoid, may overlap |
| `Hdbscan` | **no** | **no** | any shape, density-defined |

They are not three implementations of one idea. Each assumes something different
about what a cluster is, and which assumption holds is a property of the data —
which is the whole reason the fourth thing exists.

**`ClusterSelector`** fits all three, scores them, and picks the one the evidence
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

`KMeans` is public in its own right but is also EM's seeder — there is one copy,
and the mixture calls it. A second would be a second thing to keep in step with
the scikit-learn parity fixtures.

## What is deliberately not here

**Any opinion about what the data means.** No log transform chosen for a
particular discipline, no "three components because demand is correlated", no
calling a cluster a "behaviour family". Those are claims about a domain, and they
live in the toolkit that has one — see
[StructuralDesign](https://github.com/Otter-Logic/StructuralDesign) for the
worked example.

The line: if changing it would require knowing what a bending moment is, it does
not belong here.

## Layout

```
src/OtterLogic.Unsupervised/
  Clustering/            k-means, Gaussian mixture, HDBSCAN, quality measures
  Clustering/Selection/  fitting all three and choosing between them
python/                  development only - never ships, never installed by a user
  fixtures/              scikit-learn reference fixtures for the C# tests
tests/                   xunit; runs anywhere, no Rhino needed
```

Nothing here touches a Rhino or Grasshopper API. The components live in
[Rhino3D](https://github.com/Otter-Logic/Rhino3D), under the **Machine Learning**
section.

## Dependencies

None, deliberately. An EM loop with log-sum-exp and a mutual-reachability MST
over a handful of columns; MathNet would be carried for arithmetic that is
already written here. `Microsoft.ML.OnnxRuntime` arrives in MachineLearning when
inference does, and does not belong up here.
