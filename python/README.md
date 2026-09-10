# Python

**No user of the plug-in installs any of this.** Not IronPython, not Rhino 8's
embedded CPython, not a system interpreter. Everything here runs on a developer's
machine and stops there; the clustering that ships is C# end to end.

## Reference fixtures

scikit-learn is here as a *reference implementation to test the C# against*, not
as a dependency. `fixtures/make_fixtures.py` writes JSON into the test project;
the C# tests assert against it.

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python fixtures/make_fixtures.py
```

Commit the JSON it produces. Regenerate it when the fixture data or the
comparison changes — not on every run, or the tests are asserting against
whatever was generated last rather than against a fixed reference.

The fixtures answer two separate questions, and it is worth keeping them
separate:

- **`em_*.json`** pin the initial parameters, so both implementations start from
  exactly the same place. From there they should agree to machine precision, and
  any drift is an arithmetic bug. Nothing about local optima enters into it.
- **`quality.json`** and **`sweep.json`** let each implementation initialise
  itself, ten restarts each, and compare on BIC and on how far the two labellings
  agree. Matching parameter for parameter here would be a coincidence — these are
  two non-convex optimisers on the same surface.
- **`spectral.json`** asks both questions of spectral clustering at once. The
  Laplacian eigenvalues are determined by the graph, so they must match a dense
  eigensolve to rounding; the partition is k-means on the embedding, so it is
  compared with scikit-learn's by ARI.
- **`hierarchical.json`** is exact throughout. A dendrogram has no initialisation
  and no local optimum, so SciPy's `linkage` for all four linkages and
  scikit-learn's connectivity-constrained `ward_tree` must match merge for merge.
  The data is tie-free on purpose — a tie in merge distance is decided by
  implementation detail, and a fixture should not rest on one.

Adding a fixture must leave the existing ones byte-identical. Regenerate, and
check `git status` shows only the new files — that is the proof nothing already
pinned has moved.

PCA appears in this script as *preprocessing*, not as something under test: it is
how the EM fixtures get the whitened input the C# side will also be clustering.
The decomposition fixtures themselves live in
[MachineLearning](https://github.com/Otter-Logic/MachineLearning), next to the
`PrincipalComponents` they check. Both generators build their input from the same
`make_members`, which is why the two files look alike at the top.

## No training here

This repo holds algorithms that are fitted at solve time, on the data in front of
them. A Gaussian mixture computes its own parameters from whatever is on the
wire, every solve — there are no weights to find offline and nothing to ship, so
nothing here ever produces an `.onnx`.

Learned models, when they arrive, get trained in `MachineLearning/python` and
cross into the plug-in as a frozen graph. See
`Rhino3D/docs/machine-learning.md` for that half.
