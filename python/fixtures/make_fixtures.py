"""Generate scikit-learn reference fixtures for the C# clustering tests.

Run this, commit the JSON it writes, and the C# tests assert against it. Nothing
here runs at plug-in runtime — scikit-learn is a reference implementation, not a
dependency.

Two different questions are being asked, and they need different fixtures:

*Exactness.* Given identical starting parameters, does the C# EM compute the
same trajectory as scikit-learn's? Pinning the initialisation removes the local
optimum from the comparison, so any disagreement is arithmetic and is a bug.

*Quality.* Left to find its own starting point, does the C# pipeline reach as
good a fit as scikit-learn's default? This one is not expected to match
parameter for parameter — two non-convex optimisers climbing different hills —
so it is scored on BIC and on how far the two labellings agree.

    python python/fixtures/make_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

OUT = Path(__file__).resolve().parents[2] / "tests" / "OtterLogic.Unsupervised.Tests" / "Fixtures"

DOF_NAMES = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

# Relative demand across the six degrees of freedom for four member families,
# loosely shaped like what a moment frame produces: axial-dominated columns,
# beams carrying shear and major-axis moment, lighter secondary beams working
# about the minor axis, and transfer members with everything at once.
PROFILES = np.array([
    [1.00, 0.10, 0.08, 0.02, 0.35, 0.05],
    [0.15, 0.55, 0.10, 0.05, 0.80, 0.12],
    [0.10, 0.12, 0.50, 0.06, 0.10, 0.75],
    [0.60, 0.45, 0.40, 0.30, 0.55, 0.50],
])
SHARES = np.array([0.35, 0.30, 0.20, 0.15])


def make_members(n: int = 300, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Six unsigned magnitudes per member, log-normally scaled within a family.

    The log-normal scale is the point. Real member demands are heavily
    right-skewed — a few members carry most of the load and a long tail carries
    very little — and that skew is what the log transform in the C# pipeline
    exists to handle. Fixtures drawn from a tidy Gaussian would test nothing.
    """
    rng = np.random.default_rng(seed)

    family = rng.choice(len(PROFILES), size=n, p=SHARES)
    scale = np.exp(rng.normal(loc=3.2, scale=0.9, size=n))
    noise = 1.0 + rng.normal(loc=0.0, scale=0.18, size=(n, 6))

    x = np.abs(PROFILES[family] * scale[:, None] * noise)
    return x, family


def standardise(x: np.ndarray) -> np.ndarray:
    """log1p then z-score with the population standard deviation.

    Population rather than sample, matching scikit-learn's StandardScaler and
    the C# FeaturePipeline.
    """
    logged = np.log1p(x)
    return (logged - logged.mean(axis=0)) / logged.std(axis=0)


def sign_flips(components: np.ndarray) -> np.ndarray:
    """+1 or -1 per component, making the largest-magnitude loading positive."""
    dominant = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(components.shape[0]), dominant])
    signs[signs == 0] = 1.0
    return signs


def initial_parameters(x: np.ndarray, k: int, covariance_type: str) -> dict:
    """A starting point both implementations can reproduce exactly.

    Evenly spaced rows as means, uniform mixing weights, and one shared
    isotropic covariance. Deliberately not k-means: the point of this fixture is
    to take initialisation out of the comparison entirely.
    """
    n, d = x.shape
    indices = [i * n // k for i in range(k)]

    means = x[indices].copy()
    weights = np.full(k, 1.0 / k)
    spread = float(x.var(axis=0).mean())

    covariances = np.array([np.eye(d) * spread for _ in range(k)])

    return {
        "weights": weights,
        "means": means,
        "covariances": covariances,
        "spread": spread,
    }


def em_fixture(x: np.ndarray, k: int, covariance_type: str,
               reg_covar: float = 1e-6, tol: float = 1e-4, max_iter: int = 200) -> dict:
    init = initial_parameters(x, k, covariance_type)
    d = x.shape[1]

    # scikit-learn takes precisions, not covariances. The initial covariance is
    # isotropic so inverting it is exact and introduces no drift of its own.
    if covariance_type == "full":
        precisions_init = np.array([np.eye(d) / init["spread"] for _ in range(k)])
    elif covariance_type == "diag":
        precisions_init = np.full((k, d), 1.0 / init["spread"])
    elif covariance_type == "spherical":
        precisions_init = np.full(k, 1.0 / init["spread"])
    else:
        raise ValueError(covariance_type)

    gm = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        tol=tol,
        max_iter=max_iter,
        n_init=1,
        weights_init=init["weights"],
        means_init=init["means"],
        precisions_init=precisions_init,
        random_state=0,
    )
    gm.fit(x)

    responsibilities = gm.predict_proba(x)

    return {
        "name": f"em_{covariance_type}",
        "covariance_type": covariance_type,
        "components": k,
        "reg_covar": reg_covar,
        "tolerance": tol,
        "max_iterations": max_iter,
        "x": x.tolist(),
        "init": {
            "weights": init["weights"].tolist(),
            "means": init["means"].tolist(),
            "covariances": init["covariances"].tolist(),
        },
        "expected": {
            "weights": gm.weights_.tolist(),
            "means": gm.means_.tolist(),
            "covariances": expand_covariances(gm.covariances_, covariance_type, k, d).tolist(),
            "responsibilities": responsibilities.tolist(),
            # score() is the mean log-likelihood under the final parameters,
            # which is one M-step later than sklearn's own lower_bound_.
            "mean_log_likelihood": float(gm.score(x)),
            "bic": float(gm.bic(x)),
            "aic": float(gm.aic(x)),
            "iterations": int(gm.n_iter_),
            "converged": bool(gm.converged_),
            "parameter_count": int(gm._n_parameters()),
        },
    }


def expand_covariances(covariances: np.ndarray, covariance_type: str, k: int, d: int) -> np.ndarray:
    """Everything as k full matrices, so the C# side never has to branch."""
    if covariance_type == "full":
        return covariances
    if covariance_type == "diag":
        return np.array([np.diag(covariances[c]) for c in range(k)])
    if covariance_type == "spherical":
        return np.array([np.eye(d) * covariances[c] for c in range(k)])
    raise ValueError(covariance_type)


def quality_fixture(raw: np.ndarray, family: np.ndarray, k: int = 4) -> dict:
    """scikit-learn's default pipeline, doing its own initialisation.

    This is the honest comparison: both implementations get the same data and
    the same preprocessing, then each finds its own starting point ten times
    over and keeps its best. Matching parameter for parameter would be a
    coincidence; the question is whether the two reach an equally good fit and
    agree about which member belongs with which.
    """
    prepared = standardise(raw)

    pca = PCA(n_components=0.99, whiten=True, svd_solver="full")
    z = pca.fit_transform(prepared)
    z = z * sign_flips(pca.components_)

    gm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        n_init=10,
        reg_covar=1e-6,
        tol=1e-4,
        max_iter=200,
        random_state=0,
    )
    labels = gm.fit_predict(z)

    return {
        "name": "quality",
        "components": k,
        "raw": raw.tolist(),
        "true_family": family.tolist(),
        "columns": DOF_NAMES,
        "expected": {
            "retained_components": int(pca.n_components_),
            "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
            "labels": labels.tolist(),
            "bic": float(gm.bic(z)),
            "aic": float(gm.aic(z)),
            "mean_log_likelihood": float(gm.score(z)),
            "mean_confidence": float(gm.predict_proba(z).max(axis=1).mean()),
            "iterations": int(gm.n_iter_),
        },
    }


def sweep_fixture(raw: np.ndarray, lo: int = 2, hi: int = 8) -> dict:
    """BIC across a range of k, for the group-count component."""
    prepared = standardise(raw)
    pca = PCA(n_components=0.99, whiten=True, svd_solver="full")
    z = pca.fit_transform(prepared) * sign_flips(pca.components_)

    rows = []
    for k in range(lo, hi + 1):
        gm = GaussianMixture(
            n_components=k, covariance_type="diag", n_init=10,
            reg_covar=1e-6, tol=1e-4, max_iter=200, random_state=0,
        )
        gm.fit(z)
        rows.append({"groups": k, "bic": float(gm.bic(z)), "aic": float(gm.aic(z))})

    return {"name": "sweep", "minimum": lo, "maximum": hi, "raw": raw.tolist(), "expected": rows}


def symmetric_knn(x: np.ndarray, k: int) -> np.ndarray:
    """The k-nearest-neighbour graph, self excluded, symmetrised as 0.5 * (A + A.T).

    What the C# WeightedGraph.NearestNeighbours builds, and what scikit-learn's
    SpectralClustering builds from n_neighbors = k + 1 - its graph counts each
    point as its own first neighbour, then the Laplacian ignores the diagonal.
    """
    from sklearn.neighbors import kneighbors_graph

    a = kneighbors_graph(x, n_neighbors=k, mode="connectivity", include_self=False)
    return (0.5 * (a + a.T)).toarray()


def spectral_fixture(k_neighbours: int = 10, noise: float = 0.05, seed: int = 0) -> dict:
    """Two interleaved crescents: connected, but not compact.

    The case spectral clustering exists for. Each crescent is one path of close
    neighbours from end to end, so a graph method follows it; but its two ends
    are far apart and nearer the other crescent's middle, so any method that
    scores compactness - k-means, a mixture - cuts across both.

    Two questions, as for the mixture. *Exactness*: the Laplacian spectrum of
    the same graph from a dense eigensolve, which the C# eigenvalues must match.
    *Quality*: scikit-learn's own labelling, which the C# partition must agree
    with - not label for label, since numbering is arbitrary, but by ARI.

    The noise level and seed are chosen so the neighbour graph is one connected
    piece. At a little more noise the crescents start to touch and even
    scikit-learn recovers them only partly (ARI 0.81 at noise 0.06, seed 3); at a
    little less, the graph can fall into two components, and then the test would
    only be checking that a disconnection is found - not that a connected graph
    is cut in the right place.
    """
    from scipy.sparse.csgraph import laplacian
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.datasets import make_moons

    x, truth = make_moons(n_samples=300, noise=noise, random_state=seed)
    clusters = 2

    affinity = symmetric_knn(x, k_neighbours)
    lap = laplacian(affinity, normed=True)
    spectrum = np.sort(np.linalg.eigvalsh(lap))

    spectral = SpectralClustering(
        n_clusters=clusters,
        affinity="nearest_neighbors",
        n_neighbors=k_neighbours + 1,
        assign_labels="kmeans",
        random_state=0,
    ).fit(x)

    kmeans = KMeans(n_clusters=clusters, n_init=10, random_state=0).fit(x)

    return {
        "name": "spectral",
        "clusters": clusters,
        "neighbours": k_neighbours,
        "x": x.tolist(),
        "true_labels": truth.tolist(),
        "expected": {
            "laplacian_eigenvalues": spectrum[: clusters + 1].tolist(),
            "labels": spectral.labels_.tolist(),
            "kmeans_labels": kmeans.labels_.tolist(),
        },
    }


def hierarchical_fixture(seed: int = 6, k_neighbours: int = 8) -> dict:
    """SciPy's linkage for every linkage, and scikit-learn's constrained Ward.

    Both are exact checks. Hierarchical clustering has no initialisation and no
    local optimum - given the data and the linkage the tree is determined - so
    the C# merges must match merge for merge, and the distances to rounding.

    The data is small, continuous and tie-free on purpose. A tie in merge
    distance is decided by implementation detail rather than by the data, and a
    fixture should not rest on one.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.sparse.csgraph import connected_components
    from sklearn.cluster import ward_tree
    from sklearn.datasets import make_blobs

    # Spread enough that the neighbour graph is one piece. scikit-learn would
    # otherwise join the pieces itself before building the tree, by a rule the
    # C# side deliberately does not copy.
    x, _ = make_blobs(n_samples=60, n_features=3, centers=4, cluster_std=1.8, random_state=seed)

    trees = {}
    for method in ["ward", "complete", "average", "single"]:
        z = linkage(x, method=method, metric="euclidean")
        trees[method] = z.tolist()

    connectivity = symmetric_knn(x, k_neighbours)
    count, _ = connected_components(connectivity, directed=False)
    if count != 1:
        raise ValueError(f"constrained fixture needs a connected graph; got {count} components")

    rows, cols = np.nonzero(np.triu(connectivity, k=1))
    children, _, _, _, distances = ward_tree(x, connectivity=connectivity, return_distance=True)

    return {
        "name": "hierarchical",
        "x": x.tolist(),
        "neighbours": k_neighbours,
        "edges": [[int(a), int(b)] for a, b in zip(rows, cols)],
        "expected": {
            "linkage": trees,
            "constrained_ward": {
                "children": children.tolist(),
                "distances": distances.tolist(),
            },
        },
    }


def write(fixture: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{fixture['name']}.json"
    path.write_text(json.dumps(fixture, indent=1), encoding="utf-8")
    print(f"  {path.relative_to(OUT.parents[3])}  ({path.stat().st_size // 1024} KB)")


def main() -> None:
    raw, family = make_members()
    prepared = standardise(raw)

    # PCA here is preprocessing, not the thing under test - it is how the EM
    # fixtures get the whitened input the C# side will also be clustering.
    pca = PCA(n_components=0.99, whiten=True, svd_solver="full")
    whitened = pca.fit_transform(prepared) * sign_flips(pca.components_)

    print("writing fixtures:")
    write(em_fixture(whitened, k=4, covariance_type="diag"))
    write(em_fixture(whitened, k=4, covariance_type="full"))
    write(em_fixture(whitened, k=3, covariance_type="spherical"))
    write(quality_fixture(raw, family))
    write(sweep_fixture(raw))
    write(spectral_fixture())
    write(hierarchical_fixture())
    print(f"\n{raw.shape[0]} samples, {raw.shape[1]} columns, "
          f"{pca.n_components_} principal components retained "
          f"({pca.explained_variance_ratio_.sum() * 100:.2f}% of variance)")


if __name__ == "__main__":
    main()
