"""Do the arms agree on the answer?

A speed comparison between implementations that compute different things is
not a speed comparison. This module is what makes the rest of the harness
admissible: for every pair of arms it asks whether the science came out the
same, using the comparison appropriate to each output rather than naive
equality.

The four checks, and why each is the right one:

- **PCA — principal angles between subspaces, as a function of k, plus the
  variance-capture ratio.** Not elementwise difference: components are
  defined up to sign and, within a degenerate eigenvalue block, up to
  rotation. This check caught the ``Q_basis`` bug in phase 2, where singular
  *values* were right and vectors 45 degrees off.

  A single max-angle number is not enough, and measuring taught us why.
  Polariseq vs scanpy on ``heart_10k`` gives max angle **38.4 deg** at k=50
  and median **0.67 deg**. That looks alarming and is not: the angle rises to
  75.7 deg at k=45 and then *falls* to 38.4 at k=50. An angle that is
  non-monotonic in k is the signature of two solvers picking different
  members of a **degenerate eigenvalue block** — cut the block in half and
  the subspaces differ; include all of it and they agree again. A genuine
  solver bug is wrong from k=5 onward.

  Confirmed on the spectrum itself: the components whose correlation
  collapses (29, 41, 45) are exactly those with a relative eigenvalue gap
  below 0.5 % (PC29: 0.11 %, PC41: 0.18 %), and both arms capture the same
  total variance to within **0.17 %**. Neither implementation identifies
  individual components in a flat tail, because they are not identifiable.

  So the verdict rests on two things that *are* well posed: the total
  variance captured, and the subspace angle up to the last well-separated
  component. The full angle profile is reported alongside as information.

- **Clusters — adjusted Rand index, against a seed-to-seed floor.** Cluster
  ids are arbitrary labels; ARI compares partitions without caring what they
  are called. But *what ARI counts as agreement* is a property of the
  dataset, not a constant: on combat_100k the reference scores only 0.74
  against another configuration of itself, so a fixed 0.90 gate failed every
  arm there, scanpy included. The threshold is now the reference against
  itself at a second seed (:func:`agreement_floors`), which asks the
  question worth asking — does this arm agree with the reference at least as
  well as the reference agrees with itself?

- **HVG — Jaccard overlap of the selected gene sets**, plus the size of each,
  since a arm that selects fewer genes has an easier downstream job.

- **UMAP — k-nearest-neighbour overlap, never point positions.** Two
  *sequential* UMAP runs with different seeds disagree more (1.18 RMS) than
  sequential-vs-parallel does (0.38). Point-wise Procrustes between layouts
  measures repulsion noise. Neighbourhood preservation is the property UMAP
  actually claims.

Cells are aligned by barcode before anything is compared, because the arms
may filter in a different order and a comparison of two differently-ordered
matrices is silently meaningless.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

#: Angles below this are numerical noise between two exact solvers
#: (measured: exact Gram PCA vs arpack agree to 0.0004 degrees at k=20 on
#: heart_100k). Applied only up to the last well-separated component.
PCA_ANGLE_TOLERANCE_DEG = 5.0

#: A relative eigenvalue gap below this means the two components either side
#: of it are not individually identified, and no solver can be blamed for
#: ordering them differently. Measured on heart_10k: PC29 has a gap of
#: 0.11 % and the two arms' PC29 correlate at 0.376, while every component
#: with a gap above 1 % correlates above 0.95.
DEGENERATE_GAP = 0.005

#: Two implementations of the same PCA must find the same total variance in
#: their k-dimensional subspace, degeneracy or not. This is the check that
#: does *not* wash out in a flat tail. Measured agreement: 0.17 %.
VARIANCE_RATIO_TOLERANCE = 0.02

#: HVG selection is the step every later check is conditioned on, so it gets
#: its own tolerance. This was 0.984 on heart_10k (1984 of 2000 genes, 16
#: differing each way) until the bin edges were made to match pandas'
#: ``pd.cut``; it is now **1.0** on both heart_10k and combat_10k. The
#: tolerance stays below 1.0 because a future flavour or a tie at the
#: 2000th rank could legitimately differ by a gene or two, and because a
#: check that can only ever pass exactly is a check that gets deleted the
#: first time it is inconvenient.
HVG_JACCARD_TOLERANCE = 0.95

#: Fallback ARI threshold, used only when no seed-to-seed floor is
#: available for a dataset. 0.90 is the level the phase-5 work established
#: as "the same clustering" — but it is a constant, and clustering agreement
#: is not constant across datasets.
#:
#: Measured on combat_100k: `scanpy-default` vs `scanpy-tuned` reaches ARI
#: **0.74**, and the reference against *itself* at a second seed is no
#: better. Holding any arm to 0.90 there demands it agree with the reference
#: more closely than the reference agrees with itself — an impossible bar,
#: and one that says nothing about the arm. Where a floor exists, it is used
#: instead of this constant (see :func:`agreement_floors`).
ARI_TOLERANCE = 0.90

#: How far below the seed-to-seed floor an arm may sit and still count as
#: agreeing. Floors are themselves estimates from a single pair of seeds, so
#: a small allowance keeps the check from firing on the floor's own noise.
FLOOR_MARGIN = 0.05


def principal_angles_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angles between the column spaces of two embeddings, in degrees."""
    qa, _ = np.linalg.qr(np.asarray(a, dtype=np.float64))
    qb, _ = np.linalg.qr(np.asarray(b, dtype=np.float64))
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def spectral_gaps(embedding: np.ndarray) -> np.ndarray:
    """Relative eigenvalue gaps ``(v[i] - v[i+1]) / v[i]`` from an embedding.

    The variance of PCA column *i* is eigenvalue *i*, so the spectrum is
    recoverable from the embedding alone and no arm has to report it.
    """
    v = np.asarray(embedding, dtype=np.float64).var(axis=0)
    v = np.maximum(v, np.finfo(np.float64).tiny)
    return (v[:-1] - v[1:]) / v[:-1]


def last_identified_component(embedding: np.ndarray,
                              gap: float = DEGENERATE_GAP) -> int:
    """The largest k for which the first k components are well separated.

    Beyond this, disagreement between two solvers is a property of the data,
    not of either implementation, and gating on it would be gating on noise.
    """
    gaps = spectral_gaps(embedding)
    degenerate = np.flatnonzero(gaps < gap)
    return int(degenerate[0]) if degenerate.size else int(embedding.shape[1])


def pca_agreement(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Do two PCA embeddings represent the same decomposition?

    Two well-posed questions (which decide the verdict) and one descriptive
    profile (which does not, because in a degenerate tail it cannot).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    k_full = min(a.shape[1], b.shape[1])
    a, b = a[:, :k_full], b[:, :k_full]

    # 1. Same total variance in the k-dimensional subspace. Robust to
    #    degeneracy: rotating within a degenerate block preserves the sum.
    va, vb = float(a.var(0).sum()), float(b.var(0).sum())
    ratio = va / vb if vb else float("nan")

    # 2. Subspace angle up to the last well-separated component, using the
    #    *reference* arm's spectrum to choose k so that our own solver's
    #    output cannot move the goalposts.
    k_id = max(2, min(last_identified_component(b), k_full))
    angles_id = principal_angles_deg(a[:, :k_id], b[:, :k_id])
    angles_full = principal_angles_deg(a, b)

    profile = {
        str(k): round(float(principal_angles_deg(a[:, :k], b[:, :k]).max()), 4)
        for k in sorted({5, 10, 20, k_id, k_full}) if 2 <= k <= k_full
    }
    return {
        "pca_variance_ratio": round(ratio, 5),
        "pca_k_identified": k_id,
        "pca_max_angle_deg_at_k_identified": round(float(angles_id.max()), 4),
        "pca_max_angle_deg_full": round(float(angles_full.max()), 4),
        "pca_median_angle_deg_full": round(float(np.median(angles_full)), 4),
        "pca_angle_profile_by_k": profile,
        "pca_degenerate_components": [
            int(i + 1) for i in np.flatnonzero(spectral_gaps(b) < DEGENERATE_GAP)
        ],
        "pca_ok": bool(
            abs(ratio - 1.0) < VARIANCE_RATIO_TOLERANCE
            and angles_id.max() < PCA_ANGLE_TOLERANCE_DEG
        ),
    }

def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """ARI without a scikit-learn dependency."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    n = ai.size
    if n == 0:
        return float("nan")
    cont = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(cont, (ai, bi), 1)

    def comb2(x: np.ndarray | np.int64) -> np.int64:
        return (x * (x - 1) // 2).sum()

    sum_ij = comb2(cont)
    sum_i = comb2(cont.sum(axis=1))
    sum_j = comb2(cont.sum(axis=0))
    total = n * (n - 1) // 2
    # float64: the int64 product of pair counts wraps above ~400K cells
    # (found 2026-09-28; every cluster ARI at that size was wrong).
    expected = float(sum_i) * float(sum_j) / float(total) if total else 0.0
    maximum = (sum_i + sum_j) / 2
    denom = maximum - expected
    return float((sum_ij - expected) / denom) if denom else 1.0


def knn_overlap(a: np.ndarray, b: np.ndarray, k: int = 15, sample: int = 2000,
                seed: int = 0) -> float:
    """Mean fraction of shared k-nearest neighbours between two layouts.

    Sampled, because the exact computation is O(n^2) and the estimate is what
    the claim needs. float64 throughout: squared-distance kNN in float32 on
    an embedding suffers catastrophic cancellation and silently returns wrong
    neighbours — a mistake already paid for once in this repo.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = a.shape[0]
    if n < k + 1:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample, n), replace=False)

    def nn(x: np.ndarray, q: np.ndarray) -> np.ndarray:
        d = ((x[None, :, :] - q[:, None, :]) ** 2).sum(-1)
        d[np.arange(len(q)), idx] = np.inf
        return np.argpartition(d, k, axis=1)[:, :k]

    na, nb = nn(a, a[idx]), nn(b, b[idx])
    return float(
        np.mean(
            [len(set(x.tolist()) & set(y.tolist())) / k
             for x, y in zip(na, nb, strict=True)]
        )
    )


def _load(paths: dict[str, str], key: str) -> Any:
    p = paths.get(key)
    if not p or not Path(p).exists():
        return None
    if p.endswith(".npy"):
        return np.load(p)
    return Path(p).read_text().splitlines()


def compare(ref: dict[str, Any], other: dict[str, Any], *, n_neighbors: int = 15,
            floors: dict[str, float] | None = None) -> dict[str, Any]:
    """Compare two run records' artifacts. ``ref`` is the reference arm.

    ``floors`` carries the reference-against-itself scores for this dataset
    (see :func:`agreement_floors`). Where a floor is present it replaces the
    fixed tolerance, because the question worth asking is not "does this arm
    hit 0.90" but "does it agree with the reference at least as well as the
    reference agrees with itself".
    """
    ra = ref.get("outputs", {}).get("artifacts", {})
    oa = other.get("outputs", {}).get("artifacts", {})
    out: dict[str, Any] = {
        "reference": ref.get("arm"),
        "arm": other.get("arm"),
        "checks": {},
        "verdict": "unknown",
        "skipped": [],
    }

    ref_names, oth_names = _load(ra, "obs_names"), _load(oa, "obs_names")
    order_ref = order_oth = None
    if ref_names and oth_names:
        common = sorted(set(ref_names) & set(oth_names))
        if common:
            pr = {n: i for i, n in enumerate(ref_names)}
            po = {n: i for i, n in enumerate(oth_names)}
            order_ref = np.array([pr[n] for n in common])
            order_oth = np.array([po[n] for n in common])
            out["checks"]["cells_compared"] = len(common)
            out["checks"]["cells_only_in_reference"] = len(set(ref_names) - set(oth_names))
            out["checks"]["cells_only_in_arm"] = len(set(oth_names) - set(ref_names))
    else:
        out["skipped"].append(
            "cell alignment: one arm did not emit obs_names, so rows are "
            "compared positionally and the result is only valid if neither "
            "arm reordered or filtered differently"
        )

    def align(x: np.ndarray, which: str) -> np.ndarray:
        o = order_ref if which == "ref" else order_oth
        return x if o is None else x[o]

    pa, pb = _load(ra, "pca"), _load(oa, "pca")
    if pa is not None and pb is not None:
        pa, pb = align(pa, "ref"), align(pb, "oth")
        m = min(pa.shape[0], pb.shape[0])
        out["checks"].update(pca_agreement(pa[:m], pb[:m]))

    la, lb = _load(ra, "labels"), _load(oa, "labels")
    if la is not None and lb is not None:
        la, lb = align(la, "ref"), align(lb, "oth")
        m = min(la.shape[0], lb.shape[0])
        ari = adjusted_rand_index(la[:m], lb[:m])
        out["checks"]["cluster_ari"] = round(ari, 5)
        out["checks"]["n_clusters_reference"] = int(len(set(la[:m].tolist())))
        out["checks"]["n_clusters_arm"] = int(len(set(lb[:m].tolist())))
        floor = (floors or {}).get("cluster_ari")
        if floor is not None:
            out["checks"]["cluster_ari_floor"] = round(float(floor), 5)
            out["checks"]["cluster_ok"] = bool(ari >= floor - FLOOR_MARGIN)
        else:
            out["checks"]["cluster_ari_floor"] = None
            out["checks"]["cluster_ok"] = bool(ari >= ARI_TOLERANCE)

    ha, hb = _load(ra, "hvg"), _load(oa, "hvg")
    if ha and hb:
        sa, sb = set(ha), set(hb)
        jac = len(sa & sb) / len(sa | sb)
        out["checks"]["hvg_jaccard"] = round(jac, 5)
        out["checks"]["hvg_n_reference"], out["checks"]["hvg_n_arm"] = len(sa), len(sb)
        out["checks"]["hvg_n_differing"] = len(sa ^ sb) // 2
        out["checks"]["hvg_ok"] = bool(jac >= HVG_JACCARD_TOLERANCE)

    ua, ub = _load(ra, "umap"), _load(oa, "umap")
    if ua is not None and ub is not None:
        ua, ub = align(ua, "ref"), align(ub, "oth")
        m = min(ua.shape[0], ub.shape[0])
        overlap = knn_overlap(ua[:m], ub[:m], k=n_neighbors)
        out["checks"]["umap_knn_overlap"] = round(overlap, 5)
        ufloor = (floors or {}).get("umap_knn_overlap")
        out["checks"]["umap_knn_overlap_floor"] = (
            round(float(ufloor), 5) if ufloor is not None else None
        )

    # Every check after `hvg` runs on whatever genes that step selected, so a
    # difference there propagates and would otherwise be misattributed to the
    # solver downstream of it. Measured on heart_10k: the pipelines' PCA
    # subspaces differ by 29.7 deg at k=28, which reads as a PCA defect and is
    # not one — the two arms each select 16 genes the other does not, of 2000
    # (Jaccard 0.984), and on an *identical* matrix the same two PCA solvers
    # agree to 0.0009 deg with a variance ratio of exactly 1.0.
    #
    # So a verdict of "disagree" is only allowed to name the step that
    # actually diverged first. Anything downstream of an upstream difference
    # is reported as confounded, not as broken.
    jac = out["checks"].get("hvg_jaccard")
    confounded = jac is not None and jac < 1.0
    if confounded:
        out["confounded_by"] = (
            f"upstream HVG selection differs (Jaccard {jac}); the PCA, cluster "
            f"and UMAP checks below compare pipelines built on different gene "
            f"sets, so they cannot isolate those kernels. Compare kernels on an "
            f"identical matrix to do that."
        )

    flags = [v for k, v in out["checks"].items() if k.endswith("_ok")]
    if not flags:
        out["verdict"] = "unknown"
    elif all(flags):
        out["verdict"] = "agree"
    elif confounded:
        out["verdict"] = "confounded"
    else:
        out["verdict"] = "disagree"
    return out


def compare_all(records: list[dict[str, Any]], reference: str = "scanpy-tuned",
                **kw: Any) -> list[dict[str, Any]]:
    """Compare every successful arm against the reference, per dataset.

    The reference defaults to ``scanpy-tuned`` on purpose: scanpy is the
    semantic authority here. Making our own arm the reference would mean
    reporting how far scanpy deviates from us, which is backwards.
    """
    out: list[dict[str, Any]] = []
    # Grouped by (dataset, UMAP variant) for the same reason `analyze._group`
    # is: a with-UMAP reference must not be compared against a without-UMAP
    # arm — the UMAP overlap check would silently compare against nothing.
    by_dataset: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for r in records:
        if r.get("status") == "ok":
            umap = bool(r.get("params", {}).get("with_umap", False))
            by_dataset.setdefault((r["dataset"]["label"], umap), []).append(r)

    # One record per arm. Equivalence is a property of the algorithm, not of a
    # repeat: comparing every rep emitted the same row once per rep (four
    # identical pbmc3k lines in the 2026-09-09 campaign) and read every rep's
    # artifacts, which is also why artifacts could not be pruned. Prefer a
    # record whose artifacts are still on disk, so pruning duplicates cannot
    # leave the comparison pointing at files that were reclaimed.
    def _has_artifacts(r: dict[str, Any]) -> bool:
        arts = r.get("outputs", {}).get("artifacts", {})
        return bool(arts) and all(
            Path(v).exists() for v in arts.values() if isinstance(v, str)
        )

    for key, group in list(by_dataset.items()):
        best: dict[str, dict[str, Any]] = {}
        for r in group:
            arm = r["arm"]
            if arm not in best or (
                _has_artifacts(r) and not _has_artifacts(best[arm])
            ):
                best[arm] = r
        by_dataset[key] = [best[a] for a in sorted(best)]
    # Floors come from the reference against itself at a second seed, so they
    # are computed once over every record and then applied per dataset.
    floors = agreement_floors(
        records, reference,
        **({"n_neighbors": kw["n_neighbors"]} if "n_neighbors" in kw else {}),
    )
    for (label, umap), group in sorted(by_dataset.items()):
        refs = [r for r in group if r["arm"] == reference]
        if not refs:
            continue
        ref = refs[0]
        key = f"{label} (+UMAP)" if umap else label
        for other in group:
            if other["arm"] == reference:
                continue
            row = compare(ref, other, floors=floors.get(key), **kw)
            row["dataset"] = key
            out.append(row)
    return out


def agreement_floors(
    records: list[dict[str, Any]], reference: str = "scanpy-tuned",
    seed_arm: str = "scanpy-tuned-seed1", *, n_neighbors: int = 15,
) -> dict[str, dict[str, float]]:
    """What the reference scores against *itself* at a second seed.

    The same implementation, same tuning, one different seed, scored exactly
    the way a cross-implementation check is scored. This is the only honest
    yardstick for a stochastic output: if the reference against itself
    scores 0.74, an arm scoring 0.74 has agreed as well as agreement gets,
    and a fixed 0.90 threshold is measuring the dataset, not the arm.

    Two floors per dataset variant:

    - ``umap_knn_overlap`` — layout neighbourhood preservation. Measured:
      0.372 at heart_10k, 0.187 at heart_50k. It halves between them, so a
      floor from one size must never be quoted at another.
    - ``cluster_ari`` — Leiden partition agreement. Measured on combat_100k:
      scanpy-default vs scanpy-tuned reaches only 0.74, which is why that
      dataset tripped a fixed 0.90 gate for *every* arm, scanpy included.

    Point-identity of UMAP layouts is repulsion noise (chapter 00,
    pitfall 11); the floor is what makes these numbers interpretable rather
    than decorative.
    """
    out: dict[str, dict[str, float]] = {}
    groups: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for r in records:
        if r.get("status") != "ok":
            continue
        umap = bool(r.get("params", {}).get("with_umap", False))
        groups.setdefault((r["dataset"]["label"], umap), []).append(r)
    for (label, umap), group in sorted(groups.items()):
        refs = [r for r in group if r["arm"] == reference]
        seeds = [r for r in group if r["arm"] == seed_arm]
        if not refs or not seeds:
            continue
        checks = compare(refs[0], seeds[0], n_neighbors=n_neighbors)["checks"]
        found = {
            metric: float(checks[metric])
            for metric in ("umap_knn_overlap", "cluster_ari")
            if checks.get(metric) is not None
        }
        if found:
            out[f"{label} (+UMAP)" if umap else label] = found
    return out


def seed_floors(
    records: list[dict[str, Any]], reference: str = "scanpy-tuned",
    seed_arm: str = "scanpy-tuned-seed1", *, n_neighbors: int = 15,
) -> dict[str, float]:
    """The UMAP-overlap floors alone, as ``{dataset: overlap}``.

    Kept because the report and table code want just this one series; new
    code should call :func:`agreement_floors`.
    """
    return {
        label: f["umap_knn_overlap"]
        for label, f in agreement_floors(
            records, reference, seed_arm, n_neighbors=n_neighbors
        ).items()
        if "umap_knn_overlap" in f
    }


def main(results_dir: str, reference: str = "scanpy-tuned") -> None:
    from .record import load_records

    rows = compare_all(load_records(results_dir), reference=reference)
    print(json.dumps(rows, indent=1))
