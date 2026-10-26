"""Tests for the telemetry harness.

The harness produces the numbers that go into the README and the results, so a
bug here is a bug in a published claim. These tests concentrate on the two
things that would be worst to get wrong silently: the *gate* that decides
whether a run is quotable, and the *equivalence* checks that decide whether a
fast answer is also the right answer. Both are tested against cases whose
correct output is known analytically, not against the implementation itself.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from telemetry.analyze import quality, summarize  # noqa: E402
from telemetry.arms import ARMS, DEFAULT_ARMS  # noqa: E402
from telemetry.equivalence import (  # noqa: E402
    adjusted_rand_index,
    compare_all,
    knn_overlap,
    principal_angles_deg,
)
from telemetry.record import SCHEMA, Dataset, Params, RunRecord, load_records  # noqa: E402
from telemetry.sampler import MemorySampler, Timeline  # noqa: E402

FIXTURE = BENCH.parent / "tests/data/medium_10kx2k.h5ad"


class TestSampler:
    def test_records_a_series_on_a_shared_clock(self) -> None:
        tl = Timeline()
        s = MemorySampler(interval_s=0.005).start(origin=tl.origin)
        s.mark_baseline()
        with tl.step("work"):
            time.sleep(0.15)
        s.stop()

        assert len(s.t) > 5
        assert s.baseline["rss"] > 0
        assert s.peaks()["rss"] >= s.baseline["rss"]
        # Steps and samples share an origin, so a step's span indexes into
        # the series. That is what lets the analysis attribute memory to work.
        step = tl.steps[0]
        during = [t for t in s.t if step.t_begin <= t <= (step.t_end or 0)]
        assert during, "no samples fell inside the step window"

    def test_sees_an_allocation(self) -> None:
        s = MemorySampler(interval_s=0.005)
        # Every tick reads USS here; on Windows the default reads it at 2 Hz
        # (see USS_INTERVAL_S), slower than this 50 ms test.
        s.uss_interval_s = 0.0
        s.start()
        s.mark_baseline()
        block = bytearray(120_000_000)
        for i in range(0, len(block), 4096):  # fault the pages in
            block[i] = 1
        time.sleep(0.05)
        s.stop()
        assert s.peaks()["uss"] - s.baseline["uss"] > 80_000_000
        del block

    def test_reports_its_own_blindness(self) -> None:
        """The gap statistics must be real, or the gate they feed is theatre."""
        s = MemorySampler(interval_s=0.005).start()
        time.sleep(0.05)
        # A tight C-level loop holds the GIL and starves the sampler; this is
        # the same mechanism that makes Rust kernels invisible to it.
        sum(range(12_000_000))
        time.sleep(0.05)
        s.stop()
        g = s.gaps()
        assert g["n"] >= 2
        assert g["max_s"] >= g["median_s"]
        assert 0.0 <= g["blind_fraction"] <= 1.0

    def test_empty_sampler_reports_total_blindness(self) -> None:
        assert MemorySampler().gaps()["blind_fraction"] == 1.0


class TestRecord:
    def test_round_trips_through_disk(self, tmp_path: Path) -> None:
        rec = RunRecord(
            arm="polariseq-ram", arm_label="x",
            dataset=Dataset(label="d", path="p"), params=Params(),
        )
        p = rec.write(tmp_path)
        assert p.exists()
        loaded = load_records(tmp_path)
        assert len(loaded) == 1
        assert loaded[0]["arm"] == "polariseq-ram"
        assert loaded[0]["run_id"] == rec.run_id

    def test_refuses_a_foreign_schema(self, tmp_path: Path) -> None:
        """Silently mixing schema versions is how a table becomes a lie."""
        (tmp_path / "old.json").write_text(json.dumps({"schema": "something/0"}))
        assert load_records(tmp_path) == []

    def test_schema_is_stamped(self) -> None:
        rec = RunRecord(arm="a", arm_label="a", dataset=Dataset("d", "p"), params=Params())
        assert rec.to_json()["schema"] == SCHEMA


class TestEquivalence:
    def test_principal_angles_ignore_basis_and_sign(self) -> None:
        """The check that caught the Q_basis bug: same span, different basis."""
        rng = np.random.default_rng(0)
        a = rng.normal(size=(400, 8))
        rot, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        assert principal_angles_deg(a, a @ rot).max() < 1e-4
        assert principal_angles_deg(a, -a).max() < 1e-4

    def test_principal_angles_detect_a_different_subspace(self) -> None:
        rng = np.random.default_rng(1)
        assert principal_angles_deg(
            rng.normal(size=(400, 8)), rng.normal(size=(400, 8))
        ).max() > 45

    def test_ari_is_invariant_to_relabelling(self) -> None:
        x = np.repeat([0, 1, 2], 80)
        assert adjusted_rand_index(x, x) == pytest.approx(1.0)
        assert adjusted_rand_index(x, (x + 1) % 3) == pytest.approx(1.0)
        rng = np.random.default_rng(0)
        assert abs(adjusted_rand_index(x, rng.integers(0, 3, x.size))) < 0.1

    def test_knn_overlap_is_1_for_identical_layouts(self) -> None:
        e = np.random.default_rng(0).normal(size=(200, 4))
        assert knn_overlap(e, e, k=10, sample=60) == pytest.approx(1.0)
        other = np.random.default_rng(1).normal(size=(200, 4))
        assert knn_overlap(e, other, k=10, sample=60) < 0.2

    def test_compare_all_needs_a_reference(self) -> None:
        rec = RunRecord(arm="polariseq-ram", arm_label="x",
                        dataset=Dataset("d", "p"), params=Params()).to_json()
        assert compare_all([rec], reference="scanpy-tuned") == []


class TestAgreementFloors:
    """A stochastic output is judged against the reference's own variability.

    Measured on combat_100k: scanpy-default vs scanpy-tuned reaches ARI
    0.74, so a fixed 0.90 gate failed *every* arm on that dataset, scanpy
    included. A threshold that the reference itself cannot meet is measuring
    the dataset, not the arm.
    """

    @staticmethod
    def _labels(tmp_path: Path, name: str, labels: np.ndarray) -> dict[str, object]:
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "labels.npy", labels.astype(np.int32))
        (d / "obs_names.txt").write_text("\n".join(f"c{i}" for i in range(labels.size)))
        return {
            "arm": name,
            "outputs": {"artifacts": {
                "labels": str(d / "labels.npy"),
                "obs_names": str(d / "obs_names.txt"),
            }},
        }

    def _pair(self, tmp_path: Path) -> tuple[dict, dict, float]:
        from telemetry.equivalence import compare

        rng = np.random.default_rng(0)
        a = np.repeat([0, 1, 2, 3], 60)
        b = a.copy()
        b[rng.choice(a.size, 45, replace=False)] = rng.integers(0, 4, 45)
        ref = self._labels(tmp_path, "ref", a)
        arm = self._labels(tmp_path, "arm", b)
        return ref, arm, compare(ref, arm)["checks"]["cluster_ari"]

    def test_without_a_floor_the_fixed_tolerance_applies(self, tmp_path: Path) -> None:
        from telemetry.equivalence import ARI_TOLERANCE, compare

        ref, arm, ari = self._pair(tmp_path)
        assert ari < ARI_TOLERANCE
        checks = compare(ref, arm)["checks"]
        assert checks["cluster_ok"] is False
        assert checks["cluster_ari_floor"] is None

    def test_an_arm_at_the_floor_agrees(self, tmp_path: Path) -> None:
        from telemetry.equivalence import compare

        ref, arm, ari = self._pair(tmp_path)
        checks = compare(ref, arm, floors={"cluster_ari": ari})["checks"]
        assert checks["cluster_ok"] is True, (
            "an arm that matches the reference's own seed-to-seed agreement "
            "must not be marked as disagreeing"
        )
        assert checks["cluster_ari_floor"] == pytest.approx(ari)

    def test_a_floor_does_not_excuse_a_genuinely_worse_arm(self, tmp_path: Path) -> None:
        """The floor must not become a way to pass everything."""
        from telemetry.equivalence import compare

        ref, arm, _ = self._pair(tmp_path)
        assert compare(ref, arm, floors={"cluster_ari": 0.99})["checks"]["cluster_ok"] is False

    def test_floors_need_the_seed_arm(self) -> None:
        from telemetry.equivalence import agreement_floors

        assert agreement_floors([_record()]) == {}


def _record(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema": SCHEMA, "arm": "polariseq-ram", "arm_label": "ours",
        "dataset": {"label": "d"}, "status": "ok", "wall_s": 1.0,
        "params": {"with_umap": False, "n_hvg": 100, "n_pcs": 10},
        "steps": [{"name": "pca", "t_begin": 0.0, "t_end": 1.0, "wall_s": 1.0}],
        "child": {"gaps": {"blind_fraction": 0.0, "uss_available": True},
                  "peaks": {"rss": 1_000, "uss": 900}, "baseline": {"uss": 100}},
        "parent": {"peak_rss": 1_000, "swap_out_delta_bytes": 0},
        "peaks": {"child_rss": 1_000, "child_uss": 900, "parent_rss": 1_000},
        "disk": {"spill_peak_bytes": 0}, "outputs": {"n_clusters": 5},
    }
    base.update(kw)
    return base


class TestQualityGate:
    def test_a_clean_record_passes(self) -> None:
        q = quality(_record())
        assert q["ok"] and not q["blocking"] and not q["caveats"]
        assert q["uss_series_usable"]

    def test_swapping_blocks_the_run(self) -> None:
        q = quality(_record(parent={"peak_rss": 1_000, "swap_out_delta_bytes": 900_000_000}))
        assert not q["ok"]
        assert any("swap" in r for r in q["blocking"])

    def test_disagreeing_samplers_block_the_run(self) -> None:
        q = quality(_record(peaks={"child_rss": 1_000, "child_uss": 900,
                                   "parent_rss": 5_000}))
        assert not q["ok"]
        assert any("disagree" in r for r in q["blocking"])

    def test_a_blind_sampler_is_a_caveat_not_a_block(self) -> None:
        """GIL blindness invalidates the USS curve alone.

        Peaks and wall time come from the parent, outside the process, so a
        blind in-process sampler must not throw away an otherwise good run —
        every arm trips this, ours and scanpy's alike.
        """
        q = quality(_record(child={"gaps": {"blind_fraction": 0.6, "uss_available": True},
                                   "peaks": {"rss": 1_000, "uss": 900}, "baseline": {}}))
        assert q["ok"], "blindness must not block the run"
        assert not q["uss_series_usable"]
        assert any("blind" in c for c in q["caveats"])

    def test_a_failed_run_blocks(self) -> None:
        assert not quality(_record(status="oom", note="killed"))["ok"]

    def test_summarize_excludes_blocked_runs_from_medians(self) -> None:
        good = _record(wall_s=1.0)
        bad = _record(wall_s=99.0,
                      parent={"peak_rss": 1_000, "swap_out_delta_bytes": 10**9})
        rows = summarize([good, bad])
        assert len(rows) == 1
        assert rows[0]["wall_s"] == 1.0, "a swapping run must not move the median"
        assert rows[0]["n_ok"] == 1 and rows[0]["n_reps"] == 2

    def test_summarize_never_blends_umap_variants(self) -> None:
        """The ladder runs --no-umap; UMAP is measured separately at the small
        sizes. Both live in one results directory by design — averaging a
        with-UMAP wall time into a without-UMAP median would be a silent lie
        in whichever direction it erred."""
        no_umap = _record(wall_s=10.0)
        with_umap = _record(wall_s=30.0,
                            params={"with_umap": True, "n_hvg": 100, "n_pcs": 10})
        rows = summarize([no_umap, with_umap])
        assert len(rows) == 2, "one row per (dataset, arm, UMAP variant)"
        by_flag = {r["umap"]: r for r in rows}
        assert by_flag[False]["wall_s"] == 10.0
        assert by_flag[True]["wall_s"] == 30.0
        assert by_flag[False]["dataset_display"] == "d"
        assert by_flag[True]["dataset_display"] == "d (+UMAP)"


class TestArms:
    def test_every_arm_declares_its_tuning(self) -> None:
        """An undeclared tuning choice is an unreviewable one."""
        for name, arm in ARMS.items():
            assert arm.name == name
            assert arm.label and len(arm.tuning_notes) > 40, name

    def test_default_arms_exist_and_cover_both_scanpy_configs(self) -> None:
        assert set(DEFAULT_ARMS) <= set(ARMS)
        # Comparing only against scanpy's defaults overstates the result; the
        # tuned arm must always be in the default set.
        assert "scanpy-tuned" in DEFAULT_ARMS

    def test_seed_floor_arm_is_opt_in_and_never_matches_the_campaign_seed(self) -> None:
        from telemetry.arms.scanpy_arm import second_seed

        assert "scanpy-tuned-seed1" in ARMS
        assert "scanpy-tuned-seed1" not in DEFAULT_ARMS, (
            "the floor arm duplicates scanpy-tuned's timings; it must be opt-in"
        )
        # If the forced seed could equal the campaign seed, the 'floor' would
        # be an implementation against itself at the *same* seed (~1.0) and
        # every cross-implementation overlap would look remarkable.
        for primary in (0, 1, 2, 42):
            assert second_seed(primary) != primary


class TestSeedFloors:
    def _umap_records(self, tmp_path: Path) -> list[dict[str, object]]:
        import numpy as np

        rng = np.random.default_rng(0)
        a = rng.normal(size=(200, 2))
        b = rng.normal(size=(200, 2))
        pa, pb = tmp_path / "a.npy", tmp_path / "b.npy"
        np.save(pa, a)
        np.save(pb, b)
        names = tmp_path / "names.txt"
        names.write_text("\n".join(f"c{i}" for i in range(200)))

        def rec(arm: str, umap: Path) -> dict[str, object]:
            return _record(
                arm=arm,
                dataset={"label": "d", "path": ""},
                params={"with_umap": True, "n_hvg": 100, "n_pcs": 10},
                outputs={"artifacts": {"umap": str(umap), "obs_names": str(names)}},
            )

        return [rec("scanpy-tuned", pa), rec("scanpy-tuned-seed1", pb)]

    def test_floor_is_the_same_implementation_at_two_seeds(self, tmp_path: Path) -> None:
        from telemetry.equivalence import knn_overlap, seed_floors

        floors = seed_floors(self._umap_records(tmp_path), n_neighbors=10)
        assert set(floors) == {"d (+UMAP)"}
        import numpy as np

        expected = knn_overlap(
            np.load(tmp_path / "a.npy"), np.load(tmp_path / "b.npy"), k=10
        )
        assert floors["d (+UMAP)"] == pytest.approx(expected, abs=1e-5)

    def test_no_seed_arm_no_floor(self, tmp_path: Path) -> None:
        from telemetry.equivalence import seed_floors

        recs = self._umap_records(tmp_path)
        assert seed_floors([recs[0]]) == {}

    def test_floor_is_a_comparison_row_too(self, tmp_path: Path) -> None:
        """The floor is not a private number: the seed arm appears in
        compare_all like any arm, so its provenance is the records."""
        from telemetry.equivalence import compare_all

        rows = compare_all(self._umap_records(tmp_path), n_neighbors=10)
        pair = [r for r in rows if r["arm"] == "scanpy-tuned-seed1"]
        assert len(pair) == 1
        assert "umap_knn_overlap" in pair[0]["checks"]


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture missing")
class TestPredictions:
    def test_plan_predictions_cover_the_ladder_arms(self) -> None:
        """Panel D's prediction line is built from each record's own pinned
        budget (`params.ram_gb`/`params.threads`), never the ambient one.

        Calling `ps.plan()` directly would read whatever `ps.budget()` is at
        the moment the report is regenerated, not what the campaign was
        pinned to — measured to invert the pred/meas ratio between dataset
        sizes when the two diverge. `ram_gb` is therefore required on the
        record, not optional; a record without it must be skipped, not
        silently predicted against the wrong budget (see the next test)."""
        from run_telemetry import collect_predictions

        records = [
            {
                "arm": "polariseq-ram",
                "dataset": {"label": "fixture", "path": str(FIXTURE)},
                "params": {"with_umap": False, "n_hvg": 100, "n_pcs": 10,
                          "ram_gb": 4.0, "threads": 4},
            },
            {
                "arm": "polariseq-spill",
                "dataset": {"label": "fixture", "path": str(FIXTURE)},
                "params": {"with_umap": False, "n_hvg": 100, "n_pcs": 10,
                          "ram_gb": 4.0, "threads": 4},
            },
        ]
        preds = collect_predictions(records, with_umap=False)
        assert set(preds) == {"polariseq-ram", "polariseq-spill"}
        for arm, pts in preds.items():
            assert len(pts) == 1
            n_cells, peak_gb = pts[0]
            assert n_cells == 10_000
            assert peak_gb > 0, f"{arm}: a zero prediction brackets nothing"

    def test_no_polariseq_arms_no_predictions(self) -> None:
        from run_telemetry import collect_predictions

        records = [{"arm": "scanpy-tuned",
                    "dataset": {"label": "d", "path": "nowhere.h5ad"},
                    "params": {"with_umap": False}}]
        assert collect_predictions(records, with_umap=False) == {}

    def test_an_unpinned_record_is_skipped_not_mispredicted(self) -> None:
        """A record with no `ram_gb` (an unpinned run, or an old-schema one)
        must not silently fall back to the ambient budget — that fallback is
        exactly the bug this function was fixed to remove."""
        from run_telemetry import collect_predictions

        records = [{
            "arm": "polariseq-ram",
            "dataset": {"label": "fixture", "path": str(FIXTURE)},
            "params": {"with_umap": False, "n_hvg": 100, "n_pcs": 10},
        }]
        assert collect_predictions(records, with_umap=False) == {}


class TestTrialStatistics:
    """Every published number is `mean ± half-range (n)`, so the thing that
    computes it has to be right about small samples in particular."""

    def test_summarises_a_distribution(self) -> None:
        from telemetry.analyze import stat

        st = stat([1.0, 2.0, 3.0, 4.0, 5.0])
        assert st is not None
        assert st.n == 5
        assert st.mean == pytest.approx(3.0)
        assert st.median == pytest.approx(3.0)
        assert (st.lo, st.hi) == (1.0, 5.0)
        assert st.sd == pytest.approx(1.5811, abs=1e-3)  # sample sd, ddof=1
        assert not st.underpowered

    def test_marks_under_powered_samples(self) -> None:
        """Two trials is not a measurement, and must not look like one."""
        from telemetry.analyze import MIN_TRIALS_FOR_STATS, stat

        st = stat([1.0, 2.0])
        assert st.underpowered
        assert "*" in st.format()
        assert stat([1.0] * MIN_TRIALS_FOR_STATS).underpowered is False

    def test_ci_degenerates_rather_than_inventing_precision(self) -> None:
        """With n < 3 a normal-approximation CI is meaningless, so the
        reported interval falls back to the observed range."""
        from telemetry.analyze import stat

        assert stat([2.0, 4.0]).ci95 == (2.0, 4.0)
        assert stat([5.0]).ci95 == (5.0, 5.0)
        wide = stat([1.0, 2.0, 3.0, 4.0, 5.0]).ci95
        assert wide[0] > 1.0 and wide[1] < 5.0, "CI on the mean sits inside the range"

    def test_zero_variance_is_not_a_divide_by_zero(self) -> None:
        from telemetry.analyze import stat

        st = stat([2.0, 2.0, 2.0, 2.0, 2.0])
        assert st.sd == 0.0 and st.rel_spread == 0.0
        assert st.ci95 == (2.0, 2.0)

    def test_empty_input_says_nothing(self) -> None:
        from telemetry.analyze import stat

        assert stat([]) is None
        assert stat([float("nan")]) is None

    def test_rows_carry_per_metric_stats(self) -> None:
        from telemetry.analyze import summarize

        rows = summarize([_record(wall_s=1.0), _record(wall_s=3.0)])
        st = rows[0]["stats"]
        assert st["wall_s"]["n"] == 2
        assert st["wall_s"]["mean"] == pytest.approx(2.0)
        assert st["wall_s"]["min"] == 1.0 and st["wall_s"]["max"] == 3.0
        assert "pca" in st["steps"]


class TestPublicDatasets:
    """The validation ladder must be real, public and self-describing —
    a reviewer should not need our private benchmark corpus."""

    def test_registry_is_well_formed(self) -> None:
        from telemetry.datasets import DEFAULT_LADDER, PUBLIC_DATASETS

        assert set(DEFAULT_LADDER) <= set(PUBLIC_DATASETS)
        for name, ds in PUBLIC_DATASETS.items():
            assert ds.name == name
            assert ds.url.startswith("https://"), name
            assert ds.kind in {"10x_h5", "10x_mtx_targz"}, name
            assert ds.approx_cells > 0 and ds.approx_mb > 0, name
            assert len(ds.description) > 20, name

    def test_ladder_spans_a_range_of_sizes(self) -> None:
        """A single size validates a single point; the claim is about scaling."""
        from telemetry.datasets import DEFAULT_LADDER, PUBLIC_DATASETS

        sizes = sorted(PUBLIC_DATASETS[n].approx_cells for n in DEFAULT_LADDER)
        assert len(DEFAULT_LADDER) >= 3
        assert sizes[-1] >= 5 * sizes[0], f"ladder too narrow: {sizes}"

    def test_unknown_dataset_is_rejected(self) -> None:
        from telemetry.datasets import fetch

        with pytest.raises(KeyError):
            fetch("not-a-dataset", "/tmp")


class TestSyntheticFallback:
    def test_generated_data_matches_real_sparsity(self, tmp_path: Path) -> None:
        """Cost scales with nonzeros, so a generator at the wrong density
        measures the wrong workload. Real droplet data is 2.6-6.3% dense
        (measured: pbmc3k 2.6%, pbmc10k 6.3%); a naive Poisson gives 34%.
        """
        pytest.importorskip("anndata")
        from telemetry.arms.base import dataset_facts
        from validate import SYNTHETIC_DENSITY, make_synthetic

        path = tmp_path / "synth.h5ad"
        make_synthetic(path, 500, 2_000, seed=0)
        f = dataset_facts(str(path))
        assert f["n_cells"] == 500 and f["n_genes"] == 2_000
        density = f["nnz"] / (f["n_cells"] * f["n_genes"])
        assert SYNTHETIC_DENSITY * 0.5 < density < SYNTHETIC_DENSITY * 2.0, density

    def test_generated_data_is_deterministic(self, tmp_path: Path) -> None:
        pytest.importorskip("anndata")
        from telemetry.arms.base import dataset_facts
        from validate import make_synthetic

        a, b = tmp_path / "a.h5ad", tmp_path / "b.h5ad"
        make_synthetic(a, 300, 1_000, seed=7)
        make_synthetic(b, 300, 1_000, seed=7)
        assert dataset_facts(str(a))["nnz"] == dataset_facts(str(b))["nnz"]


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture missing")
class TestEndToEnd:
    def test_a_real_run_produces_a_usable_record(self, tmp_path: Path) -> None:
        from telemetry.runner import run_one

        rec = run_one(
            arm="polariseq-ram", path=str(FIXTURE), label="fixture",
            params=Params(n_hvg=100, n_pcs=10, with_umap=False, min_genes=0, threads=2),
            results_dir=tmp_path, timeout_s=600,
        )
        assert rec.status == "ok", rec.note
        assert rec.wall_s > 0
        assert rec.peaks["parent_rss"] > 0
        assert rec.peaks["child_uss"] > 0
        names = {s["name"] for s in rec.steps}
        assert {"load", "qc", "hvg", "pca", "neighbors", "leiden"} <= names
        # Artifacts are what make the equivalence check possible; a run
        # without them is a stopwatch reading, not a benchmark.
        for key in ("pca", "labels", "hvg", "obs_names"):
            assert Path(rec.outputs["artifacts"][key]).exists(), key
        assert quality(rec.to_json())["ok"]

    def test_polariseq_kernels_release_the_gil(self) -> None:
        """A Python thread must keep running while a Rust kernel executes.

        This asserts the property **directly**. An earlier version asserted it
        through the in-process sampler's blind fraction, which was a proxy and
        a bad one: blindness is dominated by whether the sampler thread gets
        *scheduled*, not by whether the GIL is held. Measured 2026-09-08 on
        brain50k with identical code — 23.9 % blind at 8 threads on 8 cores,
        1.7 % at the library default, and 8.9 % at 2 threads merely because the
        machine was otherwise loaded. The proxy therefore failed on a busy
        laptop and blamed the bindings, which is how it sent one investigation
        looking for a GIL bug that was not there.

        A counter thread distinguishes the two cleanly. Under a held GIL it
        freezes completely. Under load it merely counts more slowly. So a
        generous floor catches the real regression and tolerates a busy
        machine.
        """
        import threading

        import numpy as np
        import scipy.sparse as sp

        import polariseq as ps

        rng = np.random.default_rng(0)
        # Large enough that PCA takes well over a scheduling quantum, small
        # enough to stay quick in CI.
        X = sp.random(6000, 1200, density=0.06, format="csr",
                      random_state=0, dtype=np.float32)
        X.data = np.abs(X.data) * 10
        adata = ps.AnnData(X=X)
        ps.pp.normalize_total(adata, target_sum=1e4)
        ps.pp.log1p(adata)

        ticks = 0
        stop = threading.Event()

        def spin() -> None:
            nonlocal ticks
            while not stop.is_set():
                ticks += 1

        t = threading.Thread(target=spin, daemon=True)
        t.start()
        try:
            time.sleep(0.05)  # let the thread get going
            before = ticks
            t0 = time.perf_counter()
            ps.pp.pca(adata, n_components=50, seed=0)
            kernel_s = time.perf_counter() - t0
            advanced = ticks - before
        finally:
            stop.set()
            t.join(timeout=2.0)

        assert kernel_s > 0.05, (
            f"PCA took only {kernel_s * 1000:.0f} ms — too short to prove "
            "anything about the GIL; enlarge the fixture"
        )
        # A held GIL gives ~0. Anything in the thousands means the interpreter
        # was available to other threads throughout the kernel.
        assert advanced > 1000, (
            f"a Python thread advanced only {advanced} times during a "
            f"{kernel_s * 1000:.0f} ms PCA — a kernel is holding the GIL "
            "(see crates/polariseq-python/src/lib.rs, which should wrap every "
            "multi-millisecond kernel in py.detach)"
        )


class TestGateRequirement:
    """The gate must not ask for memory the machine never has.

    Raising the floor to a run's known peak is right on a machine that can
    reach it. On one that cannot — a Windows laptop idling at 30-45 per cent
    used — waiting cannot conjure the memory, so the run would stall for the
    whole timeout and then start anyway.
    """

    @staticmethod
    def _effective(need: int, band):
        """The runner's rule, in the form the runner applies it."""
        from dataclasses import replace

        ceiling = int(band.idle_available_bytes or need)
        capped = need > ceiling
        target = min(need, ceiling)
        if target > band.min_available_bytes:
            band = replace(band, min_available_bytes=target)
        return band, capped

    def test_floor_rises_to_a_reachable_requirement(self) -> None:
        from telemetry.loadgate import Band

        band = Band(min_available_bytes=3_000_000_000, idle_available_bytes=9_000_000_000)
        out, capped = self._effective(6_000_000_000, band)
        assert out.min_available_bytes == 6_000_000_000
        assert capped is False

    def test_floor_is_capped_at_what_the_machine_offers_at_rest(self) -> None:
        from telemetry.loadgate import Band

        band = Band(min_available_bytes=3_000_000_000, idle_available_bytes=9_000_000_000)
        out, capped = self._effective(11_900_000_000, band)
        assert out.min_available_bytes == 9_000_000_000, "waiting cannot exceed idle"
        assert capped is True

    def test_uncalibrated_band_keeps_the_old_behaviour(self) -> None:
        from telemetry.loadgate import Band

        band = Band(min_available_bytes=1_000_000_000)      # idle_available unknown
        out, capped = self._effective(4_000_000_000, band)
        assert out.min_available_bytes == 4_000_000_000
        assert capped is False

    def test_calibrate_records_what_was_free_at_rest(self) -> None:
        from telemetry.loadgate import calibrate

        band = calibrate(samples=2, gap_s=0.0, verbose=False)
        assert band.idle_available_bytes > 0
        assert band.min_available_bytes <= band.idle_available_bytes


def test_ari_matches_scikit_learn_at_atlas_size():
    """The int64 pair-count product used to wrap above ~400K cells."""
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    a = rng.integers(0, 27, 2_000_000)
    b = np.where(rng.random(a.size) < 0.8, a, rng.integers(0, 27, a.size))
    assert adjusted_rand_index(a, b) == pytest.approx(sklearn.adjusted_rand_score(a, b), abs=1e-9)
