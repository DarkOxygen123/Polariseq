"""Resource detection, budgets, and the cost model behind `ps.plan`.

The model exists to answer "will this run on this machine, and if not, what
is the limit?". These tests check the two things that makes it trustworthy:
its internal consistency (bigger data costs more, spilling costs less than
holding), and that it tracks a *measured* peak rather than drifting into
fiction.
"""

from __future__ import annotations

import subprocess
import sys

import polariseq as ps
import pytest
from polariseq.planning import estimate, max_cells
from polariseq.resources import Budget, detect_resources

FIXTURE = "tests/data/medium_10kx2k.h5ad"

_SOURCES = {"psutil", "meminfo", "sysconf", "vm_stat", "estimated"}


def budget(ram_gb: float = 8.0, threads: int = 4, **kw) -> Budget:
    return Budget(
        ram_bytes=int(ram_gb * 1e9),
        threads=threads,
        spill_dir="/tmp/polariseq_test_spill",
        spill_bytes=int(500e9),
        **kw,
    )


class TestDetection:
    def test_reports_plausible_hardware(self) -> None:
        r = detect_resources()
        assert r.total_ram_bytes > 0
        assert 0 < r.available_ram_bytes <= r.total_ram_bytes
        assert r.available_source in _SOURCES
        assert r.logical_cores >= 1
        assert 1 <= r.performance_cores <= r.logical_cores
        assert r.spill_free_bytes >= 0
        assert r.platform and r.machine
        assert "GB" in repr(r)

    def test_summary_is_serialisable(self) -> None:
        s = detect_resources().summary()
        assert s["total_ram_gb"] > 0
        assert s["available_source"] in _SOURCES


class TestBudget:
    def test_detect_stays_within_the_machine(self) -> None:
        b = Budget.detect()
        assert b.ram_bytes > 0
        assert b.detected is not None
        assert b.ram_bytes <= b.detected.available_ram_bytes
        assert b.threads >= 1
        # Approximation is never on unless asked for.
        assert b.allow_lossy is False

    def test_overrides_replace_only_what_is_given(self) -> None:
        b = Budget.detect()
        c = b.with_overrides(ram_gb=4, allow_lossy=True)
        assert c.ram_bytes == 4_000_000_000
        assert c.allow_lossy is True
        assert c.threads == b.threads and c.spill_dir == b.spill_dir

    def test_the_ssd_ceiling_follows_the_spill_folder(self, tmp_path) -> None:
        """Moving the spill folder, as ps.project does, measures the disk it is
        on, unless a ceiling is given with it."""
        from polariseq.resources import free_disk_bytes

        b = Budget.detect()
        moved = b.with_overrides(spill_dir=str(tmp_path))
        assert abs(moved.spill_bytes - 0.9 * free_disk_bytes(str(tmp_path))) < 1e9
        assert b.with_overrides(spill_dir=str(tmp_path), spill_gb=5).spill_bytes == 5_000_000_000

    def test_public_api_round_trip(self, monkeypatch) -> None:
        # The budget object only: the worker pool is per process and cannot
        # be resized, so letting this test size it would set the thread
        # count of every test after it (see the next test for the pool).
        monkeypatch.setattr(ps, "_apply_threads", lambda b: None)
        before = ps.budget()
        try:
            b = ps.set_budget(ram_gb=3.5, threads=2, allow_lossy=True)
            assert ps.budget() is b
            assert b.ram_gb == pytest.approx(3.5)
            assert b.threads == 2 and b.allow_lossy is True
            # The deprecated alias still works.
            assert ps.set_resources(max_ram_gb=2.0).ram_gb == pytest.approx(2.0)
        finally:
            ps._BUDGET = before


    def test_set_budget_sizes_the_worker_pool(self) -> None:
        """The pool gets the thread count asked for, not the detected one.

        Measured 2026-09-19: `set_budget(threads=4)` left the pool at the
        detected count (12 on a laptop, 48 on a cluster node) because
        detecting the budget sized the pool first, so the cross-platform
        campaign's pinned thread count never reached Polariseq. A fresh
        process, because the pool cannot be resized once built.
        """
        code = ("import polariseq as ps; from polariseq import _polariseq as c; "
                "ps.set_budget(threads=2); ps.set_budget(threads=2, ram_gb=3); "
                "print(c.thread_count())")
        out = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", code],
                             capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "2"

class TestCostModel:
    def test_cost_grows_with_the_dataset(self) -> None:
        b = budget()
        small = estimate(50_000, 20_000, 50_000 * 1_000, b, storage="ram")
        big = estimate(500_000, 20_000, 500_000 * 1_000, b, storage="ram")
        assert big.peak_bytes > small.peak_bytes

    def test_storage_modes_are_ordered(self) -> None:
        """Compressing and spilling both cost less than holding the matrix.

        Spilling is not always below compressing: the gene-selection pass out
        of core holds about 1.6 blocks' worth per block in flight (measured on
        scTab shards), and the packed in-memory mode has no user path and no
        measurement to order it against."""
        # A matrix much larger than its blocks (32 GB held in memory): there
        # spilling wins by a wide margin. On a small matrix it need not; the
        # gene-selection blocks can cost about what the matrix does.
        b = budget(ram_gb=64)
        shape = (2_000_000, 30_000, 2_000_000 * 2_000)
        peaks = {
            mode: estimate(*shape, b, storage=mode).peak_bytes
            for mode in ("ram", "compressed", "spill")
        }
        assert peaks["ram"] > peaks["compressed"]
        assert peaks["ram"] > 2 * peaks["spill"]

    def test_block_size_shrinks_with_the_budget(self) -> None:
        shape = (200_000, 30_000, 200_000 * 2_000)
        big = estimate(*shape, budget(ram_gb=64), storage="spill")
        small = estimate(*shape, budget(ram_gb=1), storage="spill")
        assert big.rows_per_block > small.rows_per_block

    def test_spill_has_a_working_set_floor(self) -> None:
        """Sub-floor budgets are refused, not shrunk into.

        The spill working set is floor-bound: file residency plus the
        process plus one block in flight does not shrink with the budget.
        Calibrated on the measured 2026-09-10/11 budget sweep at 800K cells
        (1.6e9 nnz): 1–3 GB requests all demanded more than they were given
        (3.2–5.1 GB) while 4 GB and above adhered, so the floor sits just
        above 3 GB and the model must refuse below it.
        """
        shape = (800_000, 27_998, 1_607_496_131)
        p3 = estimate(*shape, budget(ram_gb=3), storage="spill", with_umap=False)
        assert not p3.fits
        assert p3.min_budget_gb == pytest.approx(3.13, abs=0.15)
        assert any("working-set floor" in w for w in p3.warnings)
        p4 = estimate(*shape, budget(ram_gb=4), storage="spill", with_umap=False)
        assert p4.fits
        # Measured 3.22 GB on the 800K sweep; planned higher since the
        # gene-selection term (HVG_INFLIGHT_FACTOR, from the scTab runs) was
        # added: over, the safe side, and still inside the 4 GB asked for.
        assert p4.peak_gb == pytest.approx(3.94, abs=0.15)
        # Block sizing is untouched by the floor: 6 GB still picks 12,288
        # rows on this density, the size the measured ladder ran at.
        p6 = estimate(*shape, budget(ram_gb=6), storage="spill", with_umap=False)
        assert p6.rows_per_block == 12_288

    def test_floor_scales_with_the_dataset(self) -> None:
        """A smaller file has a lower floor; the cap keeps it bounded."""
        small = estimate(400_000, 27_998, 803_920_843, budget(ram_gb=8),
                         storage="spill", with_umap=False)
        big = estimate(4_000_000, 27_998, 8_037_480_655, budget(ram_gb=8),
                       storage="spill", with_umap=False)
        assert small.min_budget_gb == pytest.approx(2.0, abs=0.2)
        # Residency saturates at the cap, so the floor stops growing with nnz.
        assert big.min_budget_gb < 4.0

    def test_ram_mode_has_no_floor(self) -> None:
        """In-memory modes have no budget-derived sizing to refuse."""
        p = estimate(10_000, 30_000, 20_000_000, budget(ram_gb=8), storage="ram")
        assert p.min_budget_bytes == p.peak_bytes
        assert "floor" not in p.report()

    def test_summary_carries_the_floor(self) -> None:
        p = estimate(800_000, 27_998, 1_607_496_131, budget(ram_gb=6),
                     storage="spill", with_umap=False)
        s = p.summary()
        assert s["min_budget_gb"] == pytest.approx(3.1, abs=0.2)
        assert "min_budget_gb" in s

    def test_max_cells_tracks_the_budget_and_the_mode(self) -> None:
        assert max_cells(budget(ram_gb=16)) > max_cells(budget(ram_gb=4))
        # Spilling is what makes very large datasets reachable at all.
        assert max_cells(budget(), storage="spill") > 5 * max_cells(
            budget(), storage="ram"
        )

    def test_reports_when_it_does_not_fit(self) -> None:
        p = estimate(2_000_000, 30_000, 2_000_000 * 2_000, budget(ram_gb=1), storage="ram")
        assert not p.fits
        assert p.warnings
        assert "spill" in " ".join(p.warnings)

    def test_graph_stages_are_called_out_as_storage_independent(self) -> None:
        """Past a certain scale the matrix is not the problem — say so."""
        p = estimate(20_000_000, 30_000, 20_000_000 * 500, budget(ram_gb=8), storage="spill")
        assert p.limiting_step in {"neighbors", "leiden", "umap"}
        assert "not help" in p.report()


class TestAgainstMeasurement:
    def test_plan_matches_a_real_run(self) -> None:
        """The model must track a measured peak, not just be self-consistent.

        Bound is deliberately loose (within 2x): peak RSS includes allocator
        behaviour nobody models exactly. The tight agreement lives in the
        handbook's scaling chapter, where it is measured on real atlases;
        this test is here to catch the model drifting away from reality.
        """
        # getrusage is POSIX-only; on Windows the rest of this module runs.
        resource = pytest.importorskip("resource")
        before = ps.budget()
        try:
            ps.set_budget(ram_gb=8, threads=4)
            predicted = ps.plan(
                FIXTURE, n_hvg=500, n_pcs=20, storage="ram", with_umap=False, verbose=False
            )

            base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            adata = ps.read_h5ad(FIXTURE)
            ps.pp.calculate_qc_metrics(adata)
            ps.pp.normalize_total(adata, target_sum=1e4)
            ps.pp.log1p(adata)
            ps.pp.highly_variable_genes(adata, n_top_genes=500, subset=True)
            ps.pp.pca(adata, 20, seed=0)
            ps.pp.neighbors(adata, n_neighbors=15)
            measured = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - base

            # ru_maxrss is a process high-water mark, so a later test having
            # already allocated more makes `measured` meaningless. Only assert
            # when this run actually moved the mark.
            if measured > 50_000_000:
                ratio = predicted.peak_bytes / measured
                assert 0.4 < ratio < 2.5, (
                    f"predicted {predicted.peak_gb:.3f} GB vs measured "
                    f"{measured / 1e9:.3f} GB (ratio {ratio:.2f})"
                )
            assert predicted.limiting_step in {
                "load", "qc", "hvg", "hvg subset", "pca", "neighbors",
            }
        finally:
            ps._BUDGET = before


class TestPlanApi:
    def test_plan_reads_only_metadata(self) -> None:
        p = ps.plan(FIXTURE, verbose=False)
        assert p.n_cells == 10_000 and p.n_genes == 2_000
        assert p.nnz > 0
        assert p.storage in {"ram", "compressed", "spill"}
        assert isinstance(p.fits, bool)
        report = p.report()
        assert "cells" in report and "peak" in report
        assert set(p.summary()) >= {"peak_gb", "fits", "limiting_step", "storage"}

    def test_max_cells_exposed_on_the_package(self) -> None:
        before = ps.budget()
        try:
            ps.set_budget(ram_gb=8)
            assert ps.max_cells(storage="spill") > ps.max_cells(storage="ram") > 0
        finally:
            ps._BUDGET = before


def test_recommendation_is_a_mode_the_api_runs() -> None:
    """Auto-choice names only modes a user can run: in memory or out of core."""
    from polariseq.planning import estimate
    from polariseq.resources import Budget

    b = Budget.detect().with_overrides(ram_gb=6, threads=4)
    for n_cells in (10_000, 300_000, 800_000, 4_000_000):
        p = estimate(n_cells, 30_000, n_cells * 1_500, b)
        assert p.storage in {"ram", "spill"}, (n_cells, p.storage)


class TestMeasuredLargeRuns:
    """The plan against runs measured on scTab shards (2026-10-02/03). Each
    case is the measured number; the tolerance is the model's honesty."""

    def test_files_merged_in_memory_hold_the_matrix_twice(self) -> None:
        # 2 files on a workstation: 13.39 GB held; the plan said 7.71 GB.
        p2 = estimate(443_814, 19_331, 825_266_640, budget(40.0, threads=24),
                      storage="ram", n_files=2)
        assert p2.limiting_step == "load"
        assert 0.85 <= p2.peak_gb / 13.39 <= 1.25
        # 10 files on a cluster node: 66.41 GB held; the plan said 37.82 GB.
        p10 = estimate(2_219_070, 19_331, 4_110_559_436, budget(100.0, threads=8),
                       storage="ram", n_files=10)
        assert 0.85 <= p10.peak_gb / 66.41 <= 1.25
        # One file is not merged, so it does not pay for a second copy.
        p1 = estimate(2_219_070, 19_331, 4_110_559_436, budget(100.0, threads=8),
                      storage="ram")
        assert p1.peak_gb < p10.peak_gb

    def test_the_project_folder_is_planned(self, monkeypatch) -> None:
        # 22,190,622 cells out of core, written plain: the copy measured
        # 369.7 GB and the results 11.0 GB at the end of the run.
        monkeypatch.setenv("POLARISEQ_STORE", "plain")
        p = estimate(22_190_622, 19_331, 46_187_527_591, budget(100.0, threads=4),
                     storage="spill")
        assert set(p.disk) == {"copy", "pca_copy", "results"}
        assert abs(p.disk["copy"] / 1e9 - 369.7) / 369.7 < 0.01
        assert 0.8 <= p.disk["results"] / 1e9 / 11.0 <= 1.25
        assert p.disk["pca_copy"] < p.disk["copy"], "PCA copies the selected genes only"
        assert p.disk_peak_bytes == sum(p.disk.values())
        assert "project folder at most" in p.report()
        # In memory there is no copy, only the results.
        assert set(estimate(10_000, 2_000, 1_700_000, budget(), storage="ram").disk) == {"results"}
        # Compact (the default): planned at 2 bytes a value, well under the plain copy.
        monkeypatch.delenv("POLARISEQ_STORE")
        c = estimate(22_190_622, 19_331, 46_187_527_591, budget(100.0, threads=4),
                     storage="spill")
        assert c.disk["copy"] < p.disk["copy"] / 3


class TestThreadsAndGeneSelection:
    """Out of core, one block per thread is in flight, and the gene-selection
    pass costs about 1.6 blocks' worth per block (measured on scTab shards)."""

    def test_the_out_of_core_peak_tracks_what_was_measured(self) -> None:
        two = estimate(443_814, 19_331, 825_266_640, budget(100.0, threads=4), storage="spill")
        ten = estimate(2_219_070, 19_331, 4_110_559_436, budget(100.0, threads=4), storage="spill")
        hvg2 = next(s for s in two.steps if s.name == "hvg").peak / 1e9
        assert 0.9 <= hvg2 / 7.28 <= 1.25
        assert 0.9 <= ten.peak_gb / 8.25 <= 1.25

    def test_more_threads_plan_more_in_flight_never_less(self) -> None:
        few = estimate(443_814, 19_331, 825_266_640, budget(40.0, threads=4), storage="spill")
        many = estimate(443_814, 19_331, 825_266_640, budget(40.0, threads=24), storage="spill")
        assert many.peak_gb >= few.peak_gb
        # 24 threads measured 14.71 GB at the gene selection: planned at or above it.
        hvg = next(s for s in many.steps if s.name == "hvg").peak / 1e9
        assert 1.0 <= hvg / 14.71 <= 1.3


class TestDensityAndPerCellTerms:
    """The 22M-cell run (8 threads, 2026-10-03) peaked at 39.16 GB where the plan
    said 29.97 GB. Gene selection holds a block per thread, the blocks differ
    widely in density between source datasets (the densest file holds 4,857
    values per cell against an average of 2,081), and PCA and UMAP cost a fixed
    number of bytes per cell, not the smaller figures first planned."""

    SHAPE = (22_190_622, 19_331, 46_187_527_591)

    @staticmethod
    def step(plan, name: str) -> float:
        return next(s for s in plan.steps if s.name == name).peak / 1e9

    def test_the_densest_file_sets_the_gene_selection_peak(self) -> None:
        avg = estimate(*self.SHAPE, budget(100.0, threads=8), storage="spill", n_files=100)
        dense = estimate(*self.SHAPE, budget(100.0, threads=8), storage="spill", n_files=100,
                         densest_nnz_per_row=4_857)
        assert self.step(avg, "hvg") < 20.0, "at the average density, the figure first planned"
        assert 0.97 <= self.step(dense, "hvg") / 39.16 <= 1.05
        assert dense.limiting_step == "hvg"
        assert 0.97 <= dense.peak_gb / 39.16 <= 1.05
        assert "densest file" in dense.report()
        # The same headroom reproduces the 4-thread run's gene-selection step.
        four = estimate(*self.SHAPE, budget(100.0, threads=4), storage="spill", n_files=100,
                        densest_nnz_per_row=4_857)
        assert 0.97 <= self.step(four, "hvg") / 21.16 <= 1.05

    def test_a_density_well_below_the_average_changes_nothing(self) -> None:
        # The headroom stands for blocks that are denser than their file's
        # average, so a density below average / 1.2 cannot raise the charge.
        b = budget(100.0, threads=8)
        base = estimate(*self.SHAPE, b, storage="spill", n_files=100)
        for given in (None, 500, 1_700):
            same = estimate(*self.SHAPE, b, storage="spill", n_files=100,
                            densest_nnz_per_row=given)
            assert same.summary()["steps"] == base.summary()["steps"]
        # In memory there are no blocks in flight to charge.
        ram = estimate(2_219_070, 19_331, 4_110_559_436, b, storage="ram", n_files=10)
        dense_ram = estimate(2_219_070, 19_331, 4_110_559_436, b, storage="ram", n_files=10,
                             densest_nnz_per_row=9_000)
        assert ram.summary()["steps"] == dense_ram.summary()["steps"]

    def test_the_block_size_does_not_depend_on_the_density_given(self) -> None:
        # The block size sets the summation order, so it follows the budget
        # alone; the density only changes what the plan predicts.
        a = estimate(*self.SHAPE, budget(30.0, threads=8), storage="spill", n_files=100)
        b = estimate(*self.SHAPE, budget(30.0, threads=8), storage="spill", n_files=100,
                     densest_nnz_per_row=4_857)
        assert a.rows_per_block == b.rows_per_block

    def test_pca_and_umap_cost_a_fixed_number_of_bytes_per_cell(self) -> None:
        big = estimate(*self.SHAPE, budget(100.0, threads=8), storage="spill", n_files=100)
        assert 0.95 <= self.step(big, "pca") / 28.39 <= 1.05
        assert 0.95 <= self.step(big, "umap") / 29.39 <= 1.10
        ten = estimate(2_219_070, 19_331, 4_110_559_436, budget(100.0, threads=4),
                       storage="spill", n_files=10)
        assert 0.95 <= self.step(ten, "pca") / 2.81 <= 1.20
        assert 0.95 <= self.step(ten, "umap") / 3.13 <= 1.10
        # Almost proportional to the cells (the covariance part is fixed), as
        # the two measurements were: 10.0 times the cells, 10.1 times the memory.
        ratio = (self.step(big, "pca") - 0.15) / (self.step(ten, "pca") - 0.15)
        assert 0.85 <= ratio / (22_190_622 / 2_219_070) <= 1.05

    def test_where_blocks_are_even_the_headroom_errs_high_not_low(self) -> None:
        # The first shards are uniform (densest file 2,076 against an average
        # of 1,852 values per cell), and there the average already matched
        # (8.19 planned, 8.21 measured). The headroom costs them about a
        # quarter more than measured, the safe side: a declined run is
        # recoverable, a swapping one is not.
        ten = estimate(2_219_070, 19_331, 4_110_559_436, budget(100.0, threads=4),
                       storage="spill", n_files=10, densest_nnz_per_row=2_076)
        assert 1.0 <= ten.peak_gb / 8.25 <= 1.35
        two = estimate(443_814, 19_331, 825_266_640, budget(100.0, threads=4),
                       storage="spill", n_files=2, densest_nnz_per_row=2_026)
        assert 1.0 <= two.peak_gb / 7.28 <= 1.35
