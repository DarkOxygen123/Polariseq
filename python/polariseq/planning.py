"""Predict what a run will cost, and choose a plan that fits the budget.

Peak memory in this pipeline is not mysterious: every structure is an array
whose size follows from ``(n_cells, n_genes, nnz, n_hvg, n_pcs, n_neighbors,
threads)``. This module writes those sizes down, adds them up per step, and
compares the total to a :class:`~polariseq.resources.Budget`. That gives two
useful things before a single byte is read:

- **A prediction.** ``ps.plan(path)`` reports the peak of each step and where
  the run would blow up.
- **A choice.** The same numbers pick the storage mode (in RAM or spilled to
  SSD), which is the difference between a dataset running and not running
  on a given machine.

Every coefficient below is annotated with where it comes from. Several were
measured rather than derived, and those say so — a cost model with invented
constants is worse than no model, because it is believed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from polariseq.resources import Budget

__all__ = ["StepCost", "Plan", "estimate", "max_cells", "store_bytes", "loads_inputs",
           "neighbors_working_bytes", "leiden_working_bytes", "umap_working_bytes"]

_GB = 1_000_000_000

#: Steps whose working memory is O(cells) whatever the storage mode. When one
#: of these limits a plan, changing storage will not help.
_GRAPH_STEPS = frozenset({"neighbors", "leiden", "umap"})
#: Share of the memory free at the moment a graph step may fill with its
#: inputs and working memory when it reads its inputs into memory (see
#: :func:`loads_inputs`): the rest is left to the machine, whose other
#: programs (a shared cluster node's other jobs) the budget knows nothing of.
LIVE_FREE_FRACTION = 0.8

# ── storage coefficients ────────────────────────────────────────────────
#: CSR in RAM: f32 value + u32 column index per nonzero, u32 row pointer.
CSR_BYTES_PER_NNZ = 8
CSR_BYTES_PER_ROW = 4
#: The out-of-core block store: the same per nonzero, but a u64 row pointer,
#: which the store also keeps in memory for every block lookup.
STORE_BYTES_PER_ROW = 8

#: Bytes per nonzero *in flight* on the spill path — which is not the 8 a
#: resident CSR costs, because more than one buffer is alive per block.
#: Counted from the implementation rather than guessed
#: (`FileCsr::for_each_block` -> `read_rows`, then `NormalizedView`):
#:
#:   during the read     `bytes` staging (4) + `data` (4) + `indices` (4)
#:   during the compute  `data` (4) + `indices` (4) + the view's copy (4)
#:
#: Both phases come to 12. `NormalizedView::for_each_block` does
#: `blk.data.to_vec()` — one transformed copy per in-flight block — which is
#: what makes the compute phase as expensive as the read.
SPILL_BYTES_PER_NNZ_IN_FLIGHT = 12

#: Empirical margin on the in-flight term, covering what the byte count
#: above cannot: memory freed by one batch of blocks is not necessarily
#: returned to the OS before the next batch allocates, so the observed peak
#: sits above the sum of what is live at any single instant.
#:
#: Calibrated against the measured ladder, not chosen to flatter: with the
#: mechanical 12 B/nnz alone the prediction lands at 0.79–0.96x of measured
#: on the spill arm, i.e. *under*, which is the wrong direction for a
#: budget. This lifts it to bracket from above on every dataset measured.
#:
#: Treat it as what it is — a fudge factor with a reason and a measurement,
#: not a derived constant. The residual it covers is also noisy: heart_50k's
#: spill peak was 0.856 / 0.989 / 1.040 GB across three runs of identical
#: code, so any calibration finer than ~10% is fitting to noise.
SPILL_INFLIGHT_SAFETY = 1.25

#: File-backed residency a spill run occupies: the pages of the spill files
#: the kernel keeps mapped while a streaming pass touches them. They are not
#: the process's anonymous memory — USS does not see them — but they occupy
#: the machine all the same, and a budget that ignores them is a budget the
#: run silently exceeds.
#:
#: Measured as (RSS − USS) at the streaming peak on the 2026-09-10/11 budget
#: sweep and the 1,306,127-cell run: 0.4–1.5 GB at 400K–800K cells
#: (0.8–1.6e9 nnz) and 1.85 GB at 1.3M cells (2.6e9 nnz), i.e. ≈1.7 B/nnz
#: with the window flattening near 2.2 GB once the file is several times
#: larger than the kernel keeps warm. Calibrated on measured sweeps, not derived
#: (the sweep records are on the repository's ``reproduce`` branch).
#:
#: This term is what makes the working set have a *floor*. Block sizes
#: shrink with the budget, but the residency does not, so below
#: ``floor / (1 − BLOCK_BUDGET_FRACTION)`` no block size satisfies the
#: budget — and asking for less only makes the run churn harder (measured:
#: a 1 GB request on the 800K file demanded 5.12 GB where 4–8 GB requests
#: stayed inside what they asked for).
SPILL_RESIDENCY_BYTES_PER_NNZ = 1.7
#: Where the residency window stops growing with the file.
SPILL_RESIDENCY_CAP_BYTES = 2_200_000_000
#: `CompressedCsr`: u16 delta index + u8 value per nonzero, u64 row pointer.
#: Measured at 3.0 B/nnz on a u8 fixture (phase 6 step 3 asserts <= 3.1).
COMPRESSED_BYTES_PER_NNZ = 3
COMPRESSED_BYTES_PER_ROW = 8

# ── model coefficients ──────────────────────────────────────────────────
#: Highly-variable genes carry more nonzeros than their share of genes: they
#: are selected partly for being expressed. Measured on heart_100k — 2 000 of
#: ~30 000 genes (6.7 %) kept 9.3 % of the nonzeros, so ~1.4x. Only an
#: estimate; a file whose HVGs are unusually sparse will use less.
HVG_NNZ_ENRICHMENT = 1.4
#: Entries of the symmetric connectivity CSR per cell per neighbour: the
#: fuzzy set stores both directions of every edge, mutual pairs once.
#: Measured 1.75 on the 4.06M-cell fetal atlas (105M entries at k = 15).
FUZZY_ENTRY_FACTOR = 1.75
#: NN-Descent's peak memory per cell, above the embedding, at 15 neighbours:
#: a part that does not depend on the threads and one per thread,
#: ``NNDESCENT_BYTES_PER_CELL + NNDESCENT_BYTES_PER_CELL_PER_THREAD * threads``.
#:
#: Fitted on real data, not derived: `pp.neighbors` (0.2.0) on random slices
#: of the 95.6M-cell Tahoe-100M embedding on a compute node
#: (`benchmarks/relicence/neighbors_memory.py` on the reproduce branch), anonymous memory less the
#: embedding: 20M cells at 4, 8, 12 and 16 threads 669, 673, 692 and 676 bytes
#: per cell (the line through them 667 + 1.0 per thread), 40M at 16 threads
#: 668, all 95.6M at 16 threads 655 (the embedding read in place there). Set a
#: little above the line, so it plans at or above every measurement: 692 at
#: 12 threads on 20M, 3 to 7% above the others. Raised to 780 on 2026-10-15:
#: in the full analyses of the 1.3M-cell mouse brain the step held 1.72 GB on
#: Windows (planned 1.65 at 680) and on the 4.0M-cell fetal atlas 4.77 GB, so
#: it is set 3% above the higher of the two; on the bare runs above it plans
#: 9 to 17% high.
#:
#: Until 0.2.0 the forest's construction set the peak, 548 + 68 per thread
#: (1,636 bytes per cell at 16 threads, 156 GB at 95.6M cells): each thread
#: copied the rows of the subtree it split, and random hyperplanes split
#: unevenly, so a copy could hold half the cells. Subtrees above 2^20 points
#: are now split in place (`nndescent.rs`, ``GATHER_MAX``), the same graph bit
#: for bit, and the threads hardly matter.
NNDESCENT_BYTES_PER_CELL = 780
NNDESCENT_BYTES_PER_CELL_PER_THREAD = 1
#: Neighbours the fit was made at. The heaps and candidate lists grow with
#: the number of neighbours, so more is charged in proportion; fewer is not
#: credited, the forest part not depending on it.
NNDESCENT_FIT_NEIGHBORS = 15
#: Leiden's working memory per connectivity entry: the copy of the graph the
#: core owns (8 B), vertex and community strengths, and the first aggregated
#: graph. Measured 31 B from the clustering step's peak on the same atlas
#: (after 2026-09-28, when the core stopped rebuilding and cloning the graph;
#: it was ~66 B before). Raised to 33 on 2026-10-15, 3% above 0.2.0's
#: largest (the 4.0M-cell fetal atlas in example 06 on Windows: 5.33 GB).
LEIDEN_BYTES_PER_ENTRY = 33
#: faer's dense symmetric eigendecomposition allocates several workspaces on
#: top of the covariance matrix it is given. Measured on heart_10k by timing
#: PCA alone at a few HVG counts: 6.2x the Gram matrix at n_hvg=1000, 5.7x at
#: 2000, 4.0x at 3000 (the spread is high-water-mark noise at small sizes).
#: The largest is used deliberately — for a budget, over-predicting memory
#: means declining a run that would have fit, while under-predicting means
#: swapping, which is far worse. Raised to 8 on 2026-10-15: the run's own
#: resident memory on PBMC 3k (2,700 cells, where PCA's fixed part is all
#: there is) rose by 0.28 GB in PCA, the Gram matrix with its partial sum
#: (2 x 32 MB) and about 8 x 32 MB for the eigendecomposition, its
#: eigenvectors and workspace.
GRAM_EIGEN_WORKSPACE_FACTOR = 8.0
#: How the Gram matrix of exact PCA is summed (`rsvd::gram_matrix_blocks`,
#: chunks of `rsvd::chunk_rows`: max(n / 32, 32,768) rows): one n_hvg^2 f64
#: partial per chunk, all kept until they are merged in order, so up to 33
#: matrices of 32 MB at 2,000 genes (1.06 GB) above a million cells. Missing
#: from the model until 2026-10-15: PCA on the 1.3M-cell mouse brain measured
#: 3.05 GB on Windows and 2.79 GB on the Mac, against 1.97 GB planned.
GRAM_CHUNKS_MAX = 32
GRAM_CHUNK_MIN_ROWS = 32_768
#: Blocks read ahead by `H5adStream` (its `PREFETCH`).
STREAM_PREFETCH = 4
#: The gene-statistics pass of HVG selection transforms each block (scaled,
#: then the statistics' squares) on top of the read, so a block in flight
#: costs about 1.6 times what a plain pass costs. Measured on scTab shards at
#: 4 threads: 7.28 GB at 0.44M cells and 8.25 GB at 2.2M cells for this step,
#: against 5.2 and 6.0 GB planned without it.
HVG_INFLIGHT_FACTOR = 1.6
#: Blocks are cut by rows, but cells differ widely in density between source
#: datasets, so the blocks in flight together can hold far more than the
#: average. On scTab (100 files, 22.19M cells) values per cell run from 769
#: to 4,857 between files and the 678 blocks of 32,768 cells from 576 to
#: 6,146 (average 2,081); the densest 8 blocks hold 2.7 times the values of 8
#: average ones. Gene selection, which works on `threads` blocks at once, then
#: peaked at 39.2 GB at 8 threads and 21.2 GB at 4, where the average density
#: planned 15.6 and about 9. The densest 8 blocks average 1.16 times the
#: densest file, so the planner charges the in-flight blocks at the densest
#: file's density times this headroom, when the files' shapes are known. With
#: it the plan is 39.2 and 20.8 GB, against those two measurements. One
#: dataset's worth of evidence, so a prediction, not a bound. Raised to 1.25 on
#: 2026-10-15: at 1.2 the 8-thread run (39.37 GB) was planned at 39.35 GB.
BLOCK_DENSITY_HEADROOM = 1.25
#: PCA's per-cell working memory: the scores and the buffers behind them,
#: about 25 bytes per cell per component. Measured 1.27 KB per cell at 50
#: components on 2.2M cells (2.81 GB) and 1.28 KB on 22.1M cells (28.4 GB),
#: against the 0.4 KB the planner had charged (an embedding and one projection).
#: Set to 26 on 2026-10-15, with the Gram matrix's partial sums now counted
#: apart (GRAM_CHUNKS_MAX): the 1.3M-cell mouse brain's PCA (3.05 GB on
#: Windows) is then planned 4% above, the 22.1M-cell atlas's 12% above.
PCA_BYTES_PER_CELL_PER_PC = 26
#: UMAP's working memory per connectivity entry (edge list, per-edge SGD
#: schedules, the layout's buffers, the multilevel start's arrays).
#:
#: Refitted for 0.2.0 (2026-10-14) from every run of the final build, with the
#: inputs counted as the planner now counts them (LOADED_INPUT_COPIES, the
#: embedding the multilevel start reads, OBS_BYTES_PER_CELL): the UMAP phase
#: on the 4.0M-cell fetal atlas implied 41 bytes per entry in example 06 on
#: Windows, 37 to 38 on a compute node at 4 and 8 threads, 35 in the Windows benchmark
#: and 31 on the Mac (inputs read in place); on the 1.3M-cell mouse brain 22
#: to 35. Set to 44, 3% above the largest of them. (The Mac's mouse-brain UMAP
#: phase, 3.1 to 3.6 GB, is memory macOS kept from gene selection, not
#: UMAP's.) Until 0.2.0: 31, from 2.2M and 22.1M cells with the inputs not
#: counted.
UMAP_BYTES_PER_ENTRY = 44
#: How many times resident memory holds a graph step's inputs when the step
#: reads them into memory: its copy, and the pages of the file the step before
#: wrote, which stay resident. Measured on Tahoe-100M's neighbours (0.2.0,
#: embedding read into memory): 99.5 GB resident against 81.8 GB anonymous,
#: the difference (17.7 GB) the embedding's file (19.1 GB). Those pages are
#: clean and the system can drop them, but they count in resident memory and
#: in Windows' working set, which is what the plan predicts.
LOADED_INPUT_COPIES = 2
#: What the run holds in Python for each cell while the graph steps work: the
#: cell names and the quality-control columns of ``obs``. The residual of
#: neighbours and Leiden over the plan in 0.2.0's full analyses: 0.2 to 0.3 GB
#: on the 4.0M-cell fetal atlas (Windows and Mac), 0.16 GB on the 1.3M-cell
#: mouse brain (Windows), about 60 bytes per cell.
OBS_BYTES_PER_CELL = 64
#: Share of the RAM budget the in-flight row blocks may occupy. They are pure
#: working space, so they must not crowd out the structures the step needs.
BLOCK_BUDGET_FRACTION = 0.25
#: Runtime overhead that does not scale with the data: numpy/scipy temporaries
#: in the Python glue, allocator fragmentation, and the extension's own
#: working set. Measured as the residual between predicted and actual peak RSS
#: on heart_10k and heart_100k (~0.15 GB on both, i.e. flat). Without it the
#: model is accurate at scale but under-predicts small runs by ~2x. Raised to
#: 0.30 GB on 2026-10-15, measured again the same way: the process holds 0.13 GB
#: once Polariseq, NumPy, SciPy and anndata are imported, and on the four
#: smallest datasets (2,700 to 23,837 cells, five runs each) the peaks varied
#: by up to 0.12 GB between runs of one dataset (PBMC 10k: 0.79 to 0.91 GB);
#: the constant is set so that every one of those runs is planned at least 3%
#: above its peak.
PROCESS_OVERHEAD_BYTES = 300_000_000
#: In memory, what reading the file holds above the matrix (the reader's
#: buffers and the cell names). Measured with 0.2.0 on a compute node: 0.42 GB on the
#: 1.3M-cell mouse brain, 0.34 GB on the 4.06M-cell fetal atlas; set at 0.8 GB.
IN_MEMORY_READ_BYTES = 800_000_000
#: In memory, quality control's per-cell results, built in Python and kept.
#: Measured with 0.2.0 on a compute node above the matrix and the read: 0.65 GB on
#: 1.3M cells (500 bytes per cell), 1.44 GB on 4.06M (355); set at 600.
QC_BYTES_PER_CELL = 600
#: In memory, bytes per stored value of the chunks the transform views write
#: while they are read, one chunk per thread (gene selection and the
#: renormalization over the selected genes). Measured with 0.2.0 on a compute node, in
#: memory at 4 threads: gene selection held 3.63 GB above the matrix on the
#: 1.3M-cell mouse brain and 2.68 GB on the 4.06M-cell fetal atlas, chunks of
#: about 82 million values each (11 and 8 bytes per value in flight); set
#: at 12. At 8 threads the fetal atlas held less (2.43 GB), so no more than
#: IN_MEMORY_VIEW_CHUNKS_MAX chunks are charged, however many threads run.
IN_MEMORY_VIEW_BYTES_PER_NNZ = 12
IN_MEMORY_VIEW_CHUNKS_MAX = 4
#: In memory, how many copies of the selected genes the run holds from PCA on.
#: Measured with 0.2.0 on a compute node: the graph steps held 0.7 to 1.05 GB more than
#: one copy and the process (4.06M fetal cells, one copy 0.93 GB; 1.3M brain
#: cells, one copy 2.1 GB).
IN_MEMORY_HVG_COPIES = 2
#: Bounds on the derived block size: small enough to bound memory, large
#: enough that per-block overhead and I/O syscalls stay negligible.
MIN_ROWS_PER_BLOCK = 1_024
MAX_ROWS_PER_BLOCK = 32_768


def _csr_bytes(nnz: int, n_rows: int) -> int:
    return CSR_BYTES_PER_NNZ * nnz + CSR_BYTES_PER_ROW * (n_rows + 1)


def neighbors_working_bytes(n_cells: int, n_neighbors: int, threads: int) -> int:
    """NN-Descent's working memory above the embedding (fitted, see
    ``NNDESCENT_BYTES_PER_CELL``), with the distance graph it returns."""
    per_cell = (NNDESCENT_BYTES_PER_CELL + NNDESCENT_BYTES_PER_CELL_PER_THREAD * max(1, threads)) * max(
        1.0, n_neighbors / NNDESCENT_FIT_NEIGHBORS
    )
    return int(n_cells * per_cell)


def leiden_working_bytes(entries: int) -> int:
    """Leiden's working memory above the connectivity graph it clusters."""
    return entries * LEIDEN_BYTES_PER_ENTRY


def umap_working_bytes(entries: int, n_cells: int) -> int:
    """UMAP's working memory above the connectivity graph: the edge list, the
    per-edge schedules, the layout's buffers, and the 2-D layout."""
    return entries * UMAP_BYTES_PER_ENTRY + n_cells * 8


def loads_inputs(held: int, inputs: int, working: int, budget_bytes: int,
                 available: int | None = None) -> bool:
    """Whether a graph step reads its inputs (the embedding, a graph) into
    memory before it starts, or reads them in place from the project's files.

    The steps read their inputs in random order. In memory that costs nothing;
    in place every page not in the cache is fetched on its own, which a local
    disk does in a fraction of a millisecond and a network file system (a
    cluster's scratch space) in a round trip each: Tahoe-100M's neighbours and
    Leiden ran at 0.03 to 0.2 cores for hours whenever the node dropped the
    cached pages. So the inputs are read into memory, in one sequential pass,
    whenever the run (``held``), the inputs and the step's working memory fit
    the budget, and, when ``available`` (memory free now) is given, the inputs
    and the working memory fit ``LIVE_FREE_FRACTION`` of it. Otherwise they
    are read in place and the budget is spared their size; from a network
    file system they are first copied to a local disk when one has room (see
    ``_results.graph_inputs``)."""
    if held + inputs + working > budget_bytes:
        return False
    return available is None or inputs + working <= LIVE_FREE_FRACTION * available


#: The compact layout new stores are written in (byte planes deflated per
#: chunk of rows; crates/polariseq-core/src/store.rs). Its size depends on the
#: data; measured with byte planes and deflate: 1.24 bytes a value on the human
#: fetal atlas (1.0% dense) and 0.96 on scTab (8.8% dense). Planned at 2.0, about
#: twice what was measured, so a disk check never reserves too little.
COMPACT_STORE_BYTES_PER_NNZ = 2.0


def _compact_layout() -> bool:
    import os
    return os.environ.get("POLARISEQ_STORE", "compact") != "plain"


def store_bytes(nnz: int, n_rows: int, compact: bool | None = None) -> int:
    """The disk a block store of ``nnz`` values in ``n_rows`` rows takes, in
    the layout new stores are written in (compact unless the environment sets
    ``POLARISEQ_STORE=plain``), or in the one given."""
    compact = _compact_layout() if compact is None else compact
    per_nnz = COMPACT_STORE_BYTES_PER_NNZ if compact else CSR_BYTES_PER_NNZ
    return int(per_nnz * nnz) + STORE_BYTES_PER_ROW * (n_rows + 1)


def _store_bytes(nnz: int, n_rows: int) -> int:
    """The on-disk copy the out-of-core mode writes."""
    return store_bytes(nnz, n_rows)


def _compressed_bytes(nnz: int, n_rows: int) -> int:
    return COMPRESSED_BYTES_PER_NNZ * nnz + COMPRESSED_BYTES_PER_ROW * (n_rows + 1)


@dataclass(frozen=True)
class StepCost:
    """Predicted memory for one pipeline step.

    ``resident`` is what is alive across the step (the matrix, mostly);
    ``transient`` is what the step allocates on top. Peak is their sum.
    """

    name: str
    resident: int
    transient: int
    note: str = ""
    #: For the graph steps: where their inputs are read from, ``"loaded"``
    #: (into memory first; counted twice in ``resident``, the copy and the
    #: file's pages) or ``"mapped"`` (in place from the project's files; their
    #: pages counted once).
    inputs: str = ""

    @property
    def peak(self) -> int:
        return self.resident + self.transient


@dataclass
class Plan:
    """A costed plan for one dataset under one budget."""

    n_cells: int
    n_genes: int
    nnz: int
    budget: Budget
    storage: str
    #: Rows per block the kernels should use under this budget.
    rows_per_block: int = MAX_ROWS_PER_BLOCK
    steps: list[StepCost] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: Smallest RAM budget any block size can satisfy in this storage mode.
    #: Below it the plan does not fit by construction, and the caller should
    #: refuse rather than shrink blocks further: measured on the 800K-cell
    #: sweep, requests under the floor do not just fail, they demand *more*
    #: than requests at the floor, because the extra blocks cost more in
    #: transient churn than the smaller blocks save. Zero where the concept
    #: does not apply (in-memory modes have no budget-derived sizing).
    min_budget_bytes: int = 0
    #: For several datasets planned as one: each one's label, path, shape and
    #: genes kept in the merged matrix, whose ``nnz`` is then their total (an
    #: upper bound when the merge keeps only the shared genes).
    datasets: list[dict[str, Any]] = field(default_factory=list)
    #: What the run writes to the project folder, by part: ``copy`` (out of
    #: core, the copy of the counts), ``pca_copy`` (out of core, PCA's copy of
    #: the selected genes, alive during PCA) and ``results`` (the embedding,
    #: the neighbour graph and UMAP, memory-mapped in every mode).
    disk: dict[str, int] = field(default_factory=dict)

    @property
    def disk_peak_bytes(self) -> int:
        """The most the project folder holds at once: every part together."""
        return sum(self.disk.values())

    @property
    def disk_peak_gb(self) -> float:
        return self.disk_peak_bytes / _GB

    @property
    def peak_bytes(self) -> int:
        return max((s.peak for s in self.steps), default=0)

    @property
    def peak_gb(self) -> float:
        return self.peak_bytes / _GB

    @property
    def min_budget_gb(self) -> float:
        return self.min_budget_bytes / _GB

    @property
    def fits(self) -> bool:
        return self.peak_bytes <= self.budget.ram_bytes

    @property
    def limiting_step(self) -> str:
        if not self.steps:
            return ""
        return max(self.steps, key=lambda s: s.peak).name

    def summary(self) -> dict[str, Any]:
        return {
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "nnz": self.nnz,
            "storage": self.storage,
            "rows_per_block": self.rows_per_block,
            "peak_gb": round(self.peak_gb, 3),
            "budget_gb": round(self.budget.ram_gb, 3),
            "min_budget_gb": round(self.min_budget_gb, 3),
            "fits": self.fits,
            "limiting_step": self.limiting_step,
            "steps": {s.name: round(s.peak / _GB, 3) for s in self.steps},
            "graph_inputs": {s.name: s.inputs for s in self.steps if s.inputs},
            "warnings": list(self.warnings),
            "disk_gb": {k: round(v / _GB, 3) for k, v in self.disk.items()},
            "disk_peak_gb": round(self.disk_peak_gb, 3),
            **({"datasets": [dict(d) for d in self.datasets]} if self.datasets else {}),
        }

    def report(self) -> str:
        """A readable table — what `ps.plan()` prints."""
        head = ""
        if self.datasets:
            width = max(len(d["label"]) for d in self.datasets)
            head = f"{len(self.datasets)} datasets, analyzed as one:\n" + "".join(
                f"  {d['label']:<{width}}  {d['n_cells']:,} cells x {d['n_genes']:,} genes, "
                f"{d['nnz']:,} nonzeros; {d['genes_kept']:,} genes kept\n"
                for d in self.datasets
            )
        head += (
            f"{self.n_cells:,} cells x {self.n_genes:,} genes, "
            + ("at most " if any(d["genes_kept"] < d["n_genes"] for d in self.datasets) else "")
            + f"{self.nnz:,} nonzeros\n"
            f"storage: {self.storage}   budget: {self.budget.ram_gb:.1f} GB RAM, "
            f"{self.budget.threads} threads   block: {self.rows_per_block:,} rows\n"
        )
        width = max((len(s.name) for s in self.steps), default=4)
        rows = "\n".join(
            f"  {s.name:<{width}}  {s.peak / _GB:7.3f} GB"
            + (f"   {s.note}" if s.note else "")
            for s in self.steps
        )
        verdict = (
            f"peak {self.peak_gb:.2f} GB at '{self.limiting_step}' — "
            + ("fits" if self.fits else f"EXCEEDS the {self.budget.ram_gb:.1f} GB budget")
        )
        if self.storage == "spill" and self.min_budget_bytes:
            verdict += (
                f"\n  working-set floor: {self.min_budget_gb:.2f} GB — no block size"
                " meets a smaller budget"
            )
            if self.budget.ram_bytes < self.min_budget_bytes:
                verdict += (
                    f"; this {self.budget.ram_gb:.1f} GB budget is below it,"
                    " the run is refused"
                )
        if self.disk:
            parts = {"copy": "copy of the counts", "pca_copy": "PCA's copy of the selected genes",
                     "results": "results"}
            verdict += (f"\n  project folder at most {self.disk_peak_gb:.1f} GB: "
                        + ", ".join(f"{parts.get(k, k)} {v / _GB:.1f}"
                                    for k, v in self.disk.items()))
        if self.limiting_step in _GRAPH_STEPS:
            verdict += (
                f"\n  note: the limit is the '{self.limiting_step}' stage, whose working "
                "memory grows with the cells whatever the storage mode — a different "
                "storage mode will not help; reduce cells or n_neighbors"
            )
        warn = "".join(f"\n  ! {w}" for w in self.warnings)
        return f"{head}{rows}\n{verdict}{warn}"


def estimate(
    n_cells: int,
    n_genes: int,
    nnz: int,
    budget: Budget,
    *,
    n_hvg: int = 2000,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    storage: str | None = None,
    with_umap: bool = True,
    n_files: int = 1,
    densest_nnz_per_row: int | None = None,
    network_project: bool = False,
) -> Plan:
    """Cost the standard pipeline for a dataset of this shape.

    Args:
        n_cells, n_genes, nnz: Dataset shape (from a metadata scan).
        budget: What the run may use.
        n_hvg, n_pcs, n_neighbors: Pipeline settings that affect cost.
        storage: Force ``"ram"``, ``"compressed"`` or ``"spill"``; by default
            the cheapest mode that fits is chosen.
        with_umap: Include the UMAP step (the largest graph-stage allocation).
        n_files: How many files are read as this one dataset. Several files
            read into memory are merged into one matrix, and while they are
            the files' rows and the merged matrix are both alive.
        densest_nnz_per_row: Values per cell of the densest of several files.
            Out of core, the gene-selection pass has one block per thread in
            flight, and when the files differ in density the densest blocks
            hold more than the average; given this, that pass is charged at
            the densest file's density (see ``BLOCK_DENSITY_HEADROOM``).
        network_project: The project folder is on a network file system.
            A graph step that would read its inputs in place from there is
            warned about, with the budget that reads them into memory.

    Returns:
        A :class:`Plan`. Note it is a *prediction*: the estimate is built from
        array sizes, so it tracks the real peak closely for the matrix-bound
        steps and less so for allocator overhead.
    """
    threads = max(1, budget.threads)
    warnings: list[str] = []

    # Storage mode: the cheapest that fits, unless one was demanded.
    ram_bytes = _csr_bytes(nnz, n_cells)
    comp_bytes = _compressed_bytes(nnz, n_cells)
    if storage is None:
        # Recommend only the modes the public API runs: in memory
        # (ps.pp / ps.tl) or out of core (ps.process_diskbacked). The packed
        # "compressed" form exists in the core but has no user-facing path
        # yet, so recommending it would name a mode no one can choose; it
        # can still be costed by asking for it explicitly.
        storage = "ram" if ram_bytes <= budget.ram_bytes * 0.5 else "spill"
    if storage == "spill":
        # Spilled: only a block of rows is resident at a time — but the
        # streaming passes keep a window of the spill files mapped, which
        # occupies the machine without belonging to the process. Both the
        # in-flight blocks and that window are part of what the budget must
        # cover, so the model carries both.
        # The store's row pointers are held in memory too, 8 bytes a row.
        resident = int(
            min(SPILL_RESIDENCY_BYTES_PER_NNZ * nnz, SPILL_RESIDENCY_CAP_BYTES)
        ) + STORE_BYTES_PER_ROW * (n_cells + 1)
        store_bytes = _store_bytes(nnz, n_cells)
        if store_bytes > budget.spill_bytes:
            warnings.append(
                f"spill needs {store_bytes / _GB:.1f} GB but the budget allows "
                f"{budget.spill_gb:.1f} GB on {budget.spill_dir}"
            )
    else:
        resident = ram_bytes if storage == "ram" else comp_bytes

    # What a block of rows costs while it is being worked on, and how big a
    # block should therefore be. In RAM the blocks are *slices* of the matrix
    # and cost nothing extra; the compressed form decodes into per-thread
    # scratch; the streamed form reads `PREFETCH` blocks ahead.
    #
    # The block size is a *consequence of the budget*, not a constant. A fixed
    # 32 768-row block is 292 MB on a matrix with ~1 100 nonzeros per cell,
    # and several of those in flight will not fit a small budget — so solve
    # for the rows that keep the in-flight blocks inside BLOCK_BUDGET_FRACTION
    # of the RAM ceiling.
    nnz_per_row = max(1, nnz // max(1, n_cells))
    if storage == "ram":
        copies, rows_per_block, block_bytes = 0, MAX_ROWS_PER_BLOCK, 0
    else:
        # Out of core, FileCsr keeps one block per thread in flight (its
        # `for_each_block` runs batches of `rayon::current_num_threads()`), so
        # the thread count, not the read-ahead, is what is in flight. Measured:
        # 8.4 GB at 4 threads and 17.7 GB at 24 on the same two scTab files.
        copies = threads
        allowance = max(1, int(budget.ram_bytes * BLOCK_BUDGET_FRACTION))
        # Size the block with the *same* per-nonzero cost the block is then
        # charged at. Dividing the allowance by 8 while a spilled block
        # actually costs 12 (x the safety margin) made BLOCK_BUDGET_FRACTION
        # a fiction: blocks sized to fill 25% of the ceiling filled ~47% of
        # it. The budget has to mean what it says, so the sizer and the cost
        # model now use one constant.
        per_nnz = (
            SPILL_BYTES_PER_NNZ_IN_FLIGHT * SPILL_INFLIGHT_SAFETY
            if storage == "spill"
            else CSR_BYTES_PER_NNZ
        )
        rows_per_block = int(allowance // max(1, int(copies * nnz_per_row * per_nnz)))
        rows_per_block = int(min(MAX_ROWS_PER_BLOCK, max(MIN_ROWS_PER_BLOCK, rows_per_block)))
        # Quantize to 1024-row steps. The block size sets the block-fold
        # summation order for every streamed pass, so a size that floats
        # with ambient free RAM makes two runs on the same machine disagree
        # by exactly the HVG boundary genes -- measured on the heart_50k
        # ladder: reps at rows_per_block 10,785-16,038 selected different
        # gene sets and produced 90-degree PCA "disagreements" against
        # themselves. Coarse steps keep the size stable under small memory
        # variation; pin the budget (ps.set_budget(ram_gb=...)) to make it
        # exact.
        rows_per_block = max(MIN_ROWS_PER_BLOCK, (rows_per_block // 1024) * 1024)
        # What is actually in flight, which is not `copies` full-size blocks.
        # Two corrections, both of which were wrong in opposite directions and
        # between them produced a predictor that crossed from 5.8x *over* at
        # 10k cells to 0.67x *under* at 100k:
        #
        # 1. A block cannot be larger than the matrix, and there cannot be
        #    more blocks in flight than exist. Billing `copies` blocks of
        #    `rows_per_block` on a dataset smaller than one block charged
        #    110,592 rows' worth of buffers for a 10,000-row matrix.
        # 2. When the whole matrix is in flight, use the real `nnz` rather
        #    than `rows * (nnz // rows)`, whose truncation loses up to one
        #    row's worth of nonzeros per row.
        rows_eff = min(rows_per_block, max(1, n_cells))
        n_blocks = -(-max(1, n_cells) // rows_eff)  # ceil
        rows_in_flight = min(n_cells, min(copies, n_blocks) * rows_eff)
        nnz_in_flight = nnz if rows_in_flight >= n_cells else rows_in_flight * nnz_per_row
        if storage == "spill":
            block_bytes = int(
                nnz_in_flight * SPILL_BYTES_PER_NNZ_IN_FLIGHT * SPILL_INFLIGHT_SAFETY
            )
        else:
            block_bytes = _csr_bytes(nnz_in_flight, rows_in_flight)

    # The same blocks at the density of the densest ones, for the one pass
    # that works on a block per thread at once (gene selection).
    dense_per_row = nnz_per_row
    dense_block_bytes = block_bytes
    if storage == "spill" and densest_nnz_per_row:
        dense_per_row = max(nnz_per_row, int(densest_nnz_per_row * BLOCK_DENSITY_HEADROOM))
        dense_nnz_in_flight = (
            nnz if rows_in_flight >= n_cells else min(nnz, rows_in_flight * dense_per_row)
        )
        dense_block_bytes = int(
            dense_nnz_in_flight * SPILL_BYTES_PER_NNZ_IN_FLIGHT * SPILL_INFLIGHT_SAFETY
        )

    steps: list[StepCost] = []

    # Runtime overhead is present for the whole run, so it sits in every
    # step's resident total rather than being a step of its own.
    resident += PROCESS_OVERHEAD_BYTES

    # Load: the matrix, plus one block in flight. Several files read into
    # memory are merged: until the merge ends their rows and the merged matrix
    # are both alive, so the load holds the matrix twice. Measured: 13.39 GB
    # for a 6.6 GB matrix of 2 files, 66.4 GB for a 32.9 GB matrix of 10 files;
    # without this term the plan said 7.71 and 37.82 GB.
    merging = storage == "ram" and n_files > 1
    steps.append(
        StepCost(
            "load",
            resident,
            block_bytes + (ram_bytes if merging else 0) + (IN_MEMORY_READ_BYTES if storage == "ram" else 0),
            "streamed in blocks" if storage == "spill" else
            (f"{n_files} files merged: the rows and the merged matrix both alive"
             if merging else ""),
        )
    )

    # QC: per-thread gene accumulators (u32 counts + f64 totals) and the
    # per-cell outputs (f32 counts, u32 genes).
    qc_transient = threads * n_genes * (4 + 8) + n_cells * (4 + 4)
    if storage == "ram":
        # The per-cell results are built in Python, and they stay.
        qc_transient += n_cells * QC_BYTES_PER_CELL
    steps.append(StepCost("qc", resident, qc_transient + block_bytes))
    if storage == "ram":
        resident += n_cells * QC_BYTES_PER_CELL

    # normalize_total + log1p rewrite values in place.
    steps.append(StepCost("normalize+log1p", resident, block_bytes, "in place"))

    # HVG: per-thread f64 sums and sums-of-squares, then the subset — during
    # which the original and the subset are both alive. This is usually the
    # peak of the whole run.
    nnz_hvg = min(nnz, int(nnz * (n_hvg / max(1, n_genes)) * HVG_NNZ_ENRICHMENT))
    # In memory, the transforms (normalization, log, the statistics' squares)
    # are views that write each chunk's transformed values (chunks of
    # max(n / 32, 32,768) rows, `rsvd::chunk_rows`), one chunk per thread.
    view_chunks = min(GRAM_CHUNKS_MAX, max(1, -(-n_cells // GRAM_CHUNK_MIN_ROWS)))
    in_memory_views = int(min(threads, view_chunks, IN_MEMORY_VIEW_CHUNKS_MAX) * (nnz / view_chunks)
                          * IN_MEMORY_VIEW_BYTES_PER_NNZ)
    hvg_matrix = _csr_bytes(nnz_hvg, n_cells)
    steps.append(
        StepCost(
            "hvg",
            resident,
            threads * n_genes * 16
            + (int(dense_block_bytes * HVG_INFLIGHT_FACTOR) if storage == "spill"
               else block_bytes + in_memory_views),
            "gene statistics"
            + (f", blocks at {dense_per_row:,} values per cell (densest file)"
               if dense_per_row > nnz_per_row else ""),
        )
    )
    if storage == "spill":
        # The subset is written to a second spill file, not held in RAM.
        steps.append(
            StepCost("hvg subset", resident, block_bytes, "written to a second spill")
        )
        working = PROCESS_OVERHEAD_BYTES
    else:
        steps.append(
            StepCost(
                "hvg subset",
                resident,
                hvg_matrix + in_memory_views,
                f"original + {n_hvg}-gene subset both alive, renormalized through views",
            )
        )
        # After subsetting, the run holds the selected genes twice (see
        # IN_MEMORY_HVG_COPIES) and the runtime overhead.
        working = IN_MEMORY_HVG_COPIES * hvg_matrix + PROCESS_OVERHEAD_BYTES

    # PCA: the Gram path holds an n_hvg^2 f64 covariance plus faer's
    # eigendecomposition workspace (~the same again), and the scores with the
    # buffers behind them (measured per cell, see PCA_BYTES_PER_CELL_PER_PC).
    if n_hvg <= 4096:
        gram = n_hvg * n_hvg * 8
        # The Gram matrix is summed per chunk of rows, and every chunk's
        # partial is kept until all are merged in order (`gram_matrix_blocks`,
        # `BlockFold`): GRAM_CHUNKS_MAX chunks of at least GRAM_CHUNK_MIN_ROWS
        # rows, each partial one n_hvg^2 f64 matrix, plus the sum.
        chunks = min(GRAM_CHUNKS_MAX, max(1, -(-n_cells // GRAM_CHUNK_MIN_ROWS)))
        pca_transient = int(GRAM_EIGEN_WORKSPACE_FACTOR * gram) + (chunks + 1) * gram + (
            n_cells * n_pcs * PCA_BYTES_PER_CELL_PER_PC
        )
        pca_note = (f"exact Gram ({n_hvg}^2 f64, {chunks} partial sums kept until merged, "
                    "+ eigen workspace)")
    else:
        krylov = n_cells * (n_pcs + 10) * 5 * 4
        pca_transient = krylov + 2 * n_cells * n_pcs * 4
        pca_note = "randomized block Krylov"
    # From PCA on, the run also holds the per-cell annotations in Python.
    pca_resident = working + n_cells * OBS_BYTES_PER_CELL
    if storage == "ram":
        # In memory, the full matrix is released as PCA starts, not before it:
        # on Windows the step began with the matrix and the selected genes
        # still resident (fetal 1M cells: 5.42 GB at its start, 0.94 GB at
        # its end), so the step is planned at no less than that.
        pca_resident = max(pca_resident, resident + hvg_matrix - pca_transient)
    steps.append(StepCost("pca", pca_resident, pca_transient, pca_note))

    # From here the matrix is not needed: the graph stages work on the
    # embedding. Callers that keep `adata.X` alive pay `working` as well, so
    # it stays in `resident`. The embedding and the graphs are files of the
    # project, not held between steps; each graph step reads what it needs
    # into memory when it fits (`loads_inputs`, the rule the run applies) and
    # in place otherwise.
    emb = n_cells * n_pcs * 4
    graph = n_cells * n_neighbors * 8
    # The connectivity CSR (u32 index + f32 weight per entry): the fuzzy set
    # stores both directions of every edge, mutual pairs once.
    entries = int(n_cells * n_neighbors * FUZZY_ENTRY_FACTOR)
    conn = entries * 8

    # What the graph steps hold besides their inputs: the run (`working`) and
    # the per-cell annotations kept in Python (see OBS_BYTES_PER_CELL).
    held = working + n_cells * OBS_BYTES_PER_CELL

    def graph_step(name: str, inputs: int, transient: int, what: str, also_read: int = 0) -> None:
        # Resident memory counts the inputs once when they are read in place
        # (their pages) and twice when they are read into memory: the copy,
        # and the pages of the file the step before wrote, which stay
        # resident (see LOADED_INPUT_COPIES). The decision itself is the run's
        # rule, unchanged.
        loaded = loads_inputs(working, inputs, transient, budget.ram_bytes)
        how = (f"{what} read into memory ({inputs / _GB:.2f} GB)" if loaded
               else f"{what} read in place from the project ({inputs / _GB:.2f} GB)")
        note = f"NN-Descent forest at {threads} threads; {how}" if name == "neighbors" else how
        copies = LOADED_INPUT_COPIES if loaded else 1
        steps.append(StepCost(name, held + copies * inputs + also_read, transient, note,
                              inputs="loaded" if loaded else "mapped"))

    # kNN: NN-Descent's working memory (fitted, see the constants): the
    # forest's construction, which depends on the threads, sets the peak, and
    # it includes the distance graph the step returns.
    graph_step("neighbors", emb, neighbors_working_bytes(n_cells, n_neighbors, threads), "embedding")
    graph_step("leiden", conn, leiden_working_bytes(entries), "graph")
    if with_umap:
        # The fuzzy edge list, the per-edge SGD schedules and the layout's
        # buffers, per connectivity entry (measured, see UMAP_BYTES_PER_ENTRY),
        # and the 2-D embedding. The multilevel start also reads the whole
        # embedding in place, for its k-means groups.
        graph_step("umap", conn, umap_working_bytes(entries, n_cells), "graph", also_read=emb)
    if network_project:
        in_place = [s for s in steps if s.inputs == "mapped"]
        if in_place:
            need = max(s.peak + (emb if s.name == "neighbors" else conn) for s in in_place)
            warnings.append(
                f"{', '.join(s.name for s in in_place)} would read their inputs in place from the "
                f"project folder, on a network file system, where that can stall for hours (unless a "
                f"local disk has room for a copy); a budget of {need / _GB:.1f} GB reads them into memory"
            )

    # The working-set floor, spill mode only. Sizing ties the in-flight block
    # term to a fixed share of the budget (block_bytes ≈
    # BLOCK_BUDGET_FRACTION × budget while unclamped), so peak ≈ floor_base +
    # fraction × budget, and the smallest self-consistent budget solves
    # floor_base + fraction × b = b. Beneath it no block size fits — the
    # honest response is refusal, not smaller blocks: on the measured 800K
    # sweep, 1–3 GB requests demanded 3.2–5.1 GB, *more* than the 4 GB
    # request's 3.22 GB, because the churn of extra blocks outweighed their
    # size. In-memory modes have no budget-derived sizing, so their minimum
    # is simply their peak.
    if storage == "spill":
        # `resident` already carries PROCESS_OVERHEAD_BYTES (added above), so
        # this is residency + process — the budget-independent part of every
        # streaming step's peak.
        floor_base = resident
        # The in-flight term is tied to a fixed share of the budget while
        # the derived block size is unclamped, so peak(b) = floor_base +
        # clamp(fraction·b, block_min, block_max) and the smallest budget
        # that fits solves peak(b) = b. Three regimes, selected by where
        # floor_base sits relative to the clamps:
        #   floor_base ∈ [3·block_min, 3·block_max] → floor_base / (1-f)
        #   floor_base < 3·block_min  (tiny data)   → floor_base + block_min
        #   floor_base > 3·block_max  (huge/dense)  → floor_base + block_max
        min_rows_eff = min(MIN_ROWS_PER_BLOCK, max(1, n_cells))
        min_blocks = -(-max(1, n_cells) // min_rows_eff)
        min_rows_in_flight = min(n_cells, min(copies, min_blocks) * min_rows_eff)
        if min_rows_in_flight >= n_cells:
            min_nnz_in_flight = nnz
        else:
            min_nnz_in_flight = min_rows_in_flight * nnz_per_row
        block_min = int(
            min_nnz_in_flight * SPILL_BYTES_PER_NNZ_IN_FLIGHT * SPILL_INFLIGHT_SAFETY
        )
        max_rows_eff = min(MAX_ROWS_PER_BLOCK, max(1, n_cells))
        max_blocks = -(-max(1, n_cells) // max_rows_eff)
        max_rows_in_flight = min(n_cells, min(copies, max_blocks) * max_rows_eff)
        if max_rows_in_flight >= n_cells:
            max_nnz_in_flight = nnz
        else:
            max_nnz_in_flight = max_rows_in_flight * nnz_per_row
        block_max = int(
            max_nnz_in_flight * SPILL_BYTES_PER_NNZ_IN_FLIGHT * SPILL_INFLIGHT_SAFETY
        )
        if floor_base < 3 * block_min:
            min_budget_bytes = floor_base + block_min
        elif floor_base > 3 * block_max:
            min_budget_bytes = floor_base + block_max
        else:
            min_budget_bytes = int(floor_base / (1.0 - BLOCK_BUDGET_FRACTION))
    else:
        min_budget_bytes = max((s.peak for s in steps), default=0)

    if budget.ram_bytes and max(s.peak for s in steps) > budget.ram_bytes:
        worst = max(steps, key=lambda s: s.peak)
        warnings.append(
            f"'{worst.name}' needs {worst.peak / _GB:.1f} GB, over the "
            f"{budget.ram_gb:.1f} GB budget"
        )
        if storage != "spill":
            warnings.append("try storage='spill' (ps.process_diskbacked)")
        elif budget.ram_bytes < min_budget_bytes:
            warnings.append(
                f"the {budget.ram_gb:.1f} GB budget is below the spill working-set"
                f" floor of {min_budget_bytes / _GB:.1f} GB (file residency + process"
                " + in-flight blocks); no block size meets it — raise the budget"
                " or analyse a subset"
            )
        elif not budget.allow_lossy:
            warnings.append(
                "an approximate plan may fit; enable it explicitly with "
                "allow_lossy=True"
            )

    # The project folder. Results are memory-mapped files in every mode: the
    # embedding, the distance graph and connectivities, the UMAP. Out of core
    # it also holds the copy of the counts and, during PCA, the copy of the
    # selected genes (measured on 22.19M cells: results 11.0 GB, estimated
    # here 11.9 GB).
    disk = {}
    if storage == "spill":
        disk["copy"] = _store_bytes(nnz, n_cells)
        disk["pca_copy"] = _store_bytes(nnz_hvg, n_cells)
    disk["results"] = emb + graph + conn + (n_cells * 8 if with_umap else 0)

    return Plan(
        n_cells=n_cells,
        n_genes=n_genes,
        nnz=nnz,
        budget=budget,
        storage=storage,
        rows_per_block=rows_per_block,
        steps=steps,
        warnings=warnings,
        min_budget_bytes=min_budget_bytes,
        disk=disk,
    )


def max_cells(
    budget: Budget,
    *,
    n_genes: int = 30_000,
    nnz_per_cell: int = 2_000,
    storage: str | None = None,
    **kwargs: Any,
) -> int:
    """Largest cell count that fits `budget`, for a dataset of this density.

    The scaling question stated the other way round: not "will this run?" but
    "how big can it be?". Solved by bisection on :func:`estimate`, so it
    always agrees with the cost model rather than drifting from it.
    """
    def fits(n: int) -> bool:
        if n <= 0:
            return True
        plan = estimate(
            n, n_genes, n * nnz_per_cell, budget, storage=storage, **kwargs
        )
        return plan.fits

    if not fits(1_000):
        return 0
    lo, hi = 1_000, 1_000
    while fits(hi) and hi < 2_000_000_000:
        lo, hi = hi, hi * 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo
