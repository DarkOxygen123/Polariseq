# How Leiden, Harmony and the pseudobulk test were rewritten for 0.2.0

Up to 0.1.1, three modules followed code under the GNU General Public License: Leiden (`leiden.rs`), the
pseudobulk test (`pseudobulk.rs`) and Harmony (`harmony.rs`). For 0.2.0 they were written anew from the
publications, the tools' documentation and permissively licensed code (`THIRD_PARTY_NOTICES.md`), and each
was checked against the 0.1.x code and the reference tools on the datasets of the benchmarks, for quality
and speed. Then every Polariseq benchmark was run again with 0.2.0. This file records those checks and
their results; the scripts are in this folder.

The comparisons of old and new code ran on a development build that held both implementations, selected by
the switches `POLARISEQ_LEIDEN_IMPL` and `POLARISEQ_HARMONY_IMPL`. The release holds the new code only, so
with it `leiden_compare.py` and `harmony_compare.py` compare the new code with the reference tools (arms
`new` and `igraph`, `harmonypy` and `new`).

## 1. Leiden

**Rewrite.** `crates/polariseq-core/src/leiden.rs`: local moving, refinement, reduction and the level
loop are rewritten after Traag, Waltman & van Eck (2019, Algorithms A.2 and A.3) and the authors' CWTS
`networkanalysis` (MIT). Kept from 0.1.x (our own code): the graph type, modularity, the relative move
tolerance and move cap, `coarsen_degenerate`, the tests.

**A bug found in 0.1.x.** Its aggregation stored a refined community's internal weight as a self-loop
of twice the internal weight and then counted self-loops twice in the strengths, so every aggregated
level over-counted internal weight (two cliques: strength 12.05 where the members sum to 6.05). The
rewrite carries node weights as sums (no self-loops), so it cannot occur. To separate the fix from the
rewrite, the comparison has three arms.

**Comparison.** `leiden_compare.py`: the graph is built once with the current pipeline (normalization,
genes, PCA and neighbours are unaffected), then clustered with each arm and seeds 0-9. Arms: `old`
(0.1.x), `oldfix` (0.1.x with the aggregation fixed), `new`; and python-igraph's Leiden (Scanpy's
reference) where the graph is small enough. The same for Louvain (`fast_community`).

Results (medians over seeds; ARI new vs old next to old's own seed-to-seed ARI):

| dataset | cells | ARI new vs old (old seed-to-seed) | modularity old -> new (igraph) | ARI with labels old -> new | seconds old -> new |
|---|---|---|---|---|---|
| PBMC 3k | 2,700 | 0.913 (0.899) | 0.70453 -> 0.70456 (0.70479) | | 0.019 -> 0.016 |
| Kang | 28,871 | 0.925 (0.918) | 0.87101 -> 0.87152 (0.87175) | 0.491 -> 0.504 | 0.223 -> 0.177 |
| PBMC 10k | 11,769 | 0.853 (0.847) | 0.84708 -> 0.84824 (0.84805) | | 0.070 -> 0.061 |
| PBMC 20k | 23,837 | 0.839 (0.836) | 0.85080 -> 0.85117 (0.85173) | | 0.193 -> 0.148 |
| neuron 10k | 11,843 | 0.687 (0.683) | 0.84291 -> 0.84561 (0.84494) | | 0.067 -> 0.054 |
| brain 50k | 50,000 | 0.668 (0.650) | 0.84872 -> 0.85065 (0.84945) | | 0.386 -> 0.314 |
| brain 100k | 100,000 | 0.599 (0.597) | 0.85124 -> 0.85283 (0.85220) | | 0.914 -> 0.727 |
| fetal 60k | 60,000 | 0.850 (0.849) | 0.90451 -> 0.90553 (0.90513) | 0.456 -> 0.479 | 0.417 -> 0.350 |
| fetal 4.06M | 4,000,604 | 0.805 (0.807) | 0.92304 -> 0.92298 (not run) | 0.412 -> 0.441 | 78.6 -> 68.3 |
| brain 1.3M | 1,292,537 | 0.590 (0.583) | 0.86977 -> 0.87172 (not run) | | 15.2 -> 13.4 |

The new Leiden agrees with 0.1.x as well as 0.1.x agrees with itself across seeds, reaches equal or
higher modularity (at or above igraph's), agrees as well or better with published labels, and is 13-25 %
faster. Louvain gains most from the fix (Kang: ARI with types 0.42 -> 0.51, 17-19 clusters -> 13-14).
At 4M cells the modularity is level within the seed spread (old's best seed 0.9250, new's 0.9238), types
recovered 20 -> 20.5. Brain 1.3M's seconds are a separate quiet run of three seeds per arm (in the
10-seed run a build overlapped the old arm).

## 2. Pseudobulk

**Rewrite.** `crates/polariseq-core/src/pseudobulk.rs`: the filter (Chen, Lun & Smyth 2016), TMM
(Robinson & Oshlack 2010), log-CPM, the linear model and the empirical Bayes moderation with a trend
(Smyth 2004; Phipson et al. 2016), and the special functions (DLMF; libm for `lgamma` and `erfc`),
written from the papers and the edgeR and limma documentation. Kept (our own): the per-unit sums, the
design and pairing rule, the output assembly, the per-cell Welch test, the R-compatible helpers
(type-7 quantile, average ranks, BH) and the Cholesky solvers. The 0.1.x special functions included
continued-fraction code in the style of Numerical Recipes, which has its own restrictive licence; they
are replaced too.

**Black-box reference.** `pseudobulk_reference.R` generates pseudobulk datasets (negative binomial
counts, 10-fold library sizes, 2-4 conditions, paired or not, 3-12 samples per condition, all-zero genes,
planted very large counts; and with the argument `small`, 3 to 60 genes) and records edgeR 4.10.5 and
limma 3.68.5's outputs step by step. Details the documentation leaves open were fixed by comparing
candidates with R's numbers, never by reading R's code:

- CPM at the filter's cutoff is `count x 1e6 / library size` (the other orders disagree with edgeR
  where a count sits exactly on the cutoff; 0.1.x disagreed in 5 of 200 datasets).
- TMM's log ratio is `log2((o/N_o)/(r/N_r))`; with it all 948 sample pairs match (ties at the trimming
  boundaries depend on it), ranks averaged, `floor(0.3 G) + 1 ... G - floor(0.3 G)`.
- The prior's trend is a natural cubic spline with no interior knot below 6 genes, one below 30, else
  two (at the tertiles).
- Variances are raised to at least 1e-5 of their median before the moment estimation.

Agreement with R (relative differences):

| | 0.1.x | new (Rust, `pseudobulk_check.py`) |
|---|---|---|
| 200 generated datasets: genes kept as edgeR | 195 / 200 | 200 / 200 |
| p-values, adjusted p-values | 1.8e-9 | 3.5e-12 |
| prior degrees of freedom | 2.3e-3 | 9e-15 |
| 150 datasets of 3-60 genes: genes kept, p-values | | 150 / 150, 1.8e-12 |
| Kang et al., 15,474 tests: p-values | 1.7e-11 | 1.3e-11 |
| Kang, genes at adjusted p < 0.05 (R: 6,106) | 6,106 | 6,106 |

`python/polariseq/tests/test_compare.py` passes (7 tests, including the stored Kang reference and the
per-cell Welch test against SciPy). The pseudobulk results on Kang et al. are unchanged.

## 3. Harmony

**Rewrite.** `crates/polariseq-core/src/harmony.rs`: k-means++ seeding and Lloyd for the starting
centroids, the assignment rounds over shuffled blocks with the other cells' tallies fixed, the objective
and its convergence windows, the per-cluster ridge regression (solved from its Gram matrix: through the
Schur complement of the arrow matrix for one batch variable, by elimination for several), written from
Korsunsky et al. (2019) and harmony-pytorch (BSD-3); R harmony's documented options (a penalty per
variable, `lambda = alpha x` expected count when none is given, `batch_prop_cutoff`). Kept (our own): the
tiling, the tallies, the grouping of cells by batch combination, the weighted sums, the application of
the correction, determinism across thread counts.

Two choices the documentation leaves open were fixed by comparing with harmonypy 2.0.2
(`harmony_compare.py`; `diff` is how far two corrections are apart relative to the size of the
correction):

| median `diff` from harmonypy (harmonypy's own seed spread) | Kang, condition + donor (0.080) | fetal 60k, 28 fetuses (0.308) |
|---|---|---|
| paper's diversity weight `((E+1)/(O+1))^theta`, centroids from assignments | 0.072 | 0.449 |
| paper's weight, centroids from the ridge intercepts | 0.088 | 0.470 |
| `((2E+1)/(O+E+1))^theta`, centroids from assignments | 0.060 | 0.291 |
| `((2E+1)/(O+E+1))^theta`, centroids from the ridge intercepts (chosen) | 0.069 | 0.279 |

Results, medians over 5 seeds:

| | Kang, condition | Kang, condition + donor | fetal 60k, fetus |
|---|---|---|---|
| `diff` from harmonypy, old -> new (harmonypy's own) | 0.029 -> 0.032 (0.030) | 0.072 -> 0.069 (0.080) | 0.319 -> 0.279 (0.308) |
| neighbours shared with harmonypy, old -> new | 0.980 -> 0.979 | 0.951 -> 0.948 | 0.854 -> 0.865 |
| same-type neighbours, harmonypy / old / new | 0.885 / 0.885 / 0.885 | 0.873 / 0.873 / 0.873 | 0.890 / 0.887 / 0.894 |
| batch mixing, harmonypy / old / new | 0.837 / 0.837 / 0.837 | 0.837 / 0.837 / 0.837 | 0.457 / 0.457 / 0.457 |
| seconds, harmonypy / old / new | 3.9 / 0.91 / 0.97 | 6.0 / 1.19 / 1.13 | 6.5 / 1.01 / 0.95 |

Kang et al. downstream: the graph built after the old and after the new Harmony,
Leiden 0.6 seeds 0-4: 14-15 clusters both, ARI with the published types 0.49-0.52 both, the clusters
of the two agree at ARI 0.92-0.97 for the same seed (the old pipeline's own seed-to-seed: 0.90-0.97).

## 4. Planner: the neighbours step, found with the Tahoe-100M run

**What was wrong.** On 95.6M cells (16 threads, budget 150 GB) the plan said 129.4 GB, limited by UMAP;
the run held 159.5 GB of real memory (178.9 GB with mapped files) in the neighbours step. The plan was
right where it had been calibrated: the 22.1M scTab run (57.5 planned, 57.4 measured), PCA at every scale
(0.95-1.07), the 13.7M Tahoe-A run (18.6 against 19.9). The neighbours step was 1.11, 1.18 and 1.29 times
the plan at 13.7M, 22.1M and 95.6M cells.

**Where the memory is.** The monitor saw one process throughout. The step has two short spikes (minute 2
and 38 of 230) and a plateau of about 67 GB. A sweep of `pp.neighbors` on random slices of the 100M
embedding (`neighbors_memory.py`, `benchmarks/atlas/tahoe_all/neighbors_memory.sbatch`; per-phase table
by `neighbors_phases.py`):

| cells (16 threads) | peak anon GB | B per cell (embedding removed) | peak is in |
|---:|---:|---:|---|
| 5M | 9.5 | 1,706 | forest build |
| 10M | 21.0 | 1,901 | forest build |
| 20M | 36.8 | 1,638 | forest build (rest of the step: at most 19.6) |
| 40M | 75.5 | 1,686 | forest build (rest of the step: at most 39.0) |

| threads (20M cells) | 4 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|
| B per cell | 800 | 1,135 | 1,335 | 1,638 |

So the peak is proportional to the cells and grows with the threads, `548 + 68 x threads` bytes per cell
(within 43 B of each point), and it is set by the random-projection forest, which builds its 48 trees at
once. That is why the earlier runs, at 12 threads, looked lower per cell than the 16-thread run.

**What changed.** `python/polariseq/planning.py`: the neighbours step uses that fit
(`NNDESCENT_BYTES_PER_CELL`, `NNDESCENT_BYTES_PER_CELL_PER_THREAD`) in place of 73 bytes per cell per
neighbour. The plan now says 175.7 GB for the 100M run (measured 159.5 / 178.9), 34.9 GB for scTab 22M
(34.1 / 38.5), 21.5 GB for Tahoe-A (19.9 / 22.6), and that the 100M run does not fit a 150 GB budget.
Tests: `test_planning.py`. The other graph steps were refitted later on the runs of the final 0.2.0
build (`CHANGELOG.md`, 0.2.0, and `planner_v020_check.py`).

The forest itself was then changed for 0.2.0 (section 5), which halves the step's peak with the same graph.

## 5. NN-Descent (0.2.0) and the reruns at 1.3M and 4.06M cells

**NN-Descent.** The peak of the forest phase is the copy each worker thread makes of the subtree it splits
sequentially; random hyperplanes split unevenly (brain 1.3M: those subtrees held up to 49% of all cells, 99th
percentile 24%). Subtrees above `GATHER_MAX` (2^20 points) are now split in place, with the same random draws in
the same order (building the trees in groups instead saved under 1 GB at 20M and cost 13%, and was not
kept). On one compute node, the neighbours step on slices of the Tahoe-100M embedding
(`benchmarks/atlas/tahoe_all/verify_nndescent.sbatch`; SHA-256 of all six result arrays):

| cells | graph | peak real memory, 0.1.1 -> 0.2.0 | step time, 0.1.1 -> 0.2.0 |
|---|---|---|---|
| 20M | identical | 36.4 -> 17.4 GB | 668 -> 797 s |
| 40M | identical | 70.7 -> 34.7 GB | 1524 -> 1722 s |

**Laptop** (Apple M2, 8 GB, after a restart, 4 threads, 6 GB budget, with UMAP; the r4 campaign's harness and
settings, `laptop_v020.py` via `run_laptop_after_restart.sh`; 0.1.1 from `.venv-011` alternating repeat by repeat
in the same session). Records: `benchmarks/results/telemetry_v020_umap`, `telemetry_v011_umap_control`.

| out of core | r4 (0.1.x, 2026-10-05) | 0.1.1 control | 0.2.0 |
|---|---|---|---|
| brain 1.3M, seconds (median) | 244, 257, 257 (257) | 238, 237, 266 (238) | 243, 243, 255 (243) |
| fetal 4.06M, seconds (median) | 548, 570, 554 (554) | 465, 467, 465, 472 (466) | 460, 462, 465, 471 (464) |
| footprint, GB | 3.8 / 7.0 | 3.8 / 6.9-7.0 | 3.7-3.8 / 6.9 |

0.2.0 is as fast as 0.1.1 and uses the same memory; both are faster than the r4 records, which were made on a
machine with less memory free. (Before the restart, with swap nearly full and other applications running, the same
runs were 15-25% slower: `telemetry_v020_umap_before_restart_*`, not used.) Full analyses with every figure
(`examples/06_large_run.py`): `benchmarks/results/v020_laptop/{brain1306k,fetal4062k}`: brain 38 clusters,
6.4 min; fetal 41 clusters, 16.5 min, ARI 0.42 with the 77 published types, 20 recovered. Clusters identical to
the run before the restart (ARI 1.000); UMAP coordinates correlate 0.95-0.98 (UMAP is not bit-reproducible
across threads by default). Their "peak memory held" counts resident pages of mapped files and rises with free
memory (2.5 -> 3.6 GB and 3.0 -> 5.1 GB across the restart, also in the figures step); the harness's footprint is
the measure to compare.

**Compute node** (six runs, `cluster/v020_*.sbatch`, records in `$B/runs/v020`): peak memory equal to r4 in every run (within 0.05 GB), clusters 35/36/42/
42/41/41 against 35/37/42/42/43/43. Times from -2% to +29% on shared nodes whose load at start ran from 15 to 44 of
48 cores (r4: 24 to 43); per step they move both ways (UMAP, unchanged code, -18% to +42%), so single runs cannot
separate the code from the node. Leiden was slower in both fetal in-memory runs (92 -> 121 s, 77 -> 111 s) though
faster on the laptop.
