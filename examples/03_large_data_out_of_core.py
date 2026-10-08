"""3 — Data that does not fit in RAM.

Run:  python examples/03_large_data_out_of_core.py [path/to/big.h5ad]

Polariseq's distinguishing feature is that it will finish on a machine too
small for the dataset, at a memory footprint *you choose in advance*. This
example shows the three pieces of that:

    ps.set_budget(...)   what the library is allowed to use
    ps.plan(...)         what a run would cost, before running it
    ps.process_diskbacked(...)   the out-of-core path

With no argument it uses the bundled fixture, which fits in RAM easily — so
the numbers are small, but the mechanism is identical.
"""

import sys
from pathlib import Path

import polariseq as ps

DEFAULT = str(Path(__file__).resolve().parents[1] / "tests/data/medium_10kx2k.h5ad")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT

    # 1. Declare the budget. Pin it: left to auto-detect, it follows whatever
    #    RAM happens to be free, which makes two runs on one machine disagree.
    ps.set_budget(ram_gb=2, threads=4)
    print(ps.budget(), "\n")

    # 2. Ask what it would cost — before committing. This reads the file's
    #    metadata only, so it is instant even for a 100 GB file.
    for storage in ("ram", "spill"):
        plan = ps.plan(path, n_hvg=2000, n_pcs=50, storage=storage,
                       with_umap=False, verbose=False)
        fits = "fits" if plan.fits else "DOES NOT FIT"
        print(f"{storage:6s}: peak {plan.peak_gb:5.2f} GB, limited by "
              f"'{plan.limiting_step}', blocks of {plan.rows_per_block:,} rows — {fits}")

    # 3. How large could this machine go, at this budget?
    print(f"\nmax cells in RAM:        {ps.max_cells(storage='ram'):,}")
    print(f"max cells out-of-core:   {ps.max_cells(storage='spill'):,}")

    # 4. Run out-of-core. The matrix is streamed to SSD and read back in
    #    budget-sized blocks; only per-gene vectors and the embedding are ever
    #    resident. Same kernels as the in-memory path, so the answers match.
    print("\nrunning the out-of-core pipeline ...")
    adata = ps.process_diskbacked(path, n_hvg=2000, n_pcs=50, n_neighbors=15, seed=0)

    n_clusters = len(set(adata.obs["leiden"].tolist()))
    print(f"\n{adata.n_obs:,} cells -> {n_clusters} clusters, "
          f"embedding {adata.obsm['X_pca'].shape}")
    print("The matrix never had to fit in memory.")


if __name__ == "__main__":
    main()
