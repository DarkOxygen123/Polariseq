"""7 - Trying settings, keeping analyses, and handing results to other tools.

Run:  python examples/07_settings_saving_exporting.py

With the settings below unchanged it uses the 10x Genomics PBMC 3k dataset
(downloaded once, about 8 MB). What it shows:

1. Several analyses of the same cells with different settings, from one
   reading of the data (``adata.from_counts``): fewer or more genes, another
   clustering resolution, and genes of your own kept in the selection
   whatever the ranking chooses (``include=``).
2. How far those analyses agree (``ps.tl.compare_analyses``), set against how
   far one analysis agrees with itself under another random seed: a
   difference smaller than that is noise.
3. Each analysis saved in the project (``adata.save``) and one opened again
   later (``ps.load``) to continue: the counts are pointed to, not copied.
4. The chosen analysis written as an ``.h5ad`` for Scanpy, cellxgene or R
   (``adata.write_h5ad``), with the raw counts alongside.

Figures and tables go to OUTPUT; the project folder keeps the analyses.
"""

from pathlib import Path

import polariseq as ps
from _data import load

# ---- settings ------------------------------------------------------------
INPUT = ""            # your .h5ad / 10x .h5 / 10x folder; "" = download PBMC 3k
OUTPUT = Path(__file__).resolve().parent / "output" / "settings_saving_exporting"
PROJECT = Path(__file__).resolve().parent / "output" / "project_07"
RAM_GB = 4
THREADS = 4
MIN_GENES = 200
# Genes you study: kept in every analysis named "..._my_genes", even when the
# ranking would leave them out (here: markers of rare blood cell types).
MY_GENES = ["FCER1A", "PPBP", "CD8A", "FCGR3A", "GNLY"]
# Each analysis: (genes selected, Leiden resolution, genes kept whatever the ranking).
SETTINGS = {
    "hvg1000": (1000, 1.0, None),
    "hvg2000": (2000, 1.0, None),
    "hvg2000_res0.5": (2000, 0.5, None),
    "hvg1000_my_genes": (1000, 1.0, MY_GENES),
}
# --------------------------------------------------------------------------


def analyze(b, n_genes: int, resolution: float, include) -> None:
    """The steps after quality control, the same for every analysis but its settings."""
    ps.pp.filter_genes(b, min_cells=3)
    ps.pp.normalize_total(b, target_sum=1e4)
    ps.pp.log1p(b)
    ps.pp.highly_variable_genes(b, n_top_genes=n_genes, subset=True, include=include)
    ps.pp.renormalize(b)
    ps.pp.pca(b, n_components=30, scale=True, seed=0)
    ps.pp.neighbors(b, n_neighbors=15, seed=0)
    ps.tl.leiden(b, resolution=resolution, seed=0)
    ps.tl.rank_genes_groups(b, groupby="leiden")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ps.project(PROJECT)
    ps.set_budget(ram_gb=RAM_GB, threads=THREADS)

    # 1. The data as an .h5ad file. A saved analysis points to the file its
    #    counts came from and reads them from there when it is opened again,
    #    which it does for .h5ad files; a 10x folder is written as one, once.
    source = Path(INPUT) if INPUT else PROJECT / "data" / "pbmc3k.h5ad"
    if source.suffix != ".h5ad":
        source = PROJECT / "data" / (source.stem + ".h5ad")
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        load(INPUT).write_h5ad(source)          # the counts as read, no anndata needed
    adata = ps.read_h5ad(str(source), mode="memory")

    # 2. Quality control once; every analysis below starts from these cells.
    ps.pp.calculate_qc_metrics(adata)
    ps.pp.filter_cells(adata, min_genes=MIN_GENES)
    print(f"{adata.n_obs:,} cells after quality control\n")

    # 3. One analysis per setting, each from the raw counts of the same cells,
    #    each keeping its results in its own folder, each saved.
    runs = {}
    for name, (n_genes, resolution, include) in SETTINGS.items():
        b = adata.from_counts(name)
        analyze(b, n_genes, resolution, include)
        b.save(name, overwrite=True)
        runs[name] = b
        added = b.uns["hvg"].get("include_added")
        print(f"{name}: {b.n_vars:,} genes, {b.uns['leiden']['n_communities']} clusters"
              + (f" (added to the selection: {', '.join(added)})" if added else ""))

    # 4. How far do they agree, and is that more than the run-to-run variation?
    cmp = ps.tl.compare_analyses(runs, seeds=2)
    print("\n" + repr(cmp))
    which = cmp.matches[("hvg1000", "hvg2000_res0.5")]
    print("\nwhich cluster of hvg1000 went where at resolution 0.5:")
    print(which.head(10).to_string())
    which.to_csv(OUTPUT / "cluster_matches.csv")
    cmp.ari.to_csv(OUTPUT / "agreement_ari.csv")

    # 5. Later (another day, another process): open one and continue.
    chosen = ps.load(PROJECT, "hvg1000_my_genes")
    ps.tl.umap(chosen, seed=0)
    ps.pl.umap(chosen, color=["leiden", *MY_GENES[:3]], show=False,
               save=str(OUTPUT / "umap_my_genes.png"))
    for name in runs:            # the comparison plot needs one embedding
        if name != "hvg1000_my_genes":
            runs[name].obsm["X_umap"] = chosen.obsm["X_umap"]
    runs["hvg1000_my_genes"] = chosen
    ps.pl.compare_analyses(runs, show=False, save=str(OUTPUT / "compare_analyses.png"))
    print(f"\ntop markers of {chosen.analysis}:")
    print(ps.get.markers(chosen, n=3, wide=True).to_string())

    # 6. For other tools: X as the analysis left it, the raw counts in
    #    layers["counts"], and uns["polariseq"] pointing back to the source.
    path = chosen.write_h5ad(OUTPUT / "pbmc3k_analysis.h5ad", counts=True)
    print(f"\nwritten {path} ({path.stat().st_size / 1e6:.1f} MB)")
    sizes = ps.project(PROJECT).status()
    print("project:", {k: f"{v / 1e6:.1f} MB" for k, v in sizes.items() if v})


if __name__ == "__main__":
    main()
