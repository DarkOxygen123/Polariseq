"""5 — Comparing conditions: control against treated, within each cell type.

Run:  python examples/05_comparing_conditions.py

Data: Kang, H. M., Subramaniam, M., Targ, S. et al. Multiplexed droplet
single-cell RNA-sequencing using natural genetic variation. *Nature
Biotechnology* 36, 89-94 (2018). GEO accession GSE96583. Peripheral blood
mononuclear cells from eight donors, cultured for six hours without (control)
and with interferon-beta, each condition captured in its own 10x run. The
authors identified each cell's donor, and the droplets holding two donors'
cells, from genotypes (demuxlet), and annotated the cell types.

What it shows:

1. The two runs read as one dataset, each cell keeping its condition and
   the authors' annotations (donor, cell type, doublet call).
2. Doublet detection per capture, checked against the authors' genotype calls.
3. One joint analysis: quality control, integration of the conditions,
   clusters named from marker genes.
4. The comparison itself, per cell type, with the donors as replicates: the
   interferon response in every cell type.
5. The figures: conditions on the UMAP, split violins, a volcano plot, fold
   changes across cell types, and a split dot plot.

Downloads about 60 MB from GEO on the first run (cached in examples/data/)
and needs ``anndata`` to write the two runs as ``.h5ad`` files. Figures and
the result table go to examples/output/.
"""

from __future__ import annotations

import gzip
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import polariseq as ps

OUT = Path(__file__).parent / "output" / "comparing_conditions"
DATA = Path(__file__).parent / "data" / "kang_2018"
GEO = "https://ftp.ncbi.nlm.nih.gov/geo"
FILES = {
    "GSM2560248_2.1.mtx.gz": f"{GEO}/samples/GSM2560nnn/GSM2560248/suppl/",
    "GSM2560248_barcodes.tsv.gz": f"{GEO}/samples/GSM2560nnn/GSM2560248/suppl/",
    "GSM2560249_2.2.mtx.gz": f"{GEO}/samples/GSM2560nnn/GSM2560249/suppl/",
    "GSM2560249_barcodes.tsv.gz": f"{GEO}/samples/GSM2560nnn/GSM2560249/suppl/",
    "GSE96583_batch2.genes.tsv.gz": f"{GEO}/series/GSE96nnn/GSE96583/suppl/",
    "GSE96583_batch2.total.tsne.df.tsv.gz": f"{GEO}/series/GSE96nnn/GSE96583/suppl/",
}

# Canonical markers of the blood cell types in this dataset.
MARKERS = {
    "CD4 T cells": ["IL7R", "CD3E", "CD3D", "LDHB"],
    "CD8 T cells": ["CD8A", "CD8B", "CD3D"],
    "NK cells": ["GNLY", "NKG7", "KLRD1"],
    "B cells": ["MS4A1", "CD79A", "CD79B"],
    "CD14+ Monocytes": ["CD14", "LYZ", "S100A9"],
    "FCGR3A+ Monocytes": ["FCGR3A", "MS4A7"],
    "Dendritic cells": ["FCER1A", "HLA-DQA1", "CST3"],
    "Megakaryocytes": ["PPBP", "PF4"],
}


def fetch_kang() -> dict[str, str]:
    """The authors' two runs as two ``.h5ad`` files, with their annotations."""
    runs = {"control": DATA / "control.h5ad", "IFN-beta": DATA / "ifn_beta.h5ad"}
    if all(p.exists() for p in runs.values()):
        return {k: str(v) for k, v in runs.items()}
    import anndata as ad
    import scipy.io
    import scipy.sparse as sp

    DATA.mkdir(parents=True, exist_ok=True)
    for name, base in FILES.items():
        if not (DATA / name).exists():
            print(f"downloading {name} ...", flush=True)
            req = urllib.request.Request(base + name, headers={"User-Agent": "polariseq-example/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r, open(DATA / name, "wb") as fh:
                fh.write(r.read())
    genes = pd.read_csv(DATA / "GSE96583_batch2.genes.tsv.gz", sep="\t", header=None,
                        names=["gene_ids", "symbol"])
    meta = pd.read_csv(DATA / "GSE96583_batch2.total.tsne.df.tsv.gz", sep="\t", index_col=0)
    for (gsm, tag, stim), (label, out) in zip(
            [("GSM2560248", "2.1", "ctrl"), ("GSM2560249", "2.2", "stim")], runs.items(),
            strict=True):
        with gzip.open(DATA / f"{gsm}_{tag}.mtx.gz", "rb") as fh:
            x = sp.csr_matrix(scipy.io.mmread(fh).T, dtype=np.float32)  # cells x genes
        barcodes = pd.read_csv(DATA / f"{gsm}_barcodes.tsv.gz", header=None)[0].tolist()
        part = meta[meta["stim"] == stim]
        # The authors' table (written by R) makes the barcodes both runs share
        # unique by turning "-1" into "-11" in the second run's rows.
        part.index = part.index.str.replace(r"-11$", "-1", regex=True)
        assert list(part.index) == barcodes, f"{label}: annotations out of order"
        obs = pd.DataFrame({
            "donor": part["ind"].astype(str).values,
            "published_type": part["cell"].fillna("unassigned").values,
            "demuxlet": part["multiplets"].values,
        }, index=[f"{b}-{stim}" for b in barcodes]).astype("category")
        var = pd.DataFrame({"gene_ids": genes["gene_ids"].values}, index=genes["symbol"].values)
        a = ad.AnnData(X=x, obs=obs, var=var)
        a.var_names_make_unique()
        a.write_h5ad(out)
    return {k: str(v) for k, v in runs.items()}


def name_clusters(adata, markers: dict[str, list[str]], key: str = "leiden") -> None:
    """Name each cluster after the marker set most specific to it.

    Each cluster's counts are summed (``ps.get.pseudobulk``, read back from
    the files: the matrix is normalized by now), taken as log CPM, and each
    marker is standardized across clusters; the set with the highest mean
    wins. Comparing clusters rather than cells keeps a small set (CD8A,
    CD8B) from losing to a large one that shares genes (the NK set, which
    cytotoxic CD8 T cells also express).
    """
    pb = ps.get.pseudobulk(adata, key)
    counts = pd.DataFrame(np.asarray(pb.X), index=[str(c) for c in pb.obs[key]],
                          columns=list(pb.var_names))
    logcpm = np.log1p(counts.div(counts.sum(axis=1), axis=0) * 1e6)
    z = (logcpm - logcpm.mean()) / logcpm.std(ddof=0).replace(0, 1)
    scores = pd.DataFrame({t: z[[g for g in genes if g in z.columns]].mean(axis=1)
                           for t, genes in markers.items()})
    best = scores.idxmax(axis=1)
    adata.obs["cell_type"] = pd.Categorical(
        best.loc[np.asarray(adata.obs[key]).astype(str)].to_numpy(), categories=list(markers))


def main() -> None:
    runs = fetch_kang()
    OUT.mkdir(parents=True, exist_ok=True)
    ps.set_budget(ram_gb=4, threads=4)
    ps.project(OUT / "project")  # where per-cell results are kept, memory-mapped
    t0 = time.perf_counter()

    # 1. Two runs, one dataset: each cell keeps its condition and annotations.
    adata = ps.read_h5ad(runs, batch_key="condition")
    print(adata)
    print("annotations:", list(adata.obs))

    # 2. Doublets form within one capture, so each run is scored on its own.
    ps.pp.scrublet(adata, batch_key="condition")
    ps.pl.scrublet_score_distribution(adata, show=False, save=str(OUT / "doublet_scores.png"))
    genotype = np.asarray(adata.obs["demuxlet"] == "doublet")
    called = np.asarray(adata.obs["predicted_doublet"])
    print(f"doublets called: {called.sum():,}; of the {genotype.sum():,} the authors found "
          f"by genotype, {called[genotype].mean():.0%} were called")
    # Keep the cells neither method calls a doublet.
    adata = adata[~called & np.asarray(adata.obs["demuxlet"] == "singlet")]

    # 3. One joint analysis.
    ps.pp.calculate_qc_metrics(adata)
    ps.pp.filter_cells(adata, min_genes=200)
    ps.pp.filter_genes(adata, min_cells=3)
    ps.pp.normalize_total(adata, target_sum=1e4)
    ps.pp.log1p(adata)
    ps.pp.highly_variable_genes(adata, n_top_genes=2000)
    ps.pp.pca(adata, n_components=30)
    ps.pp.harmony_integrate(adata, batch_key="condition")  # align the conditions
    ps.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca_harmony")
    ps.tl.leiden(adata, resolution=0.6)
    ps.tl.umap(adata)
    name_clusters(adata, MARKERS)
    agree = np.mean(np.asarray(adata.obs["cell_type"]) == np.asarray(adata.obs["published_type"]))
    print(f"cell types named from markers agree with the authors' for {agree:.0%} of cells")
    ps.pl.umap(adata, color="cell_type", split_by="condition", show=False,
               save=str(OUT / "umap_by_condition.png"))

    # 4. The comparison: donors are the replicates, and each donor was
    #    measured in both conditions, so the test pairs them.
    res = ps.tl.compare_conditions(adata, "condition", "control", groupby="cell_type",
                                   sample_key="donor")
    res.to_csv(OUT / "ifn_beta_vs_control.csv", index=False)
    print(adata.uns["compare_conditions"]["groups"][["group", "samples", "paired"]]
          .to_string(index=False))
    sig = res[res["adj_p_value"] < 0.05]
    print("genes changed (adjusted p < 0.05), by cell type:")
    print(sig.groupby("cell_type", observed=True).size().to_string())
    for cell_type in ("CD14+ Monocytes", "CD4 T cells"):
        top = sig[sig["cell_type"] == cell_type].head(8)
        print(f"most significant in {cell_type}:", ", ".join(top["gene"]))

    # 5. Figures.
    ps.pl.violin(adata, ["ISG15", "IFI6", "CXCL10"], groupby="cell_type", split_by="condition",
                 show=False, save=str(OUT / "violins.png"))
    ps.pl.volcano(res, "CD14+ Monocytes", show=False, save=str(OUT / "volcano_cd14_monocytes.png"))
    ps.pl.fold_changes(res, n_genes=4, show=False, save=str(OUT / "fold_changes.png"))
    ps.pl.dotplot(adata, ["ISG15", "IFIT1", "CXCL10", "CCL8", "LYZ", "CD3E", "MS4A1", "GNLY"],
                  groupby="cell_type", split_by="condition", show=False,
                  save=str(OUT / "dotplot.png"))

    # The summed counts, for DESeq2 or edgeR users.
    pb = ps.get.pseudobulk(adata, "donor", groupby="cell_type", condition_key="condition",
                           min_cells=10)
    pd.DataFrame(np.asarray(pb.X), index=pb.obs_names, columns=pb.var_names).to_csv(
        OUT / "pseudobulk_counts.csv")
    print(f"done in {time.perf_counter() - t0:.0f} s; figures and tables in {OUT}")


if __name__ == "__main__":
    main()
