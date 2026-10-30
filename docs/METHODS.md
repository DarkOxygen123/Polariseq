# Methods and optimizations

The engineering shared by every step, then each algorithm: the published method it is based on and how
this implementation computes it. Code adapted from other projects is credited, with its license, in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Full citations are in [References](#references).

**Contents:** [Optimizations](#optimizations) · [Gene selection](#default-preprocessing) ·
[PCA](#randomized-pca) · [Harmony](#harmony) · [NN-Descent](#nn-descent) · [Leiden](#leiden) ·
[Paris](#paris-hierarchy) · [UMAP](#umap) · [UMAP initialization](#multilevel-spectral-umap-initialization) ·
[Scrublet](#scrublet) · [Marker genes](#marker-genes) · [Gene scores](#gene-scores) ·
[Cell types](#cell-type-annotation) · [Pseudobulk](#pseudobulk-testing) · [PAGA](#cluster-graph) ·
[Pseudotime](#pseudotime) · [Comparing analyses](#comparing-analyses) · [References](#references)

## Optimizations

**A compressed copy of the counts.** Out of core, the counts are copied once into a store on disk that
takes a little over one byte per stored value, against eight uncompressed. Nothing is lost: results
match those from an uncompressed copy, and each part of the store is checked when it is read.

Rust: <a href="../rust/api/polariseq_core/store/"><code>polariseq_core::store</code></a>

**A memory budget.** You set how much memory and how many threads Polariseq may use
(`ps.set_budget`). Each step works through the data in blocks sized to fit, and `ps.plan` tells you the
peak memory of every step before anything runs.

Python: [`ps.set_budget`](api/ps/set_budget.md), [`ps.plan`](api/ps/plan.md)

**Fast file reading.** Opening an `.h5ad` file unpacks its compressed parts on every core at once.

Rust: <a href="../rust/api/polariseq_io/h5ad/"><code>polariseq_io::h5ad</code></a>

**Repeatable arithmetic.** PCA, the neighbor graph and UMAP add their numbers up in a fixed order, so a
run gives the same result every time.

Rust: <a href="../rust/api/polariseq_core/rowblocks/"><code>polariseq_core::rowblocks</code></a>

## Algorithms

### Default preprocessing

*Based on the dispersion-based gene selection of Seurat (Satija et al. 2015), as Scanpy implements it
(`flavor="seurat"`; Wolf et al. 2018). The renormalization follows Scarf (Dhapola et al. 2022).*

Polariseq keeps the 2,000 most variable genes, choosing only among genes detected in more than 1% of the
cells, so that genes seen in a handful of cells do not crowd out the informative ones. Each cell is then
normalized again over the selected genes, and each gene is scaled before PCA.
`ps.set_workflow("scanpy")` switches back to Scanpy's own steps, to compare the two step by step. With
`flavor="scarf"`, genes are ranked the way Scarf ranks them, against a smooth curve (LOWESS; Cleveland
1979) through the least variable genes.

Python: [`ps.pp.highly_variable_genes`](api/pp/highly_variable_genes.md), [`ps.pp.renormalize`](api/pp/renormalize.md) · Rust: <a href="../rust/api/polariseq_core/preprocess/"><code>polariseq_core::preprocess</code></a>, <a href="../rust/api/polariseq_core/lowess/"><code>polariseq_core::lowess</code></a>

### Randomized PCA

*Based on Halko, Martinsson and Tropp (2011), with block Krylov iteration (Musco and Musco 2015).*

Finds the top principal components with a randomized method that needs only a few passes over the data.
The matrix stays sparse the whole time. For the smaller matrix of selected genes, an exact method is
used.

Python: [`ps.pp.pca`](api/pp/pca.md) · Rust: <a href="../rust/api/polariseq_core/pca/"><code>polariseq_core::pca</code></a>, <a href="../rust/api/polariseq_core/rsvd/"><code>polariseq_core::rsvd</code></a>

### Harmony

*Based on Korsunsky et al. (2019), harmony-pytorch and R harmony's documented options.*

Corrects the principal components so that cells from different batches mix while cell types stay apart.
Polariseq's version keeps only one large working array, so memory stays low on large datasets, and it
gives the same result at any thread count. Its corrections differ from harmonypy's no more than
harmonypy's own runs differ from each other with different seeds.

Python: [`ps.pp.harmony_integrate`](api/pp/harmony_integrate.md) · Rust: <a href="../rust/api/polariseq_core/harmony/"><code>polariseq_core::harmony</code></a>

### NN-Descent

*Based on Dong, Moses and Li (2011), as pynndescent implements it.*

Finds each cell's nearest neighbors approximately, which is far faster than comparing every pair of
cells. It is used above 10,000 cells; smaller datasets use an exact search. Memory stays within the
budget, and the graph is the same at any thread count.

Python: [`ps.pp.neighbors`](api/pp/neighbors.md) · Rust: <a href="../rust/api/polariseq_core/nndescent/"><code>polariseq_core::nndescent</code></a>

### Leiden

*Based on Traag, Waltman and van Eck (2019) and their CWTS networkanalysis implementation.*

Groups cells into clusters that are densely connected in the neighbor graph. Polariseq's version works
on plain arrays, which keeps it fast and light on memory, and draws its random numbers from a single
seeded stream, so the same graph and seed always give the same clusters. `method="fast"` runs the
earlier Louvain method (Blondel et al. 2008) on the same graph.

Python: [`ps.tl.leiden`](api/tl/leiden.md) · Rust: <a href="../rust/api/polariseq_core/clustering/"><code>polariseq_core::clustering</code></a>, <a href="../rust/api/polariseq_core/louvain/"><code>polariseq_core::louvain</code></a>

### Paris hierarchy

*Based on Bonald et al. (2018), as scikit-network implements it (Bonald et al. 2020); balanced cut from
Scarf (Dhapola et al. 2022).*

Builds the whole tree of clusters in one pass over the graph, from single cells up to one cluster.
Cutting the tree at different heights gives clusterings at any resolution without clustering again. The
tree matches scikit-network's exactly.

Python: [`ps.tl.paris`](api/tl/paris.md) · Rust: <a href="../rust/api/polariseq_core/hierarchy/"><code>polariseq_core::hierarchy</code></a>

### UMAP

*Based on McInnes, Healy and Melville (2018) and umap-learn (McInnes et al. 2018).*

Draws the cells in two dimensions so that neighbors in the graph stay close. The graph UMAP needs is
built once and shared by its two stages, the starting layout and the optimization. The optimization
runs on all threads; `deterministic=True` runs it on one thread, giving the same layout every time.

Python: [`ps.tl.umap`](api/tl/umap.md) · Rust: <a href="../rust/api/polariseq_core/umap/"><code>polariseq_core::umap</code></a>

### Multilevel spectral UMAP initialization

*Builds on the spectral start of umap-learn and on multiscale eigenvector computation for large graphs
(Koren, Carmel and Harel 2002).*

UMAP refines a starting layout, and a good start keeps related cell types near each other. The standard
start is slow to compute exactly for millions of cells. Polariseq first groups similar cells, computes
the start for the much smaller graph between groups, places each cell at its group's position and then
smooths the positions over the full graph. This takes seconds for millions of cells and keeps the
large-scale structure of the data.

Python: [`ps.tl.umap`](api/tl/umap.md) · Rust: <a href="../rust/api/polariseq_core/umap/"><code>polariseq_core::umap</code></a>

### Scrublet

*Based on Wolock, Lopez and Klein (2019) and its reference implementation; genes are selected by the
v-score of Klein et al. (2015).*

Finds droplets that held two cells. Doublets are simulated by adding up random pairs of cells, and each
real cell is scored by how many simulated doublets are among its nearest neighbors. Polariseq's version
never stores the simulated doublets themselves, only 30 numbers per cell, and scores batches in
parallel: 443,797 cells in 141 batches took 22 s on 8 cores, against 71 s for the original tool on 8
processes.

Python: [`ps.pp.scrublet`](api/pp/scrublet.md) · Rust: <a href="../rust/api/polariseq_core/doublets/"><code>polariseq_core::doublets</code></a>

### Marker genes

*Based on Welch's t-test (Welch 1947), as Scanpy's `rank_genes_groups(method="t-test")` applies it, with
Benjamini–Hochberg adjusted p-values (Benjamini and Hochberg 1995).*

Compares each cluster with the other cells, gene by gene, and adjusts the p-values for the number of
genes tested. One pass over the data collects everything the test needs, in memory or out of core.

Python: [`ps.tl.rank_genes_groups`](api/tl/rank_genes_groups.md) · Rust: <a href="../rust/api/polariseq_core/de/"><code>polariseq_core::de</code></a>

### Gene scores

*After Seurat's module score (Tirosh et al. 2016).*

Scores each cell for a gene set: its average expression of those genes, minus its average over control
genes drawn at random from all the other genes. Seurat's module score draws its control genes from
genes of similar average expression instead.

Python: [`ps.tl.score_genes`](api/tl/score_genes.md)

### Cell type annotation

*Based on ScType (Ianevski, Giri and Aittokallio 2022) and the Cell Marker Accordion (Busarello et al.
2025).*

Names clusters from marker genes in published databases. Every cell is scored for every cell type: a
marker counts more when many sources name it and when few cell types share it, and genes a cell type
should not express count against it. Scoring takes two passes over the data, in memory or out of core.

Python: [`ps.tl.auto_annotate`](api/tl/auto_annotate.md) · Rust: <a href="../rust/api/polariseq_core/annotate/"><code>polariseq_core::annotate</code></a>

### Pseudobulk testing

*Based on edgeR and limma (Robinson, McCarthy and Smyth 2010; Ritchie et al. 2015): the expression filter
of Chen, Lun and Smyth (2016), TMM normalization (Robinson and Oshlack 2010) and limma-trend testing
(Smyth 2004; Law et al. 2014; Phipson et al. 2016).*

Compares conditions within each cell type with the donors as replicates. Counts are added up per donor,
condition and cell type, and the sums are tested as bulk RNA-seq samples, the way edgeR and limma test
them. The results match edgeR's and limma's: the same genes, with p-values within 1e-11, on 350
generated datasets. The raw counts are read once, from memory or from the store on disk.

Python: [`ps.tl.compare_conditions`](api/tl/compare_conditions.md), [`ps.get.pseudobulk`](api/get/pseudobulk.md) · Rust: <a href="../rust/api/polariseq_core/pseudobulk/"><code>polariseq_core::pseudobulk</code></a>

### Cluster graph

*After PAGA (Wolf et al. 2019).*

`ps.tl.paga` connects two clusters by the number of neighbor-graph edges between them, counted in both
directions, divided by the number of cells in the two. PAGA's own measure compares those edges with the
number expected by chance.

Python: [`ps.tl.paga`](api/tl/paga.md) · Rust: <a href="../rust/api/polariseq_core/trajectory/"><code>polariseq_core::trajectory</code></a>

### Pseudotime

`ps.tl.dpt` gives each cell its shortest-path distance from a root cell over the neighbor graph, with the
neighbor distances as edge lengths, scaled so the farthest cell reached is at 1. It is a graph distance;
diffusion pseudotime (Haghverdi et al. 2016) is a different measure.

Python: [`ps.tl.dpt`](api/tl/dpt.md) · Rust: <a href="../rust/api/polariseq_core/trajectory/"><code>polariseq_core::trajectory</code></a>

### Comparing analyses

*Based on the adjusted Rand index (Hubert and Arabie 1985).*

`ps.tl.compare_analyses` measures how far two clusterings of the same cells agree. With `seeds`, it
also reruns each analysis with other seeds, so a real difference can be told from random variation.

Python: [`ps.tl.compare_analyses`](api/tl/compare_analyses.md)

## References

- Benjamini, Y. & Hochberg, Y. Controlling the false discovery rate: a practical and powerful approach to
  multiple testing. *J. R. Stat. Soc. B* 57, 289–300 (1995).
- Blondel, V. D., Guillaume, J.-L., Lambiotte, R. & Lefebvre, E. Fast unfolding of communities in large
  networks. *J. Stat. Mech.* 2008, P10008 (2008).
- Bonald, T., Charpentier, B., Galland, A. & Hollocou, A. Hierarchical graph clustering using node pair
  sampling. Preprint at https://arxiv.org/abs/1806.01664 (2018).
- Bonald, T., de Lara, N., Lutz, Q. & Charpentier, B. Scikit-network: graph analysis in Python.
  *J. Mach. Learn. Res.* 21, 1–6 (2020).
- Busarello, E. et al. Cell Marker Accordion: interpretable single-cell and spatial omics annotation in
  health and disease. *Nat. Commun.* 16, 5399 (2025).
- Chen, Y., Lun, A. T. L. & Smyth, G. K. From reads to genes to pathways: differential expression analysis
  of RNA-Seq experiments using Rsubread and the edgeR quasi-likelihood pipeline. *F1000Research* 5, 1438
  (2016).
- Cleveland, W. S. Robust locally weighted regression and smoothing scatterplots. *J. Am. Stat. Assoc.*
  74, 829–836 (1979).
- Dhapola, P. et al. Scarf enables a highly memory-efficient analysis of large-scale single-cell genomics
  data. *Nat. Commun.* 13, 4616 (2022).
- Dong, W., Moses, C. & Li, K. Efficient k-nearest neighbor graph construction for generic similarity
  measures. In *Proc. 20th International Conference on World Wide Web* 577–586 (ACM, 2011).
- Haghverdi, L., Büttner, M., Wolf, F. A., Buettner, F. & Theis, F. J. Diffusion pseudotime robustly
  reconstructs lineage branching. *Nat. Methods* 13, 845–848 (2016).
- Halko, N., Martinsson, P. G. & Tropp, J. A. Finding structure with randomness: probabilistic algorithms
  for constructing approximate matrix decompositions. *SIAM Rev.* 53, 217–288 (2011).
- Hubert, L. & Arabie, P. Comparing partitions. *J. Classif.* 2, 193–218 (1985).
- Ianevski, A., Giri, A. K. & Aittokallio, T. Fully-automated and ultra-fast cell-type identification
  using specific marker combinations from single-cell transcriptomic data. *Nat. Commun.* 13, 1246 (2022).
- Klein, A. M. et al. Droplet barcoding for single-cell transcriptomics applied to embryonic stem cells.
  *Cell* 161, 1187–1201 (2015).
- Koren, Y., Carmel, L. & Harel, D. ACE: a fast multiscale eigenvectors computation for drawing huge
  graphs. In *Proc. IEEE Symposium on Information Visualization* 137–144 (IEEE, 2002).
- Korsunsky, I. et al. Fast, sensitive and accurate integration of single-cell data with Harmony.
  *Nat. Methods* 16, 1289–1296 (2019).
- Law, C. W., Chen, Y., Shi, W. & Smyth, G. K. voom: precision weights unlock linear model analysis tools
  for RNA-seq read counts. *Genome Biol.* 15, R29 (2014).
- McInnes, L., Healy, J. & Melville, J. UMAP: uniform manifold approximation and projection for dimension
  reduction. Preprint at https://arxiv.org/abs/1802.03426 (2018).
- McInnes, L., Healy, J., Saul, N. & Großberger, L. UMAP: Uniform Manifold Approximation and Projection.
  *J. Open Source Softw.* 3, 861 (2018).
- Musco, C. & Musco, C. Randomized block Krylov methods for stronger and faster approximate singular value
  decomposition. In *Advances in Neural Information Processing Systems* 28 (2015).
- Phipson, B., Lee, S., Majewski, I. J., Alexander, W. S. & Smyth, G. K. Robust hyperparameter estimation
  protects against hypervariable genes and improves power to detect differential expression.
  *Ann. Appl. Stat.* 10, 946–963 (2016).
- Ritchie, M. E. et al. limma powers differential expression analyses for RNA-sequencing and microarray
  studies. *Nucleic Acids Res.* 43, e47 (2015).
- Robinson, M. D., McCarthy, D. J. & Smyth, G. K. edgeR: a Bioconductor package for differential
  expression analysis of digital gene expression data. *Bioinformatics* 26, 139–140 (2010).
- Robinson, M. D. & Oshlack, A. A scaling normalization method for differential expression analysis of
  RNA-seq data. *Genome Biol.* 11, R25 (2010).
- Satija, R., Farrell, J. A., Gennert, D., Schier, A. F. & Regev, A. Spatial reconstruction of single-cell
  gene expression data. *Nat. Biotechnol.* 33, 495–502 (2015).
- Smyth, G. K. Linear models and empirical Bayes methods for assessing differential expression in
  microarray experiments. *Stat. Appl. Genet. Mol. Biol.* 3, Article 3 (2004).
- Tirosh, I. et al. Dissecting the multicellular ecosystem of metastatic melanoma by single-cell RNA-seq.
  *Science* 352, 189–196 (2016).
- Traag, V. A., Waltman, L. & van Eck, N. J. From Louvain to Leiden: guaranteeing well-connected
  communities. *Sci. Rep.* 9, 5233 (2019).
- Welch, B. L. The generalization of "Student's" problem when several different population variances are
  involved. *Biometrika* 34, 28–35 (1947).
- Wolf, F. A., Angerer, P. & Theis, F. J. SCANPY: large-scale single-cell gene expression data analysis.
  *Genome Biol.* 19, 15 (2018).
- Wolf, F. A. et al. PAGA: graph abstraction reconciles clustering with trajectory inference through a
  topology preserving map of single cells. *Genome Biol.* 20, 59 (2019).
- Wolock, S. L., Lopez, R. & Klein, A. M. Scrublet: computational identification of cell doublets in
  single-cell transcriptomic data. *Cell Syst.* 8, 281–291 (2019).
