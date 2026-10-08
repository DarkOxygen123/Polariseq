//! Spill-to-SSD and the disk-backed preprocessing pipeline.
//!
//! This is the path for a matrix that does not fit in RAM. Two rules make
//! it honest:
//!
//! 1. **Nothing is ever fully materialised.** [`spill_h5ad`] streams row
//!    blocks out of the file and appends them to three flat files as it
//!    goes; peak memory is one block, not one matrix. (The function it
//!    replaces read the whole matrix, then built a second complete copy as
//!    a `Vec<u8>` before writing a byte of it.)
//! 2. **The kernels are the same ones the in-memory path uses.** Once
//!    spilled, the matrix is a `DiskBackedCSR`, which is a
//!    [`RowBlocks`](polariseq_core::rowblocks::RowBlocks) source, so QC,
//!    gene statistics, HVG and PCA run through the identical code and
//!    produce identical numbers.

use std::path::{Path, PathBuf};

use polariseq_core::file_matrix::FileCsr;
use polariseq_core::preprocess::{
    gene_mean_var_blocks, gene_stats_detected_blocks, hvg_seurat, qc_metrics_blocks,
    row_totals_blocks, FilteredRows, GeneStats, HvgResult, NormalizedView, QcMetrics,
};
use polariseq_core::rowblocks::RowBlocks;
use polariseq_core::store::{Layout, StoreWriter};
use polariseq_core::{
    pca::{pca_blocks, pca_scaled_blocks},
    PcaOptions,
};

use crate::stream::H5adStream;
use crate::{IoError, Result};

/// Rows read and written per block while spilling, when the caller does not
/// say otherwise. `ps.plan()` derives a budget-appropriate value; this is
/// only the fallback for callers that have no budget to hand.
const DEFAULT_SPILL_ROWS: usize = 32_768;

/// Where a spilled matrix lives and how big it is.
#[derive(Debug, Clone)]
pub struct Spilled {
    /// File prefix; the triple is `{prefix}.data/.indices/.indptr`.
    pub prefix: PathBuf,
    /// Rows (cells).
    pub n_rows: usize,
    /// Columns (genes).
    pub n_cols: usize,
    /// Stored nonzeros.
    pub nnz: usize,
}

impl Spilled {
    /// Open the spilled triple as a block source.
    ///
    /// Reads blocks explicitly rather than memory-mapping: mapped pages are
    /// charged to this process and accumulate over a full-matrix pass, which
    /// is the opposite of what an out-of-core path is for. See
    /// [`FileCsr`] for the measurements.
    ///
    /// # Errors
    /// Returns [`IoError::Io`] if the files cannot be opened.
    pub fn open(&self) -> Result<FileCsr> {
        self.open_with(super::spill::DEFAULT_SPILL_ROWS)
    }

    /// Open with an explicit cap on rows per block — the budget's lever over
    /// the footprint of every pass that follows.
    ///
    /// # Errors
    /// Returns [`IoError::Io`] if the files cannot be opened.
    pub fn open_with(&self, block_rows: usize) -> Result<FileCsr> {
        FileCsr::open_at(&self.prefix, self.n_rows, self.n_cols)
            .map(|f| f.with_max_block_rows(block_rows))
            .map_err(IoError::Io)
    }
}

/// Stream `path` into `{prefix}.data/.indices/.indptr` without holding the
/// matrix in memory.
///
/// # Errors
/// Returns [`IoError`] on any read or write failure.
pub fn spill_h5ad(path: &str, prefix: &Path) -> Result<Spilled> {
    spill_h5ad_with(path, prefix, DEFAULT_SPILL_ROWS)
}

/// [`spill_h5ad`] with an explicit block size — the memory this pass uses is
/// one block, so this is the knob a budget sets.
///
/// # Errors
/// Returns [`IoError`] on any read or write failure.
pub fn spill_h5ad_with(path: &str, prefix: &Path, block_rows: usize) -> Result<Spilled> {
    let block_rows = block_rows.max(1);
    if let Some(dir) = prefix.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let stream = H5adStream::open(path)?;
    let (n_rows, n_cols) = (stream.n_rows(), stream.n_cols());

    // 64-bit row pointers: a store has no 2^32 ceiling on stored values.
    let mut writer = StoreWriter::create(prefix, Layout::for_new_stores())?;

    // Blocks arrive in row order (the reader is sequential), and the writes
    // must stay in that order, so this pass is deliberately serial: the
    // work here is I/O, not arithmetic.
    let mut r0 = 0;
    let mut lens: Vec<u32> = Vec::new();
    while r0 < n_rows {
        let r1 = (r0 + block_rows).min(n_rows);
        let blk = crate::lazy::read_rows_any(path, r0, r1)?;
        lens.clear();
        lens.extend((0..blk.nrows).map(|i| blk.indptr[i + 1] - blk.indptr[i]));
        writer.push(&blk.data, &blk.indices, &lens)?;
        r0 = r1;
    }
    let nnz = writer.finish()?;

    Ok(Spilled {
        prefix: prefix.to_path_buf(),
        n_rows,
        n_cols,
        nnz: nnz as usize,
    })
}

/// Stream several `.h5ad` files, one after another, into one block store:
/// the concatenation of their cells over one set of genes, never held in
/// memory and never written as a merged file (see [`crate::merge`]).
///
/// `gene_maps[k][j]` is the merged column of gene `j` of file `k`, or
/// [`crate::merge::DROPPED`].
///
/// # Errors
/// Returns [`IoError`] on read or write failure, or if a map does not match
/// its file's genes.
pub fn spill_h5ad_many(
    paths: &[String],
    gene_maps: &[Vec<u32>],
    n_cols: usize,
    prefix: &Path,
    block_rows: usize,
) -> Result<Spilled> {
    if let Some(dir) = prefix.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut writer = StoreWriter::create(prefix, Layout::for_new_stores())?;
    let n_rows =
        crate::merge::for_each_block(paths, gene_maps, n_cols, block_rows, |d, ix, lens| {
            writer.push(d, ix, lens)?;
            Ok(())
        })?;
    let nnz = writer.finish()?;
    Ok(Spilled {
        prefix: prefix.to_path_buf(),
        n_rows,
        n_cols,
        nnz: nnz as usize,
    })
}

/// Options for [`disk_backed_preprocess`].
#[derive(Debug, Clone)]
pub struct DiskPipelineOptions {
    /// Target counts per cell for `normalize_total`.
    pub target_sum: f32,
    /// Highly-variable genes to keep.
    pub n_hvg: usize,
    /// Principal components.
    pub n_pcs: usize,
    /// Minimum detected genes for a cell to be kept (scanpy's
    /// `filter_cells(min_genes=...)`; 0 keeps every cell). Without this the
    /// out-of-core path answers a different question than the in-RAM
    /// pipeline on the same parameters — the telemetry harness caught it
    /// comparing the two arms' outputs.
    pub min_genes: u32,
    /// Maximum detected genes for a cell to be kept (`u32::MAX`: no limit).
    pub max_genes: u32,
    /// Minimum total counts for a cell to be kept.
    pub min_counts: f32,
    /// Maximum total counts for a cell to be kept (infinite: no limit).
    pub max_counts: f32,
    /// Maximum percentage of a cell's counts from mitochondrial genes
    /// (infinite: no limit; a finite value needs `mito`).
    pub max_pct_mt: f32,
    /// Which genes are mitochondrial, one flag per gene (`None`: no such
    /// genes, so no percentage is computed).
    pub mito: Option<Vec<bool>>,
    /// Minimum number of kept cells in which a gene is detected for it to be
    /// kept (scanpy's `filter_genes(min_cells=...)`, counted after the cell
    /// filter as scanpy counts it; 0 keeps every gene).
    pub min_cells: u32,
    /// Random seed (unused on the exact Gram path; kept for the wide path).
    pub seed: u64,
    /// Directory for the spill files.
    pub spill_dir: PathBuf,
    /// Rows per block for the spill and for every streamed pass. This is the
    /// budget's main lever over peak memory on the out-of-core path: the
    /// working set is a small multiple of one block.
    pub block_rows: usize,
    /// Rank genes only among those detected in more than this share of the
    /// kept cells (0: every gene, Scanpy's seurat ranking). Polariseq's
    /// workflow uses 0.01.
    pub hvg_floor_frac: f32,
    /// After the genes are selected, normalize each cell again over them to
    /// this total (then log), as Scarf does; `None` keeps the totals over all
    /// genes.
    pub renormalize: Option<f32>,
    /// Standardize each gene before PCA (scanpy's `pp.scale`, never formed).
    pub scale_pcs: bool,
}

impl Default for DiskPipelineOptions {
    fn default() -> Self {
        Self {
            target_sum: 1e4,
            n_hvg: 2000,
            n_pcs: 50,
            min_genes: 0,
            max_genes: u32::MAX,
            min_counts: 0.0,
            max_counts: f32::INFINITY,
            max_pct_mt: f32::INFINITY,
            mito: None,
            min_cells: 0,
            seed: 0,
            spill_dir: std::env::temp_dir().join("polariseq_spill"),
            block_rows: DEFAULT_SPILL_ROWS,
            hvg_floor_frac: 0.0,
            renormalize: None,
            scale_pcs: false,
        }
    }
}

/// What the disk-backed pass produced.
#[derive(Debug)]
pub struct DiskPipelineResult {
    /// PCA embedding, row-major `n_rows × n_pcs` — kept rows only.
    pub embedding: Vec<f32>,
    /// Variance ratio per component.
    pub variance_ratio: Vec<f32>,
    /// Components actually returned.
    pub n_components: usize,
    /// QC metrics from the raw counts. `n_counts`/`n_genes` cover the rows
    /// of `embedding` (the kept cells); the per-gene fields cover all input
    /// genes, as scanpy computes them before filtering.
    pub qc: QcMetrics,
    /// HVG selection (mask over the original genes).
    pub hvg: HvgResult,
    /// Rows in the embedding (= cells that passed `min_genes`).
    pub n_rows: usize,
    /// Rows in the input matrix, before `min_genes` filtering.
    pub n_rows_input: usize,
    /// Which input rows survived `min_genes`, as a mask over `n_rows_input`.
    ///
    /// Callers need this to line the embedding back up with the file's cell
    /// barcodes: the disk path filters, so row *i* of `embedding` is not
    /// input row *i*. Without it, anything comparing this path's output to
    /// another implementation's has to assume the orders match, which is
    /// exactly the assumption that hides a filtering bug.
    pub cell_keep: Vec<bool>,
    /// Percentage of counts from mitochondrial genes for the kept cells
    /// (`None` when no mitochondrial mask was given).
    pub pct_counts_mt: Option<Vec<f32>>,
    /// Which genes passed `min_cells`, one flag per gene of the input. The
    /// removed genes take no part in normalisation, gene selection or PCA.
    pub gene_keep: Vec<bool>,
    /// Genes in the original matrix.
    pub n_cols: usize,
    /// Nonzeros spilled.
    pub nnz: usize,
}

/// The full preprocessing pipeline on a matrix too large for RAM:
/// spill → QC → `normalize_total` → `log1p` → HVG → PCA.
///
/// Every step reads or rewrites the memory-mapped spill; the only things
/// held in RAM are per-gene vectors (`n_cols`) and the final embedding
/// (`n_rows × n_pcs`). Normalisation and `log1p` are applied **in place** on
/// the mapped values, and the HVG subset is written to a second, much
/// smaller spill (2 000 of ~30 000 genes) which PCA then reads.
///
/// The result matches what the in-memory pipeline produces on the same
/// file, because it is the same code on a different block source.
///
/// # Errors
/// Returns [`IoError`] on read/write failure, or if PCA fails.
#[allow(clippy::too_many_lines)] // one pass per pipeline step, in order
pub fn disk_backed_preprocess(
    path: &str,
    opts: &DiskPipelineOptions,
) -> Result<DiskPipelineResult> {
    let pid = std::process::id();
    let raw_prefix = opts.spill_dir.join(format!("raw_{pid}"));
    let hvg_prefix = opts.spill_dir.join(format!("hvg_{pid}"));

    let trace = std::env::var_os("POLARISEQ_TRACE").is_some();
    let t0 = std::time::Instant::now();
    macro_rules! phase {
        ($name:expr) => {
            if trace {
                eprintln!(
                    "[spill] {:>22} done at {:6.2}s",
                    $name,
                    t0.elapsed().as_secs_f64()
                );
            }
        };
    }

    // 1. Spill the raw counts (streaming; never materialised).
    let spilled = spill_h5ad_with(path, &raw_prefix, opts.block_rows)?;
    phase!("spill_h5ad");
    // The raw spill is this call's scratch: delete it when `db` goes out of
    // scope, as `select_columns` already does for the HVG spill. Until
    // 2026-09-29 it was never deleted, so every run left 8 bytes per stored
    // value behind (19.6 GB for the fetal atlas) and repeated runs filled
    // the disk.
    let db = spilled.open_with(opts.block_rows)?.delete_on_drop(true);
    phase!("open");

    // 2. QC on the raw counts, with the mitochondrial counts when asked for.
    let qc_all = qc_metrics_blocks(&db, opts.mito.as_deref());
    phase!("qc_metrics");

    // The mitochondrial percentage exactly as the in-memory
    // `calculate_qc_metrics` computes it: f64 division, a zero total divided
    // by one, stored as f32.
    let pct_all: Option<Vec<f32>> = qc_all.mt_counts.as_ref().map(|mt| {
        mt.iter()
            .zip(&qc_all.n_counts)
            .map(|(&m, &t)| {
                let t = if t == 0.0 { 1.0 } else { f64::from(t) };
                (f64::from(m) / t * 100.0) as f32
            })
            .collect()
    });
    if opts.max_pct_mt.is_finite() && pct_all.is_none() {
        return Err(IoError::InvalidLayout(
            "max_pct_mt needs mitochondrial genes, and none were marked".into(),
        ));
    }

    // 2b. filter_cells: decide which rows survive, then filter everything
    //     downstream, with the thresholds and comparisons of the in-memory
    //     `filter_cells`. Row totals are row-local, so normalising before or
    //     after the filter is identical for the kept rows — only the gene
    //     statistics and PCA must see the filtered stream, which
    //     `FilteredRows` provides without rewriting the spill.
    let keep: Vec<bool> = (0..qc_all.n_genes.len())
        .map(|i| {
            let (c, g) = (qc_all.n_counts[i], qc_all.n_genes[i]);
            let mt_ok = match (&pct_all, opts.max_pct_mt.is_finite()) {
                (Some(p), true) => p[i] <= opts.max_pct_mt,
                _ => true,
            };
            g >= opts.min_genes
                && g <= opts.max_genes
                && c >= opts.min_counts
                && c <= opts.max_counts
                && mt_ok
        })
        .collect();

    // 2c. filter_genes(min_cells), counted over the kept cells as scanpy
    //     counts it after filter_cells. One more pass, only when asked for.
    let gene_keep: Option<Vec<bool>> = if opts.min_cells > 0 {
        let n_cells_kept = qc_metrics_blocks(&FilteredRows::new(&db, &keep), None).gene_n_cells;
        let mask: Vec<bool> = n_cells_kept.iter().map(|&n| n >= opts.min_cells).collect();
        if !mask.iter().any(|&k| k) {
            return Err(IoError::InvalidLayout(format!(
                "min_cells={} removes every gene",
                opts.min_cells
            )));
        }
        phase!("filter_genes");
        (!mask.iter().all(|&k| k)).then_some(mask)
    } else {
        None
    };

    // The normalisation scale. Without a gene filter the QC totals are the
    // row totals; with one, each row is totalled over the kept genes, as
    // `normalize_total` would total the gene-filtered matrix.
    let scale = match &gene_keep {
        None => NormalizedView::scale_from_totals(&qc_all.n_counts, opts.target_sum),
        Some(genes) => {
            let totals = row_totals_blocks(&db, Some(genes));
            phase!("row_totals");
            NormalizedView::scale_from_totals(&totals, opts.target_sum)
        }
    };
    let pct_counts_mt = pct_all.map(|p| {
        p.iter()
            .zip(&keep)
            .filter(|(_, &k)| k)
            .map(|(&v, _)| v)
            .collect()
    });
    let qc = QcMetrics {
        n_counts: keep
            .iter()
            .zip(&qc_all.n_counts)
            .filter(|(&k, _)| k)
            .map(|(_, &c)| c)
            .collect(),
        n_genes: keep
            .iter()
            .zip(&qc_all.n_genes)
            .filter(|(&k, _)| k)
            .map(|(_, &g)| g)
            .collect(),
        ..qc_all
    };

    // 3. normalize_total + log1p as a *view*, not a rewrite. Mutating the
    //    mapped values in place would dirty every page of the matrix and
    //    make the whole thing resident — which is what this path exists to
    //    avoid. The transform is applied per block as it is read instead.
    //    `FilteredRows` wraps *outside* the view: the scale vector is
    //    indexed by original row, the filter emits compacted offsets.
    let view = NormalizedView::new(&db, scale.clone(), true);
    let normalized = FilteredRows::new(&view, &keep);

    // 4. HVG on the log-normalised values (Seurat flavour: statistics of
    //    expm1(x), i.e. of the normalised counts).
    //    With a gene filter, selection runs on the kept genes alone (their
    //    mean bins depend on which genes are present) and maps back.
    let hvg = if opts.hvg_floor_frac > 0.0 {
        // rank only among genes detected in more than the floor's share of kept cells
        let (stats, detected) = gene_stats_detected_blocks(&normalized, true);
        phase!("gene_stats_detected");
        let floor = (f64::from(opts.hvg_floor_frac) * normalized.n_kept() as f64) as u64;
        let allowed: Vec<bool> = (0..spilled.n_cols)
            .map(|g| detected[g] > floor && gene_keep.as_ref().is_none_or(|k| k[g]))
            .collect();
        hvg_on_subset(&stats, &allowed, opts.n_hvg)
    } else {
        let stats = gene_mean_var_blocks(&normalized, true);
        phase!("gene_mean_var");
        match &gene_keep {
            None => hvg_seurat(&stats, opts.n_hvg, 20),
            Some(genes) => hvg_on_subset(&stats, genes, opts.n_hvg),
        }
    };
    phase!("hvg_seurat");

    // 5. Subset to the HVGs — a second, much smaller spill of the *raw*
    //    counts — then PCA through the same view. Normalisation uses the
    //    full-gene row totals, as scanpy does (normalize precedes HVG
    //    selection), so the scale factors carry over unchanged.
    let sub = db
        .select_columns(&hvg.mask, &hvg_prefix, opts.block_rows)
        .map_err(IoError::Io)?;
    phase!("select_columns");
    // With `renormalize`, each cell is scaled over the selected genes alone
    // (Scarf's normalization), from the copy just written.
    let scale = match opts.renormalize {
        Some(target) => NormalizedView::scale_from_totals(&row_totals_blocks(&sub, None), target),
        None => scale,
    };
    let sub_view = NormalizedView::new(&sub, scale, true);
    let sub_normalized = FilteredRows::new(&sub_view, &keep);
    let pca_opts = PcaOptions {
        n_components: opts.n_pcs,
        center: true,
        oversample: 10,
        n_iter: 4,
        seed: opts.seed,
    };
    let pca = if opts.scale_pcs {
        pca_scaled_blocks(&sub_normalized, &pca_opts, None)
    } else {
        pca_blocks(&sub_normalized, &pca_opts)
    }
    .map_err(|e| IoError::InvalidLayout(format!("disk-backed PCA failed: {e}")))?;
    phase!("pca_blocks");

    Ok(DiskPipelineResult {
        embedding: pca.embedding,
        variance_ratio: pca.variance_ratio,
        n_components: pca.n_components,
        qc,
        hvg,
        n_rows: normalized.n_kept(),
        n_rows_input: spilled.n_rows,
        cell_keep: keep,
        pct_counts_mt,
        gene_keep: gene_keep.unwrap_or_else(|| vec![true; spilled.n_cols]),
        n_cols: spilled.n_cols,
        nnz: spilled.nnz,
    })
}

/// Seurat-flavour HVG selection on the genes flagged in `genes`, mapped back
/// to every gene: removed genes are never selected and carry `NaN`
/// statistics, as they would be absent from a gene-filtered matrix.
fn hvg_on_subset(stats: &GeneStats, genes: &[bool], n_hvg: usize) -> HvgResult {
    let idx: Vec<usize> = (0..genes.len()).filter(|&g| genes[g]).collect();
    let sub = GeneStats {
        means: idx.iter().map(|&g| stats.means[g]).collect(),
        variances: idx.iter().map(|&g| stats.variances[g]).collect(),
        n_obs: stats.n_obs,
    };
    let h = hvg_seurat(&sub, n_hvg, 20);
    let n = genes.len();
    let mut out = HvgResult {
        mask: vec![false; n],
        means: vec![f32::NAN; n],
        dispersions: vec![f32::NAN; n],
        dispersions_norm: vec![f32::NAN; n],
        n_hvg: h.n_hvg,
    };
    for (k, &g) in idx.iter().enumerate() {
        out.mask[g] = h.mask[k];
        out.means[g] = h.means[k];
        out.dispersions[g] = h.dispersions[k];
        out.dispersions_norm[g] = h.dispersions_norm[k];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use polariseq_core::CsrMatrix;

    fn fixture() -> String {
        format!(
            "{}/../../tests/data/sparse_100x200.h5ad",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    fn in_ram() -> CsrMatrix {
        let f = crate::AnnDataFile::open(fixture()).unwrap();
        match f.read_x().unwrap() {
            polariseq_core::Matrix::Csr(c) => c,
            polariseq_core::Matrix::Dense(_) => unreachable!(),
        }
    }

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir()
            .join("polariseq_spill_tests")
            .join(name)
    }

    /// A spilled matrix must be the same matrix — every value, every index.
    #[test]
    fn spill_round_trips_exactly() {
        let s = spill_h5ad(&fixture(), &tmp("round_trip")).unwrap();
        let ram = in_ram();
        assert_eq!((s.n_rows, s.n_cols, s.nnz), (100, 200, ram.nnz()));

        let db = s.open().unwrap();
        assert!(db
            .indptr()
            .iter()
            .copied()
            .eq(ram.indptr.iter().map(|&v| u64::from(v))));
        // Read it back through the block API — the interface the kernels use.
        let (mut bytes, mut data, mut indices) = (Vec::new(), Vec::new(), Vec::new());
        db.read_rows(0, ram.nrows, &mut bytes, &mut data, &mut indices)
            .unwrap();
        assert_eq!(data, ram.data);
        assert_eq!(indices, ram.indices);
    }

    /// The kernels must not care that the rows came off a disk.
    #[test]
    fn spilled_kernels_match_in_ram() {
        let s = spill_h5ad(&fixture(), &tmp("kernels")).unwrap();
        let db = s.open().unwrap();
        let ram = in_ram();
        assert_eq!(db.n_rows(), ram.nrows);
        assert_eq!(qc_metrics_blocks(&db, None), qc_metrics_blocks(&ram, None));
        assert_eq!(
            gene_mean_var_blocks(&db, false),
            gene_mean_var_blocks(&ram, false)
        );
    }

    /// End to end: the disk-backed pipeline must agree with running the same
    /// steps in memory — the whole justification for the abstraction.
    #[test]
    fn disk_pipeline_matches_the_in_memory_pipeline() {
        use polariseq_core::preprocess::{log1p_inplace, normalize_total_inplace};

        let opts = DiskPipelineOptions {
            n_hvg: 40,
            n_pcs: 5,
            spill_dir: tmp("pipeline"),
            ..Default::default()
        };
        let got = disk_backed_preprocess(&fixture(), &opts).unwrap();

        // The same pipeline, in memory.
        let mut ram = in_ram();
        let qc = qc_metrics_blocks(&ram, None);
        let _ = normalize_total_inplace(&mut ram, opts.target_sum);
        log1p_inplace(&mut ram);
        let stats = gene_mean_var_blocks(&ram, true);
        let hvg = hvg_seurat(&stats, opts.n_hvg, 20);
        let sub = ram.select_cols_mask(&hvg.mask).unwrap();
        let want = polariseq_core::pca_csr(
            &sub,
            &PcaOptions {
                n_components: opts.n_pcs,
                center: true,
                oversample: 10,
                n_iter: 4,
                seed: opts.seed,
            },
        )
        .unwrap();

        assert_eq!(got.qc, qc, "QC metrics");
        assert_eq!(got.hvg.mask, hvg.mask, "HVG selection");
        assert_eq!(got.n_components, want.n_components);

        // The embeddings agree to ~1e-4 relative but are not bit-identical,
        // for one deliberate reason: the disk path scales rows by the f64
        // totals QC already computed, while `normalize_total_inplace` sums
        // each row in f32. (The disk path is the more accurate of the two.)
        // On this 100-cell fixture that difference is enough to flip which
        // entry wins `canonicalize_signs`, so compare per component with the
        // sign aligned — the standard way to compare PCA embeddings.
        let k = got.n_components;
        for j in 0..k {
            let col =
                |e: &[f32]| -> Vec<f64> { (0..100).map(|i| f64::from(e[i * k + j])).collect() };
            let (g, w) = (col(&got.embedding), col(&want.embedding));
            let dot: f64 = g.iter().zip(&w).map(|(a, b)| a * b).sum();
            let sign = if dot < 0.0 { -1.0 } else { 1.0 };
            let scale = w.iter().fold(0.0_f64, |m, x| m.max(x.abs())).max(1e-9);
            for (a, b) in g.iter().zip(&w) {
                assert!(
                    (sign * a - b).abs() < 1e-3 * scale,
                    "PC{j}: {a} vs {b} (scale {scale})"
                );
            }
        }
    }

    #[test]
    fn reports_a_missing_file() {
        assert!(spill_h5ad("/nonexistent/never.h5ad", &tmp("missing")).is_err());
    }
}
