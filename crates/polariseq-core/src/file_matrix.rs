//! A CSR matrix on disk, read in bounded blocks.
//!
//! [`DiskBackedCSR`](crate::disk_matrix::DiskBackedCSR) memory-maps its spill
//! files. That is convenient, but it does not bound memory: every pass over
//! the matrix touches every page, each touched page becomes resident in *this
//! process*, and the OS only reclaims under pressure. Measured on
//! `heart_100k`, private memory tracked the whole matrix (1.14 GB for an
//! 892 MB matrix) and did not move when the block size was swept over a 32×
//! range — the mapped pages, not the kernels, were the footprint.
//!
//! [`FileCsr`] reads each block explicitly into a buffer it owns instead.
//! The trade:
//!
//! - **Footprint becomes exactly what we allocate** — `concurrency × block`,
//!   which is what [`crate::rowblocks::RowBlocks`] callers and the planner
//!   already reason about. Nothing accumulates. Measured on the same
//!   pipeline: private memory 1.30 GB (mmap, unmoved by block size) against
//!   0.52 / 0.21 / 0.16 GB at 16 384 / 4 096 / 1 024-row blocks — up to 8×
//!   less, and finally responsive to the knob.
//! - **Speed is unchanged.** Both paths go through the same page cache, so
//!   repeated passes still hit warm cache; the difference is one copy per
//!   byte (memcpy, ~10–20 GB/s, against arithmetic that is far slower) in
//!   exchange for dropping one page fault per 4 KiB page.
//! - **It is safe under a memory limit.** Mapped file pages count against a
//!   cgroup's limit, so mmap-based out-of-core can be OOM-killed in a
//!   container even though the pages are evictable in principle. A bounded
//!   read buffer cannot be.
//!
//! mmap remains the better tool for *random* access (`head`, `sample`), so
//! `DiskBackedCSR` keeps its place; this is for the sequential full-matrix
//! passes that make up the pipeline.

use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::rowblocks::RowBlocks;
use crate::rsvd::CsrRef;
use crate::store::{self, Layout, StoreWriter, STORE_EXTENSIONS};

/// Read exactly `buf.len()` bytes at `offset`, without a shared file cursor
/// (so concurrent block reads do not interfere).
#[cfg(unix)]
fn read_exact_at(f: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, offset)
}

#[cfg(windows)]
fn read_exact_at(f: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0usize;
    while done < buf.len() {
        match f.seek_read(&mut buf[done..], offset + done as u64)? {
            0 => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "short read from the spill file",
                ))
            }
            n => done += n,
        }
    }
    Ok(())
}

fn decode_f32(bytes: &[u8], out: &mut Vec<f32>) {
    out.clear();
    out.extend(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
    );
}

fn decode_u32(bytes: &[u8], out: &mut Vec<u32>) {
    out.clear();
    out.extend(
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])),
    );
}

/// A CSR triple stored as three flat little-endian files, read in blocks.
///
/// The row pointers are 64-bit, so a store is not limited to the 2^32 stored
/// values a 32-bit matrix can address (an atlas of tens of millions of cells
/// holds tens of billions); stores written before this, with 32-bit row
/// pointers, are recognized by their size and still read. Each block is
/// handed to the kernels with 32-bit offsets relative to its first row, which
/// a block of cells never outgrows. The row pointers are held in memory,
/// `(nrows + 1) × 8` bytes (8 MB for a million cells), because every block
/// lookup needs them; the values and column indices, the large arrays, are
/// never all resident.
pub struct FileCsr {
    backend: Backend,
    indptr: Vec<u64>,
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Number of stored nonzeros.
    pub nnz: usize,
    prefix: PathBuf,
    delete_on_drop: bool,
    max_block_rows: usize,
}

/// Where a store's values and gene ids are: the two plain files, or the
/// compact chunks and their index (see [`crate::store`]).
enum Backend {
    Plain {
        data: File,
        indices: File,
    },
    Compact {
        file: File,
        chunk_rows: usize,
        offsets: Vec<u64>,
    },
}

/// Default cap on rows per block when no budget says otherwise.
const DEFAULT_MAX_BLOCK_ROWS: usize = 32_768;

impl FileCsr {
    /// Open a triple written as `{prefix}.data/.indices/.indptr`.
    ///
    /// # Errors
    /// Returns an error if a file is missing, or if the row pointers do not
    /// describe a matrix of `nrows` rows.
    pub fn open_at(prefix: &Path, nrows: usize, ncols: usize) -> io::Result<Self> {
        let backend = match Layout::of(prefix) {
            Layout::Compact => {
                let (chunk_rows, offsets) = store::read_index(prefix, nrows)?;
                Backend::Compact {
                    file: File::open(prefix.with_extension("cz"))?,
                    chunk_rows,
                    offsets,
                }
            }
            Layout::Plain => Backend::Plain {
                data: File::open(prefix.with_extension("data"))?,
                indices: File::open(prefix.with_extension("indices"))?,
            },
        };
        let indptr_bytes = std::fs::read(prefix.with_extension("indptr"))?;
        // 64-bit row pointers, or 32-bit ones from a store written before;
        // the row count makes the two sizes unambiguous.
        let indptr: Vec<u64> = if indptr_bytes.len() == (nrows + 1) * 8 {
            indptr_bytes
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect()
        } else if indptr_bytes.len() == (nrows + 1) * 4 {
            let mut narrow = Vec::new();
            decode_u32(&indptr_bytes, &mut narrow);
            narrow.into_iter().map(u64::from).collect()
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "indptr has {} bytes, which is neither 8 nor 4 bytes for each of the {} row \
                     pointers of {nrows} rows",
                    indptr_bytes.len(),
                    nrows + 1
                ),
            ));
        };
        let nnz = usize::try_from(indptr[nrows]).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "more stored values than addressable",
            )
        })?;
        Ok(Self {
            backend,
            indptr,
            nrows,
            ncols,
            nnz,
            prefix: prefix.to_path_buf(),
            delete_on_drop: false,
            max_block_rows: DEFAULT_MAX_BLOCK_ROWS,
        })
    }

    /// Delete the backing files when this value is dropped.
    #[must_use]
    pub fn delete_on_drop(mut self, yes: bool) -> Self {
        self.delete_on_drop = yes;
        self
    }

    /// Cap the rows per block, whatever a kernel asks for.
    ///
    /// Kernels pick a block size to suit their arithmetic — `chunk_rows`
    /// returns 32 768 rows for a 100 000-cell matrix — and have no idea
    /// whether a block is a slice of resident memory or a fresh read off a
    /// disk. Only the source knows, so the source clamps. This is the knob
    /// `ps.plan()` computes.
    #[must_use]
    pub fn with_max_block_rows(mut self, rows: usize) -> Self {
        self.max_block_rows = rows.max(1);
        self
    }

    /// Row pointers (resident; small).
    #[must_use]
    pub fn indptr(&self) -> &[u64] {
        &self.indptr
    }

    /// Read rows `[r0, r1)` into `data`/`indices`, reusing their allocations.
    ///
    /// # Errors
    /// Returns an error if the underlying reads fail, or if a compact chunk is
    /// damaged (fails its checksum).
    ///
    /// # Panics
    /// Panics if a single row holds 2^32 values or more.
    pub fn read_rows(
        &self,
        r0: usize,
        r1: usize,
        bytes: &mut Vec<u8>,
        data: &mut Vec<f32>,
        indices: &mut Vec<u32>,
    ) -> io::Result<()> {
        let (file, chunk_rows, offsets) = match &self.backend {
            Backend::Plain {
                data: df,
                indices: ixf,
            } => {
                let (s, e) = (self.indptr[r0] as usize, self.indptr[r1] as usize);
                let len = (e - s) * 4;
                bytes.resize(len, 0);
                if len > 0 {
                    read_exact_at(df, bytes, s as u64 * 4)?;
                }
                decode_f32(bytes, data);
                if len > 0 {
                    read_exact_at(ixf, bytes, s as u64 * 4)?;
                }
                decode_u32(bytes, indices);
                return Ok(());
            }
            Backend::Compact {
                file,
                chunk_rows,
                offsets,
            } => (file, *chunk_rows, offsets),
        };
        // Compact: decode the chunks the rows fall in. A chunk wholly inside
        // the range is decoded straight into the output; a chunk cut by an end
        // of the range is decoded aside and sliced.
        data.clear();
        indices.clear();
        if r0 >= r1 {
            return Ok(());
        }
        let (mut part_d, mut part_i, mut scratch) = (Vec::new(), Vec::new(), Vec::new());
        for c in r0 / chunk_rows..=(r1 - 1) / chunk_rows {
            let (a, b) = (offsets[c], offsets[c + 1]);
            bytes.resize((b - a) as usize, 0);
            read_exact_at(file, bytes, a)?;
            let (c0, c1) = (c * chunk_rows, ((c + 1) * chunk_rows).min(self.nrows));
            let lens: Vec<u32> = (c0..c1)
                .map(|r| {
                    u32::try_from(self.indptr[r + 1] - self.indptr[r])
                        .expect("a row holds fewer than 2^32 values")
                })
                .collect();
            if r0 <= c0 && c1 <= r1 {
                store::decode_chunk(bytes, &lens, data, indices, &mut scratch)?;
            } else {
                part_d.clear();
                part_i.clear();
                store::decode_chunk(bytes, &lens, &mut part_d, &mut part_i, &mut scratch)?;
                let lo = (self.indptr[r0.max(c0)] - self.indptr[c0]) as usize;
                let hi = (self.indptr[r1.min(c1)] - self.indptr[c0]) as usize;
                data.extend_from_slice(&part_d[lo..hi]);
                indices.extend_from_slice(&part_i[lo..hi]);
            }
        }
        Ok(())
    }

    /// Write the columns selected by `mask` to a new triple at `new_prefix`,
    /// streaming `block_rows` rows at a time. Neither matrix is ever fully
    /// resident.
    ///
    /// # Errors
    /// Returns an error on any read or write failure.
    ///
    /// # Panics
    /// Panics if `mask.len() != ncols`.
    pub fn select_columns(
        &self,
        mask: &[bool],
        new_prefix: &Path,
        block_rows: usize,
    ) -> io::Result<Self> {
        assert_eq!(mask.len(), self.ncols, "mask length must match n_cols");
        let mut col_map = vec![u32::MAX; self.ncols];
        let mut next = 0_u32;
        for (j, &keep) in mask.iter().enumerate() {
            if keep {
                col_map[j] = next;
                next += 1;
            }
        }
        let new_ncols = next as usize;

        let mut writer = StoreWriter::create(new_prefix, Layout::for_new_stores())?;
        let mut lens: Vec<u32> = Vec::new();

        let block_rows = block_rows.max(1);
        let (mut bytes, mut data, mut indices) = (Vec::new(), Vec::new(), Vec::new());
        let (mut out_d, mut out_i) = (Vec::new(), Vec::new());
        let mut r0 = 0;
        while r0 < self.nrows {
            let r1 = (r0 + block_rows).min(self.nrows);
            self.read_rows(r0, r1, &mut bytes, &mut data, &mut indices)?;
            out_d.clear();
            out_i.clear();
            lens.clear();
            let base = self.indptr[r0];
            for i in r0..r1 {
                let (a, b) = (
                    (self.indptr[i] - base) as usize,
                    (self.indptr[i + 1] - base) as usize,
                );
                let before = out_d.len();
                for (&c, &v) in indices[a..b].iter().zip(&data[a..b]) {
                    let nc = col_map[c as usize];
                    if nc != u32::MAX {
                        out_i.push(nc);
                        out_d.push(v);
                    }
                }
                lens.push(
                    u32::try_from(out_d.len() - before)
                        .expect("a row holds fewer than 2^32 values"),
                );
            }
            writer.push(&out_d, &out_i, &lens)?;
            r0 = r1;
        }
        writer.finish()?;

        let mut out = Self::open_at(new_prefix, self.nrows, new_ncols)?;
        out.delete_on_drop = true;
        out.max_block_rows = self.max_block_rows;
        Ok(out)
    }
}

impl Drop for FileCsr {
    fn drop(&mut self) {
        if self.delete_on_drop {
            for ext in STORE_EXTENSIONS {
                let _ = std::fs::remove_file(self.prefix.with_extension(ext));
            }
        }
    }
}

/// Blocks are read into per-task buffers, so the resident cost of a pass is
/// `concurrency × block`, not the matrix. Read failures cannot be reported
/// through the trait, so a failing block is skipped; callers that need the
/// error use [`FileCsr::read_rows`] directly.
impl RowBlocks for FileCsr {
    fn n_rows(&self) -> usize {
        self.nrows
    }

    fn n_cols(&self) -> usize {
        self.ncols
    }

    fn for_each_block(&self, rows_per_block: usize, f: &(dyn Fn(usize, CsrRef<'_>) + Sync)) {
        let rpb = rows_per_block.clamp(1, self.max_block_rows.max(1));
        let n_blocks = self.nrows.div_ceil(rpb);
        // Blocks run in batches of at most `concurrency`, so the number of
        // blocks alive at any moment is bounded and a pass costs
        // `concurrency × block` — the quantity `ps.plan()` predicts.
        //
        // Two alternatives measured on heart_100k and rejected: an unbounded
        // `par_iter` over every block lets rayon keep far more than
        // `concurrency` of them live, and `for_each_with` is worse still,
        // because it clones its scratch per work-split rather than per
        // thread (1.22 GB against 0.75 GB for this form).
        let concurrency = rayon::current_num_threads().max(1);
        let mut b0 = 0;
        while b0 < n_blocks {
            let b1 = (b0 + concurrency).min(n_blocks);
            (b0..b1).into_par_iter().for_each(|b| {
                let (r0, r1) = (b * rpb, ((b + 1) * rpb).min(self.nrows));
                let (mut bytes, mut data, mut indices) = (Vec::new(), Vec::new(), Vec::new());
                if self
                    .read_rows(r0, r1, &mut bytes, &mut data, &mut indices)
                    .is_err()
                {
                    return;
                }
                drop(bytes);
                let base = self.indptr[r0];
                // Offsets within the block: a block of at most 32,768 cells
                // holds far fewer than 2^32 values.
                let local: Vec<u32> = self.indptr[r0..=r1]
                    .iter()
                    .map(|&v| {
                        u32::try_from(v - base).expect("a block holds fewer than 2^32 values")
                    })
                    .collect();
                f(
                    r0,
                    CsrRef {
                        data: &data,
                        indices: &indices,
                        indptr: &local,
                        n_rows: r1 - r0,
                        n_cols: self.ncols,
                    },
                );
            });
            b0 = b1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::CsrMatrix;
    use crate::preprocess::{gene_mean_var_blocks, qc_metrics_blocks};

    fn write_triple(m: &CsrMatrix, prefix: &Path) {
        use std::io::Write;
        std::fs::create_dir_all(prefix.parent().unwrap()).unwrap();
        let mut f = File::create(prefix.with_extension("data")).unwrap();
        for v in &m.data {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        let mut f = File::create(prefix.with_extension("indices")).unwrap();
        for v in &m.indices {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        let mut f = File::create(prefix.with_extension("indptr")).unwrap();
        for v in &m.indptr {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }

    /// 40 rows with empty rows, an explicit zero, and a ragged tail.
    fn sample() -> CsrMatrix {
        let mut indptr = vec![0_u32];
        let mut indices: Vec<u32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        for i in 0..40_u32 {
            if i % 5 != 0 {
                indices.extend([i % 6, (i + 2) % 6]);
                data.extend([(i % 7) as f32, if i % 11 == 0 { 0.0 } else { 1.5 }]);
            }
            indptr.push(data.len() as u32);
        }
        CsrMatrix::new(indptr, indices, data, 40, 6).unwrap()
    }

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir()
            .join("polariseq_filecsr_tests")
            .join(name)
    }

    /// Rows stored past 2^32 values are read at the right place. The files
    /// are sparse: the first 2^32 + 3 entries, all in row 0, are a hole that
    /// takes no disk space and is never read; rows 1 to 3 follow it.
    #[test]
    fn rows_past_two_to_the_32_values_are_read_where_they_lie() {
        use std::io::{Seek, SeekFrom, Write};
        let prefix = tmp("past_u32");
        std::fs::create_dir_all(prefix.parent().unwrap()).unwrap();
        let hole: u64 = (1 << 32) + 3;
        let vals = [1.5_f32, 2.0, 3.25, 4.0, 5.5];
        let cols = [0_u32, 3, 2, 1, 4];
        for (ext, bytes) in [
            (
                "data",
                vals.iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
            (
                "indices",
                cols.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
        ] {
            let mut f = File::create(prefix.with_extension(ext)).unwrap();
            f.seek(SeekFrom::Start(hole * 4)).unwrap();
            f.write_all(&bytes).unwrap();
        }
        let indptr = [0_u64, hole, hole + 2, hole + 3, hole + 5];
        let mut f = File::create(prefix.with_extension("indptr")).unwrap();
        for v in indptr {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(f);
        let m = FileCsr::open_at(&prefix, 4, 5).unwrap();
        assert_eq!(m.nnz, usize::try_from(hole + 5).unwrap());
        let (mut bytes, mut data, mut indices) = (Vec::new(), Vec::new(), Vec::new());
        m.read_rows(1, 4, &mut bytes, &mut data, &mut indices)
            .unwrap();
        assert_eq!(data, vals);
        assert_eq!(indices, cols);
        m.read_rows(2, 3, &mut bytes, &mut data, &mut indices)
            .unwrap();
        assert_eq!(
            (data.as_slice(), indices.as_slice()),
            (&vals[2..3], &cols[2..3])
        );
        for ext in ["data", "indices", "indptr"] {
            let _ = std::fs::remove_file(prefix.with_extension(ext));
        }
    }

    /// A store written with 32-bit row pointers, as before 2026-09-30, reads
    /// as it did; one written now has 64-bit ones.
    #[test]
    fn stores_of_either_pointer_width_are_read() {
        let m = sample();
        let old = tmp("width_u32");
        write_triple(&m, &old); // 32-bit row pointers
        assert_eq!(
            std::fs::metadata(old.with_extension("indptr"))
                .unwrap()
                .len(),
            41 * 4
        );
        let a = FileCsr::open_at(&old, m.nrows, m.ncols).unwrap();
        let new = tmp("width_u64");
        let b = a.select_columns(&[true; 6], &new, 7).unwrap();
        assert_eq!(
            std::fs::metadata(new.with_extension("indptr"))
                .unwrap()
                .len(),
            41 * 8
        );
        let widened: Vec<u64> = m.indptr.iter().map(|&v| u64::from(v)).collect();
        assert_eq!(
            (a.indptr(), b.indptr()),
            (widened.as_slice(), widened.as_slice())
        );
        assert!(FileCsr::open_at(&old, m.nrows + 1, m.ncols).is_err());
    }

    #[test]
    fn blocks_reproduce_the_matrix_at_every_block_size() {
        let m = sample();
        let prefix = tmp("blocks");
        write_triple(&m, &prefix);
        let f = FileCsr::open_at(&prefix, m.nrows, m.ncols).unwrap();
        assert_eq!((f.n_rows(), f.n_cols(), f.nnz), (40, 6, m.nnz()));

        for rpb in [1, 3, 7, 40, 128] {
            let seen = std::sync::Mutex::new(Vec::new());
            f.for_each_block(rpb, &|off, blk| {
                let mut rows = Vec::new();
                for i in 0..blk.n_rows {
                    let (a, b) = (blk.indptr[i] as usize, blk.indptr[i + 1] as usize);
                    rows.push((off + i, blk.indices[a..b].to_vec(), blk.data[a..b].to_vec()));
                }
                seen.lock().unwrap().extend(rows);
            });
            let mut rows = seen.into_inner().unwrap();
            rows.sort_by_key(|r| r.0);
            assert_eq!(rows.len(), 40, "rpb={rpb}");
            for (i, cols, vals) in rows {
                let (want_c, want_v) = m.row(i);
                assert_eq!(cols, want_c, "row {i} at rpb={rpb}");
                assert_eq!(vals, want_v, "row {i} at rpb={rpb}");
            }
        }
    }

    /// The point of the type: identical kernel results to the in-RAM matrix.
    #[test]
    fn kernels_match_the_in_ram_matrix() {
        let m = sample();
        let prefix = tmp("kernels");
        write_triple(&m, &prefix);
        let f = FileCsr::open_at(&prefix, m.nrows, m.ncols).unwrap();
        assert_eq!(qc_metrics_blocks(&f, None), qc_metrics_blocks(&m, None));
        assert_eq!(
            gene_mean_var_blocks(&f, false),
            gene_mean_var_blocks(&m, false)
        );
    }

    #[test]
    fn select_columns_streams_a_subset() {
        let m = sample();
        let prefix = tmp("select_src");
        write_triple(&m, &prefix);
        let f = FileCsr::open_at(&prefix, m.nrows, m.ncols).unwrap();

        let mask = vec![true, false, true, false, true, false];
        let sub = f.select_columns(&mask, &tmp("select_dst"), 7).unwrap();
        let want = m.select_cols_mask(&mask).unwrap();

        assert_eq!((sub.n_rows(), sub.n_cols(), sub.nnz), (40, 3, want.nnz()));
        assert_eq!(
            qc_metrics_blocks(&sub, None),
            qc_metrics_blocks(&want, None)
        );
    }

    #[test]
    fn rejects_an_indptr_that_does_not_match() {
        let m = sample();
        let prefix = tmp("bad");
        write_triple(&m, &prefix);
        assert!(FileCsr::open_at(&prefix, 39, m.ncols).is_err());
    }

    #[test]
    fn compact_and_plain_stores_read_the_same_rows_for_any_range() {
        use crate::store::{Layout, StoreWriter, CHUNK_ROWS};
        // 3,000 rows (three chunks and a partial one), sorted rows, counts.
        let n_rows = 3 * CHUNK_ROWS - 72;
        let (mut d, mut ix, mut lens) = (Vec::new(), Vec::new(), Vec::new());
        let mut s = 0x9E37_79B9_u64;
        for _ in 0..n_rows {
            let mut l = 0_u32;
            for c in 0..800_u32 {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                if s % 13 == 0 {
                    ix.push(c);
                    d.push((1 + s % 5) as f32);
                    l += 1;
                }
            }
            lens.push(l);
        }
        let dir = std::env::temp_dir().join(format!("pq_layouts_{}", std::process::id()));
        let open = |layout: Layout| {
            let prefix = dir.join(format!("{layout:?}"));
            let mut w = StoreWriter::create(&prefix, layout).unwrap();
            w.push(&d, &ix, &lens).unwrap();
            w.finish().unwrap();
            FileCsr::open_at(&prefix, n_rows, 800).unwrap()
        };
        let (plain, compact) = (open(Layout::Plain), open(Layout::Compact));
        let mut sizes = [0_u64; 2];
        for (i, ext) in [["data", "indices"], ["cz", "czi"]].iter().enumerate() {
            for e in ext {
                sizes[i] += std::fs::metadata(
                    dir.join(format!(
                        "{:?}",
                        if i == 0 {
                            Layout::Plain
                        } else {
                            Layout::Compact
                        }
                    ))
                    .with_extension(e),
                )
                .unwrap()
                .len();
            }
        }
        assert!(
            sizes[1] * 2 < sizes[0],
            "compact {} vs plain {} bytes",
            sizes[1],
            sizes[0]
        );
        let ranges = [
            (0, n_rows),
            (0, 1),
            (5, 6),
            (1000, 1100),
            (1023, 1025),
            (CHUNK_ROWS, 2 * CHUNK_ROWS),
            (17, 2950),
            (n_rows - 3, n_rows),
            (700, 700),
        ];
        let (mut b, mut d1, mut i1, mut d2, mut i2) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (r0, r1) in ranges {
            plain.read_rows(r0, r1, &mut b, &mut d1, &mut i1).unwrap();
            compact.read_rows(r0, r1, &mut b, &mut d2, &mut i2).unwrap();
            assert_eq!(i1, i2, "gene ids differ for rows {r0}..{r1}");
            assert_eq!(
                d1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                d2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "values differ for rows {r0}..{r1}"
            );
        }
        // And every kernel built on blocks agrees, at a block size that is not a chunk multiple.
        assert_eq!(
            qc_metrics_blocks(&plain.with_max_block_rows(777), None),
            qc_metrics_blocks(&compact.with_max_block_rows(777), None)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
