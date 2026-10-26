//! Chunk-parallel reads of compressed 1-D HDF5 datasets.
//!
//! The HDF5 library inflates gzip chunks one at a time inside `H5Dread`;
//! on an 8-core machine that leaves most of the machine idle during
//! load. This reader enumerates the dataset's chunks
//! ([`polariseq_hdf5_raw`]), reads their raw compressed bytes sequentially
//! (pure I/O, in windows so memory stays bounded), then inflates
//! (zlib — HDF5's deflate is zlib-wrapped) and decodes them in parallel
//! with rayon, straight into the typed output. Byte-shuffle (anndata
//! writes it on scanpy-produced files) is undone after inflate; filters
//! are applied in pipeline order on write, so reversed on read.
//!
//! Anything unusual — dataset not chunked/gzipped, dtypes or layouts we do
//! not recognise, filter pipelines beyond deflate+shuffle, or any inflate
//! failure — returns `None` and the caller falls back to the library
//! path. `POLARISEQ_TRACE=1` prints which path ran and how long it took.

#![allow(clippy::too_many_lines)]
use flate2::read::ZlibDecoder;
use hdf5_metno::types::{FloatSize, IntSize, TypeDescriptor};
use hdf5_metno::Dataset;
use polariseq_hdf5_raw as raw;
use polariseq_hdf5_raw::filter_ids;
use rayon::prelude::*;
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};

/// Chunks read (sequentially) per window; bounds raw compressed bytes in
/// flight to ~32 chunks.
const WINDOW: usize = 32;

/// Read a 1-D float dataset into `Vec<f32>` (f64 narrows), or `None` to
/// fall back to the library path.
#[must_use]
pub fn read_f32(ds: &Dataset) -> Option<Vec<f32>> {
    match ds.dtype().ok()?.to_descriptor().ok()? {
        TypeDescriptor::Float(FloatSize::U4) => {
            read_chunks(ds, 4, |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        }
        TypeDescriptor::Float(FloatSize::U8) => read_chunks(ds, 8, |b| {
            f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
        }),
        _ => None,
    }
}

/// Read a 1-D integer dataset into `Vec<u32>` (i32/i64 widen), or `None`
/// to fall back to the library path.
#[must_use]
pub fn read_u32(ds: &Dataset) -> Option<Vec<u32>> {
    match ds.dtype().ok()?.to_descriptor().ok()? {
        TypeDescriptor::Integer(IntSize::U4) => read_chunks(ds, 4, |b| {
            i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32
        }),
        TypeDescriptor::Integer(IntSize::U8) => read_chunks(ds, 8, |b| {
            i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as u32
        }),
        _ => None,
    }
}

/// The chunk-parallel engine shared by both readers.
fn read_chunks<T: Send + Sync + Copy>(
    ds: &Dataset,
    elem_size: usize,
    decode: impl Fn(&[u8]) -> T + Sync,
) -> Option<Vec<T>> {
    let trace = std::env::var_os("POLARISEQ_TRACE").is_some();
    let t0 = std::time::Instant::now();

    // The vendored HDF5 library is not thread-safe; the raw:: helpers each
    // take the global FFI lock (do NOT wrap them in another ffi_lock —
    // the std Mutex is not recursive), and the inflate stage is pure
    // Rust. The fast path is opt-in (POLARISEQ_FASTREAD), so it never runs
    // concurrently with the parallel test suite's library reads.
    let (filters, shape, chunk_shape) = (raw::filters(ds), ds.shape(), ds.chunk());
    // Only deflate (optionally with shuffle) is handled; anything else —
    // including plain uncompressed — falls back to the library path.
    let mut shuffle = false;
    let mut deflate = false;
    for f in &filters {
        match f.id {
            filter_ids::DEFLATE => deflate = true,
            filter_ids::SHUFFLE => {
                // cd_values[0] is the element size, or 0 for "use the
                // type size" (what h5py writes).
                let s = f.cd_values.first().copied().unwrap_or(0);
                if s != 0 && s as usize != elem_size {
                    return None;
                }
                shuffle = true;
            }
            _ => return None,
        }
    }
    if !deflate {
        return None;
    }
    // Shuffle (if present) always runs at this dataset's element size.
    let shuffle_size = shuffle.then_some(elem_size);
    if shape.len() != 1 {
        return None;
    }
    let n_elems = shape[0];

    let n_chunks = raw::num_chunks(ds)?;
    let chunk_elems = *chunk_shape?.first()?;
    if chunk_elems == 0 {
        return None;
    }
    if trace {
        eprintln!(
            "[io] fast read probe: n_elems={n_elems} chunk_elems={chunk_elems} n_chunks={n_chunks} (expected {})",
            n_elems.div_ceil(chunk_elems)
        );
    }
    if n_chunks != n_elems.div_ceil(chunk_elems) {
        // Chunk geometry is not regular (filtered/externally stored);
        // the arithmetic addressing below would be wrong.
        return None;
    }

    let mut out: Vec<T> = Vec::with_capacity(n_elems);
    out.resize_with(n_elems, || decode(&[0_u8; 8][..elem_size]));
    let out = std::sync::Mutex::new(out);
    let failed = AtomicBool::new(false);
    let t_read = std::sync::atomic::AtomicU64::new(0);
    let t_inflate = std::sync::atomic::AtomicU64::new(0);

    let mut start = 0_usize;
    while start < n_chunks {
        let end = (start + WINDOW).min(n_chunks);
        // Sequential I/O: read the window's raw compressed chunks. The
        // element offsets of 1-D chunks are regular (i * chunk_elems); the
        // stored sizes come from by-coordinate lookups (O(log n) each —
        // by-index walks the chunk index linearly and is quadratic over
        // the whole dataset).
        let metas: Vec<raw::ChunkMeta> = (start..end)
            .map(|i| raw::chunk_meta_by_coord(ds, (i * chunk_elems) as u64))
            .collect::<Option<Vec<_>>>()?;
        let t0w = std::time::Instant::now();
        let bufs: Vec<Vec<u8>> = metas
            .iter()
            .map(|m| {
                let mut buf = vec![0_u8; m.nbytes as usize];
                if !raw::read_chunk(ds, m.elem_offset, &mut buf) {
                    if trace {
                        eprintln!(
                            "[io] fast read: chunk read failed at elem {} ({} B)",
                            m.elem_offset, m.nbytes
                        );
                    }
                    return None;
                }
                Some(buf)
            })
            .collect::<Option<Vec<_>>>()?;
        t_read.fetch_add(t0w.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // Parallel inflate + unshuffle + decode into the output rows.
        let t0i = std::time::Instant::now();
        metas.par_iter().zip(&bufs).for_each(|(m, compressed)| {
            let first = m.elem_offset as usize;
            if first >= n_elems {
                failed.store(true, Ordering::Relaxed);
                return;
            }
            let count = chunk_elems.min(n_elems - first);
            let mut inflated = Vec::with_capacity(chunk_elems * elem_size);
            let ok = ZlibDecoder::new(&compressed[..])
                .read_to_end(&mut inflated)
                .is_ok()
                // Partial edge chunks are stored full-size (HDF5 zero-pads
                // before filtering); the padding is never decoded.
                && inflated.len() == chunk_elems * elem_size;
            if !ok {
                if trace {
                    eprintln!(
                        "[io] fast read: inflate mismatch at elem {} (want {} B, got {})",
                        m.elem_offset,
                        count * elem_size,
                        inflated.len()
                    );
                }
                failed.store(true, Ordering::Relaxed);
                return;
            }
            if let Some(s) = shuffle_size {
                unshuffle(&mut inflated, s);
            }
            // Only the real elements; a partial edge chunk's tail is
            // zero padding.
            let mut local = Vec::with_capacity(count);
            for b in inflated.chunks_exact(elem_size).take(count) {
                local.push(decode(b));
            }
            let mut guard = out.lock().expect("fastread output poisoned");
            guard[first..first + count].copy_from_slice(&local);
        });
        t_inflate.fetch_add(t0i.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if failed.load(Ordering::Relaxed) {
            return None;
        }
        start = end;
    }
    let out = out.into_inner().expect("fastread output poisoned");

    if trace {
        eprintln!(
            "[io] fast chunk read: {n_elems} elems, {n_chunks} chunks, total {:.2}s (raw read {:.2}s, inflate+decode {:.2}s)",
            t0.elapsed().as_secs_f64(),
            t_read.load(Ordering::Relaxed) as f64 / 1e9,
            t_inflate.load(Ordering::Relaxed) as f64 / 1e9
        );
    }
    Some(out)
}

/// Undo HDF5's byte shuffle: the stored buffer groups bytes by their
/// position within the element (`stored[j*n + i] = orig[i*s + j]`).
fn unshuffle(buf: &mut [u8], s: usize) {
    if s <= 1 {
        return;
    }
    let n = buf.len() / s;
    let mut out = vec![0_u8; buf.len()];
    for (i, dst) in out.chunks_exact_mut(s).enumerate() {
        for (j, d) in dst.iter_mut().enumerate() {
            *d = buf[j * n + i];
        }
    }
    buf.copy_from_slice(&out);
}
