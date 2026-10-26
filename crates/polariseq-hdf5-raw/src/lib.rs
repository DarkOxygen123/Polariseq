//! Minimal raw-FFI HDF5 chunk access.
//!
//! `hdf5-metno` 0.14 exposes no safe wrapper for `H5Dread_chunk` (reading
//! a chunk's raw *compressed* bytes), and `polariseq-io` is
//! `#![forbid(unsafe_code)]` — so exactly those four FFI calls live here,
//! each a one-screen wrapper with its invariants documented, and nothing
//! else. The safe parallel inflate/unshuffle pipeline that consumes this
//! lives in `polariseq-io`.
//!
//! All functions borrow an open [`Dataset`] handle; none of them modify
//! the file (read-only access), and buffers are always caller-owned.

use hdf5_metno::Dataset;
use std::sync::{Mutex, MutexGuard};

/// Serialises every FFI call in this crate. The vendored HDF5 is not built
/// thread-safe (`hdf5-metno-sys`'s `threadsafe` feature is off), and these
/// property-list/chunk-index calls touch global library state — the
/// library's ordinary reads never run concurrently here, but parallel
/// tests (and later parallel readers) would crash without this.
static FFI_LOCK: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    FFI_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

/// Take the HDF5 FFI lock (the vendored library is not thread-safe, so any
/// sequence of hdf5-metno calls that touches global state — dtype/shape/
/// chunk queries included — must hold it).
pub fn ffi_lock() -> MutexGuard<'static, ()> {
    lock()
}
use hdf5_metno_sys::h5::{herr_t, hsize_t};
use hdf5_metno_sys::h5d::{
    H5Dget_chunk_info, H5Dget_chunk_info_by_coord, H5Dget_create_plist, H5Dget_num_chunks,
};
use hdf5_metno_sys::h5i::hid_t;

// hdf5-metno-sys 0.12 never compiled its H5Dread_chunk2 binding (the cfg
// feature does not exist), while the vendored HDF5 2.2 renamed the old
// entry point away — so declare it verbatim from the vendored H5Dpublic.h.
extern "C" {
    fn H5Dread_chunk2(
        dset_id: hid_t,
        dxpl_id: hid_t,
        offset: *const hsize_t,
        filters: *mut u32,
        buf: *mut std::os::raw::c_void,
        buf_size: *mut usize,
    ) -> herr_t;
}
use hdf5_metno_sys::h5p::{H5Pclose, H5Pget_filter2, H5Pget_nfilters};
use hdf5_metno_sys::h5s::H5Sget_simple_extent_ndims;
type CUInt = std::os::raw::c_uint;

/// Filter identifiers (`H5Z_*` constants; not exported by the sys crate).
pub mod filter_ids {
    use crate::CUInt;
    /// `H5Z_FILTER_DEFLATE` — gzip.
    pub const DEFLATE: CUInt = 1;
    /// `H5Z_FILTER_SHUFFLE` — byte shuffle.
    pub const SHUFFLE: CUInt = 2;
}

/// One filter in a dataset's pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterInfo {
    /// HDF5 filter id (`filter_ids::*`).
    pub id: CUInt,
    /// Auxilliary cd-values (e.g. shuffle element size, deflate level).
    pub cd_values: Vec<CUInt>,
}

/// Metadata of one stored chunk, from `H5Dget_chunk_info`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkMeta {
    /// Logical element offset of the chunk's first element (1-D datasets).
    pub elem_offset: u64,
    /// File byte offset of the chunk (`H5Dread_chunk` key).
    pub byte_offset: u64,
    /// Stored (compressed) size in bytes.
    pub nbytes: u64,
}

/// Number of chunks of a chunked 1-D dataset. `None` when the call fails
/// (not chunked, or an HDF5 version without the chunk-index API).
#[must_use]
pub fn num_chunks(ds: &Dataset) -> Option<usize> {
    let _guard = lock();
    let space = ds.space().ok()?;
    let mut n: hdf5_metno_sys::h5::hsize_t = 0;
    let r = unsafe { H5Dget_num_chunks(ds.id(), space.id(), &mut n) };
    (r >= 0).then_some(n as usize)
}

/// Chunk metadata by index. Indices enumerate chunks in logical order for
/// 1-D datasets (chunk 0 covers the first elements).
#[must_use]
pub fn chunk_meta(ds: &Dataset, index: usize) -> Option<ChunkMeta> {
    let _guard = lock();
    let space = ds.space().ok()?;
    let ndims = unsafe { H5Sget_simple_extent_ndims(space.id()) };
    if ndims < 1 {
        return None;
    }
    let mut coord = vec![0_u64; ndims.max(1) as usize];
    let mut mask: CUInt = 0;
    let mut byte_offset: u64 = 0;
    let mut nbytes: u64 = 0;
    let r = unsafe {
        H5Dget_chunk_info(
            ds.id(),
            space.id(),
            index as hdf5_metno_sys::h5::hsize_t,
            coord.as_mut_ptr(),
            &mut mask,
            &mut byte_offset,
            &mut nbytes,
        )
    };
    (r >= 0).then_some(ChunkMeta {
        elem_offset: coord[0],
        byte_offset,
        nbytes,
    })
}

/// Chunk metadata addressed by the chunk's **logical element offset**
/// (`O(log n)` B-tree lookup — use this when iterating; `chunk_meta` by
/// index walks the index linearly and is `O(n)` per call).
#[must_use]
pub fn chunk_meta_by_coord(ds: &Dataset, elem_offset: u64) -> Option<ChunkMeta> {
    let _guard = lock();
    let space = ds.space().ok()?;
    let ndims = unsafe { H5Sget_simple_extent_ndims(space.id()) };
    if ndims < 1 {
        return None;
    }
    let coord = [elem_offset as hdf5_metno_sys::h5::hsize_t];
    let mut mask: CUInt = 0;
    let mut byte_offset: u64 = 0;
    let mut nbytes: u64 = 0;
    let r = unsafe {
        H5Dget_chunk_info_by_coord(
            ds.id(),
            coord.as_ptr(),
            &mut mask,
            &mut byte_offset,
            &mut nbytes,
        )
    };
    (r >= 0).then_some(ChunkMeta {
        elem_offset,
        byte_offset,
        nbytes,
    })
}

/// Read one chunk's raw stored bytes (compressed, filters not applied),
/// addressing the chunk by its **logical element offset** (`ChunkMeta::
/// elem_offset` — HDF5's chunk coordinate, not the file byte address).
/// This bypasses the library's decompression and chunk cache entirely.
/// `buf` must be at least the chunk's stored size.
///
/// # Panics
/// Panics if `buf` is shorter than the chunk's stored size.
pub fn read_chunk(ds: &Dataset, elem_offset: u64, buf: &mut [u8]) -> bool {
    let _guard = lock();
    let mut mask: u32 = 0;
    let offset = elem_offset as hsize_t;
    let mut buf_size: usize = buf.len();
    let r = unsafe {
        H5Dread_chunk2(
            ds.id(),
            hdf5_metno_sys::h5p::H5P_DEFAULT,
            &offset,
            &mut mask,
            buf.as_mut_ptr().cast(),
            &mut buf_size,
        )
    };
    r >= 0
}

/// The dataset's filter pipeline (creation property list).
#[must_use]
pub fn filters(ds: &Dataset) -> Vec<FilterInfo> {
    let _guard = lock();
    let mut out = Vec::new();
    let plist = unsafe { H5Dget_create_plist(ds.id()) };
    if plist < 0 {
        return out;
    }
    let n = unsafe { H5Pget_nfilters(plist) };
    for i in 0..n {
        let mut name = [0_u8; 64];
        let mut cd_values = [0 as CUInt; 16];
        let mut flags = 0 as CUInt;
        let mut n_cd: usize = 16; // in: capacity, out: number written
        let mut filter_config = 0 as CUInt;
        let id = unsafe {
            H5Pget_filter2(
                plist,
                i as CUInt,
                &mut flags,
                &mut n_cd,
                cd_values.as_mut_ptr(),
                64,
                name.as_mut_ptr().cast(),
                &mut filter_config,
            )
        };
        if id >= 0 {
            out.push(FilterInfo {
                id: id as CUInt,
                cd_values: cd_values[..n_cd.min(16)].to_vec(),
            });
        }
    }
    unsafe { H5Pclose(plist) };
    out
}
