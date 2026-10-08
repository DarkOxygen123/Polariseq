//! The on-disk block store's two layouts, and the writer every copy uses.
//!
//! - **Plain**: `{prefix}.data` (f32 values) and `{prefix}.indices` (u32 gene
//!   ids), 8 bytes a stored value.
//! - **Compact** (the default for new stores): rows in chunks of
//!   [`CHUNK_ROWS`]. In each chunk the gene ids (gaps within a row when the row
//!   is sorted, absolute otherwise) and the values (u16 when every value is a
//!   whole number from 0 to 65,535, else the f32 bits) are split into byte
//!   planes, and each plane is deflated on its own. Planes keep the bytes that
//!   vary (the low ones) apart from those that are nearly always zero, which
//!   is what lets a general compressor come close to the entropy of count
//!   data: measured 1.24 bytes a value on the human fetal atlas, against 8.
//!
//! Decoding gives back exactly the f32 and u32 arrays the plain layout holds
//! (a value is narrowed only when it comes back bit for bit), so every result
//! computed from a compact store is identical to one computed from a plain
//! store. Both layouts keep `{prefix}.indptr` (u64 row pointers, resident when
//! the store is open). A compact store adds `{prefix}.cz` (the chunks one after
//! another) and `{prefix}.czi` (a magic word, the chunk size in rows, then the
//! byte offset of every chunk and of the end). Every chunk carries a CRC-32 of
//! its planes, checked before decoding: raw deflate has no checksum of its own,
//! and a damaged byte must be an error, never a wrong count.
//!
//! The layout of new stores is compact unless `POLARISEQ_STORE=plain`; a store
//! is read in whichever layout it was written.

use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;
use flate2::Compression;
use rayon::prelude::*;

/// Rows per compact chunk. Block sizes are multiples of 1,024 rows (the
/// planner quantizes them so), so a block read decodes whole chunks.
pub const CHUNK_ROWS: usize = 1024;
const CHUNK_MAGIC: &[u8; 4] = b"PQC1";
/// Magic, flags, values in the chunk, CRC-32 of the planes.
const HEADER: usize = 16;
const INDEX_MAGIC: &[u8; 8] = b"PQCZIDX1";
/// Deflate level: 1 is several times faster than 6 to write and gives up a
/// few percent of size; decoding speed does not depend on it.
const LEVEL: u32 = 1;

/// How a store is laid out on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// f32 values and u32 gene ids, 8 bytes a stored value.
    Plain,
    /// Byte planes deflated per chunk of rows; lossless.
    Compact,
}

impl Layout {
    /// The layout new stores are written in: compact, unless the environment
    /// sets `POLARISEQ_STORE=plain`.
    #[must_use]
    pub fn for_new_stores() -> Self {
        match std::env::var("POLARISEQ_STORE").ok().as_deref() {
            Some("plain") => Self::Plain,
            _ => Self::Compact,
        }
    }

    /// The layout of the store at `prefix`, from the files that are there.
    #[must_use]
    pub fn of(prefix: &Path) -> Self {
        if prefix.with_extension("czi").exists() {
            Self::Compact
        } else {
            Self::Plain
        }
    }
}

/// Every file a store at `prefix` may consist of, in either layout.
pub const STORE_EXTENSIONS: [&str; 5] = ["data", "indices", "indptr", "cz", "czi"];

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

/// Writes a store row block by row block, in either layout.
pub struct StoreWriter {
    prefix: PathBuf,
    indptr: Vec<u64>,
    running: u64,
    sink: Sink,
}

enum Sink {
    Plain {
        data: BufWriter<File>,
        indices: BufWriter<File>,
    },
    Compact {
        file: BufWriter<File>,
        offsets: Vec<u64>,
        written: u64,
        data: Vec<f32>,
        indices: Vec<u32>,
        lens: Vec<u32>,
    },
}

impl StoreWriter {
    /// Start a store at `prefix`, removing any store already there (of either
    /// layout, so that a stale file of the other layout cannot be read in
    /// place of the new one).
    ///
    /// # Errors
    /// Returns an error if the files cannot be created.
    pub fn create(prefix: &Path, layout: Layout) -> io::Result<Self> {
        if let Some(dir) = prefix.parent() {
            std::fs::create_dir_all(dir)?;
        }
        for ext in STORE_EXTENSIONS {
            let _ = std::fs::remove_file(prefix.with_extension(ext));
        }
        let sink = match layout {
            Layout::Plain => Sink::Plain {
                data: BufWriter::new(File::create(prefix.with_extension("data"))?),
                indices: BufWriter::new(File::create(prefix.with_extension("indices"))?),
            },
            Layout::Compact => Sink::Compact {
                file: BufWriter::new(File::create(prefix.with_extension("cz"))?),
                offsets: vec![0],
                written: 0,
                data: Vec::new(),
                indices: Vec::new(),
                lens: Vec::new(),
            },
        };
        Ok(Self {
            prefix: prefix.to_path_buf(),
            indptr: vec![0],
            running: 0,
            sink,
        })
    }

    /// Append rows: their values and gene ids one row after another, and the
    /// length of each row.
    ///
    /// # Errors
    /// Returns an error if a write fails.
    ///
    /// # Panics
    /// Panics if `data`, `indices` and the sum of `lens` disagree.
    pub fn push(&mut self, data: &[f32], indices: &[u32], lens: &[u32]) -> io::Result<()> {
        assert_eq!(data.len(), indices.len(), "values and gene ids must pair up");
        assert_eq!(
            lens.iter().map(|&l| l as usize).sum::<usize>(),
            data.len(),
            "row lengths must add up to the values given"
        );
        for &l in lens {
            self.running += u64::from(l);
            self.indptr.push(self.running);
        }
        match &mut self.sink {
            Sink::Plain { data: dw, indices: iw } => {
                write_le(dw, data.iter().map(|v| v.to_le_bytes()))?;
                write_le(iw, indices.iter().map(|v| v.to_le_bytes()))?;
            }
            Sink::Compact { file, offsets, written, data: pd, indices: pi, lens: pl } => {
                pd.extend_from_slice(data);
                pi.extend_from_slice(indices);
                pl.extend_from_slice(lens);
                let rows = (pl.len() / CHUNK_ROWS) * CHUNK_ROWS;
                if rows > 0 {
                    flush_chunks(file, offsets, written, pd, pi, pl, rows)?;
                }
            }
        }
        Ok(())
    }

    /// Write what is pending and the row pointers; returns the stored values.
    ///
    /// # Errors
    /// Returns an error if a write fails.
    pub fn finish(mut self) -> io::Result<u64> {
        match &mut self.sink {
            Sink::Plain { data, indices } => {
                data.flush()?;
                indices.flush()?;
            }
            Sink::Compact { file, offsets, written, data, indices, lens } => {
                let rows = lens.len();
                if rows > 0 {
                    flush_chunks(file, offsets, written, data, indices, lens, rows)?;
                }
                file.flush()?;
                let mut w = BufWriter::new(File::create(self.prefix.with_extension("czi"))?);
                w.write_all(INDEX_MAGIC)?;
                w.write_all(&(CHUNK_ROWS as u64).to_le_bytes())?;
                write_le(&mut w, offsets.iter().map(|o| o.to_le_bytes()))?;
                w.flush()?;
            }
        }
        let mut w = BufWriter::new(File::create(self.prefix.with_extension("indptr"))?);
        write_le(&mut w, self.indptr.iter().map(|v| v.to_le_bytes()))?;
        w.flush()?;
        Ok(self.running)
    }
}

/// Write little-endian values through a bounded buffer.
fn write_le<W: Write, const N: usize>(w: &mut W, xs: impl Iterator<Item = [u8; N]>) -> io::Result<()> {
    let mut buf: Vec<u8> = Vec::with_capacity(1 << 18);
    for x in xs {
        buf.extend_from_slice(&x);
        if buf.len() >= 1 << 18 {
            w.write_all(&buf)?;
            buf.clear();
        }
    }
    w.write_all(&buf)
}

/// Encode the first `rows` pending rows as chunks, in parallel, and append
/// them in order.
fn flush_chunks(
    file: &mut BufWriter<File>,
    offsets: &mut Vec<u64>,
    written: &mut u64,
    data: &mut Vec<f32>,
    indices: &mut Vec<u32>,
    lens: &mut Vec<u32>,
    rows: usize,
) -> io::Result<()> {
    let mut off = Vec::with_capacity(rows + 1);
    off.push(0_usize);
    for &l in &lens[..rows] {
        off.push(off[off.len() - 1] + l as usize);
    }
    let bounds: Vec<(usize, usize)> = (0..rows)
        .step_by(CHUNK_ROWS)
        .map(|a| (a, (a + CHUNK_ROWS).min(rows)))
        .collect();
    let chunks: Vec<Vec<u8>> = bounds
        .par_iter()
        .map(|&(a, b)| {
            encode_chunk(&data[off[a]..off[b]], &indices[off[a]..off[b]], &lens[a..b])
        })
        .collect();
    for c in chunks {
        file.write_all(&c)?;
        *written += c.len() as u64;
        offsets.push(*written);
    }
    data.drain(..off[rows]);
    indices.drain(..off[rows]);
    lens.drain(..rows);
    Ok(())
}

/// Whether `v` survives a trip through u16: a whole number from 0 to 65,535
/// whose f32 bits come back unchanged (so not -0.0, not NaN).
fn narrows(v: f32) -> bool {
    let u = v as u32;
    u16::try_from(u).is_ok() && (u as f32).to_bits() == v.to_bits()
}

fn push_plane(out: &mut Vec<u8>, plane: &[u8]) {
    let mut enc = DeflateEncoder::new(Vec::with_capacity(plane.len() / 3 + 64), Compression::new(LEVEL));
    enc.write_all(plane).expect("deflate into memory cannot fail");
    let z = enc.finish().expect("deflate into memory cannot fail");
    out.extend_from_slice(&u32::try_from(z.len()).expect("a chunk plane is far below 4 GB").to_le_bytes());
    out.extend_from_slice(&z);
}

/// One chunk of rows as bytes: a header, then the gene-id planes, then the
/// value planes, each deflated.
///
/// # Panics
/// Panics if `lens` do not add up to the values given, or if a chunk holds
/// 2^32 values or more (a chunk is 1,024 rows).
#[must_use]
pub fn encode_chunk(data: &[f32], indices: &[u32], lens: &[u32]) -> Vec<u8> {
    let nnz = data.len();
    let mut ids: Vec<u32> = Vec::with_capacity(nnz);
    let mut sorted = true;
    let mut p = 0_usize;
    'rows: for &l in lens {
        let row = &indices[p..p + l as usize];
        for (j, &c) in row.iter().enumerate() {
            if j == 0 {
                ids.push(c);
            } else if c > row[j - 1] {
                ids.push(c - row[j - 1] - 1);
            } else {
                sorted = false;
                break 'rows;
            }
        }
        p += l as usize;
    }
    if !sorted {
        ids.clear();
        ids.extend_from_slice(indices);
    }
    let id_bytes: usize = if ids.iter().all(|&v| u16::try_from(v).is_ok()) { 2 } else { 4 };
    let small = data.iter().all(|&v| narrows(v));
    let value_bytes: usize = if small { 2 } else { 4 };

    let mut out = Vec::with_capacity(16 + nnz);
    out.extend_from_slice(CHUNK_MAGIC);
    out.push(u8::from(!sorted));
    out.push(id_bytes as u8);
    out.push(u8::from(!small));
    out.push(0);
    out.extend_from_slice(&u32::try_from(nnz).expect("a chunk holds fewer than 2^32 values").to_le_bytes());
    out.extend_from_slice(&[0; 4]); // the checksum, filled in below
    let mut plane: Vec<u8> = Vec::with_capacity(nnz);
    for b in 0..id_bytes {
        plane.clear();
        plane.extend(ids.iter().map(|&v| (v >> (8 * b)) as u8));
        push_plane(&mut out, &plane);
    }
    for b in 0..value_bytes {
        plane.clear();
        if small {
            plane.extend(data.iter().map(|&v| ((v as u32) >> (8 * b)) as u8));
        } else {
            plane.extend(data.iter().map(|&v| (v.to_bits() >> (8 * b)) as u8));
        }
        push_plane(&mut out, &plane);
    }
    let crc = crc32fast::hash(&out[HEADER..]);
    out[12..HEADER].copy_from_slice(&crc.to_le_bytes());
    out
}

fn inflate_plane(buf: &[u8], pos: &mut usize, n: usize, out: &mut Vec<u8>) -> io::Result<()> {
    if *pos + 4 > buf.len() {
        return Err(bad("compact chunk ends inside a plane header"));
    }
    let len = u32::from_le_bytes([buf[*pos], buf[*pos + 1], buf[*pos + 2], buf[*pos + 3]]) as usize;
    *pos += 4;
    if *pos + len > buf.len() {
        return Err(bad("compact chunk ends inside a plane"));
    }
    out.clear();
    out.resize(n, 0);
    let mut dec = DeflateDecoder::new(&buf[*pos..*pos + len]);
    dec.read_exact(out).map_err(|_| bad("compact chunk plane is shorter than its values"))?;
    let mut rest = [0_u8; 1];
    if dec.read(&mut rest)? != 0 {
        return Err(bad("compact chunk plane is longer than its values"));
    }
    *pos += len;
    Ok(())
}

/// Decode a chunk written by [`encode_chunk`], appending its values and gene
/// ids to `data` and `indices`. `lens` are its rows' lengths (from the row
/// pointers); `scratch` is reused between calls.
///
/// # Errors
/// Returns an error if the bytes are not a well-formed chunk of these rows.
pub fn decode_chunk(
    buf: &[u8],
    lens: &[u32],
    data: &mut Vec<f32>,
    indices: &mut Vec<u32>,
    scratch: &mut Vec<u8>,
) -> io::Result<()> {
    if buf.len() < HEADER || &buf[..4] != CHUNK_MAGIC {
        return Err(bad("not a compact chunk"));
    }
    let crc = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
    if crc32fast::hash(&buf[HEADER..]) != crc {
        return Err(bad("compact chunk fails its checksum: the store is damaged"));
    }
    let absolute = buf[4] != 0;
    let id_bytes = buf[5] as usize;
    let floats = buf[6] != 0;
    let nnz = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
    if lens.iter().map(|&l| l as usize).sum::<usize>() != nnz || !matches!(id_bytes, 2 | 4) {
        return Err(bad("compact chunk does not match its rows"));
    }
    let mut pos = HEADER;

    let i0 = indices.len();
    indices.resize(i0 + nnz, 0);
    for b in 0..id_bytes {
        inflate_plane(buf, &mut pos, nnz, scratch)?;
        for (dst, &byte) in indices[i0..].iter_mut().zip(scratch.iter()) {
            *dst |= u32::from(byte) << (8 * b);
        }
    }
    if !absolute {
        let mut p = i0;
        for &l in lens {
            let mut prev = 0_u32;
            for j in 0..l as usize {
                let v = indices[p + j];
                let c = if j == 0 { v } else { prev + v + 1 };
                indices[p + j] = c;
                prev = c;
            }
            p += l as usize;
        }
    }

    let value_bytes = if floats { 4 } else { 2 };
    let mut acc = vec![0_u32; nnz];
    for b in 0..value_bytes {
        inflate_plane(buf, &mut pos, nnz, scratch)?;
        for (dst, &byte) in acc.iter_mut().zip(scratch.iter()) {
            *dst |= u32::from(byte) << (8 * b);
        }
    }
    if floats {
        data.extend(acc.into_iter().map(f32::from_bits));
    } else {
        data.extend(acc.into_iter().map(|v| v as f32));
    }
    Ok(())
}

/// Read a compact store's chunk index: the chunk size in rows and the byte
/// offset of every chunk and of the end.
///
/// # Errors
/// Returns an error if the index is missing or does not describe `nrows` rows.
///
/// # Panics
/// Does not panic: the slices converted are exactly eight bytes long.
pub fn read_index(prefix: &Path, nrows: usize) -> io::Result<(usize, Vec<u64>)> {
    let bytes = std::fs::read(prefix.with_extension("czi"))?;
    if bytes.len() < 16 || &bytes[..8] != INDEX_MAGIC {
        return Err(bad("not a compact store index"));
    }
    let chunk_rows = u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes")) as usize;
    let offsets: Vec<u64> = bytes[16..]
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes")))
        .collect();
    if chunk_rows == 0 || offsets.len() != nrows.div_ceil(chunk_rows) + 1 {
        return Err(bad("compact store index does not match the number of rows"));
    }
    Ok((chunk_rows, offsets))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(seed: u64, n_rows: usize, n_cols: u32, density: f64, floats: bool) -> (Vec<f32>, Vec<u32>, Vec<u32>) {
        // A small deterministic generator: rows sorted by gene, counts mostly 1.
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let (mut d, mut ix, mut lens) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..n_rows {
            let mut l = 0;
            for c in 0..n_cols {
                if (next() % 10_000) as f64 / 10_000.0 < density {
                    ix.push(c);
                    let r = next();
                    d.push(if floats { (r % 1000) as f32 / 7.0 } else { (1 + (r % 7) * (r % 3)) as f32 });
                    l += 1;
                }
            }
            lens.push(l);
        }
        (d, ix, lens)
    }

    fn round_trip(d: &[f32], ix: &[u32], lens: &[u32]) {
        let buf = encode_chunk(d, ix, lens);
        let (mut d2, mut i2, mut s) = (Vec::new(), Vec::new(), Vec::new());
        decode_chunk(&buf, lens, &mut d2, &mut i2, &mut s).unwrap();
        assert_eq!(i2, ix);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&d2), bits(d), "values must come back bit for bit");
    }

    #[test]
    fn counts_round_trip_exactly() {
        let (d, ix, lens) = rows(7, 300, 2000, 0.05, false);
        round_trip(&d, &ix, &lens);
    }

    #[test]
    fn floats_and_wide_gene_ids_round_trip_exactly() {
        let (d, ix, lens) = rows(11, 50, 70_000, 0.002, true);
        assert!(ix.iter().any(|&c| c > 65_535));
        round_trip(&d, &ix, &lens);
    }

    #[test]
    fn values_that_do_not_narrow_keep_their_bits() {
        for v in [-0.0_f32, f32::NAN, -3.0, 0.5, 65_536.0, 1e9] {
            round_trip(&[1.0, v, 2.0], &[0, 5, 9], &[3]);
        }
    }

    #[test]
    fn unsorted_rows_and_empty_rows_round_trip() {
        round_trip(&[1.0, 2.0, 3.0, 4.0], &[9, 2, 2, 7], &[0, 3, 0, 1]);
    }

    #[test]
    fn counts_compress_well_below_eight_bytes_a_value() {
        let (d, ix, lens) = rows(3, 1024, 20_000, 0.08, false);
        let bytes = encode_chunk(&d, &ix, &lens).len() as f64 / d.len() as f64;
        assert!(bytes < 3.0, "{bytes:.2} bytes a value");
    }

    #[test]
    fn a_damaged_chunk_is_an_error_not_wrong_numbers() {
        let (d, ix, lens) = rows(5, 20, 500, 0.1, false);
        let mut buf = encode_chunk(&d, &ix, &lens);
        let n = buf.len();
        buf[n / 2] ^= 0xFF;
        let (mut d2, mut i2, mut s) = (Vec::new(), Vec::new(), Vec::new());
        let r = decode_chunk(&buf, &lens, &mut d2, &mut i2, &mut s);
        assert!(r.is_err(), "corruption must be an error");
        buf.truncate(n - 3);
        assert!(decode_chunk(&buf, &lens, &mut Vec::new(), &mut Vec::new(), &mut s).is_err());
    }

    #[test]
    fn the_writer_produces_a_readable_store_in_both_layouts() {
        let dir = std::env::temp_dir().join(format!("pq_store_{}", std::process::id()));
        let (d, ix, lens) = rows(13, 2500, 3000, 0.03, false);
        for layout in [Layout::Plain, Layout::Compact] {
            let prefix = dir.join(format!("{layout:?}"));
            let mut w = StoreWriter::create(&prefix, layout).unwrap();
            // Pushed in uneven pieces, so chunks straddle pushes.
            let (mut r, mut v) = (0, 0);
            for piece in [700_usize, 1, 1500, 299] {
                let n: usize = lens[r..r + piece].iter().map(|&l| l as usize).sum();
                w.push(&d[v..v + n], &ix[v..v + n], &lens[r..r + piece]).unwrap();
                r += piece;
                v += n;
            }
            assert_eq!(w.finish().unwrap() as usize, d.len());
            assert_eq!(Layout::of(&prefix), layout);
            if layout == Layout::Compact {
                let (cr, offsets) = read_index(&prefix, lens.len()).unwrap();
                assert_eq!((cr, offsets.len()), (CHUNK_ROWS, 2500_usize.div_ceil(CHUNK_ROWS) + 1));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
