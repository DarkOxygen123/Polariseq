//! Several `.h5ad` files read as one matrix: their cells one after another,
//! over one set of genes.
//!
//! Genes are matched by name before the matrix is read, which gives each file
//! a map from its columns to the merged ones; the maps are applied while the
//! rows stream past, so the merged matrix is never assembled from copies of
//! the files. Each row's entries come out in merged-column order, so the
//! matrix is the one a physically merged, sorted file holds, whether it is
//! kept in memory ([`read_many`]) or written to the block store
//! ([`crate::spill::spill_h5ad_many`]).

use polariseq_core::rowblocks::RowBlocks;
use polariseq_core::CsrMatrix;

use crate::stream::H5adStream;
use crate::{IoError, Result};

/// A map entry for a gene that is not in the merged set.
pub const DROPPED: u32 = u32::MAX;

/// Stream the merged matrix a block of rows at a time: `sink` receives each
/// block's values and merged column indices, row after row, and the entry
/// count of each row. Returns the number of rows.
///
/// `gene_maps[k][j]` is the merged column of gene `j` of file `k`, or
/// [`DROPPED`].
///
/// # Errors
/// Returns [`IoError`] on a read failure, if a map does not match its file's
/// genes or points past `n_cols`, or as `sink` does.
pub fn for_each_block(
    paths: &[String],
    gene_maps: &[Vec<u32>],
    n_cols: usize,
    block_rows: usize,
    mut sink: impl FnMut(&[f32], &[u32], &[u32]) -> Result<()>,
) -> Result<usize> {
    if paths.len() != gene_maps.len() || paths.is_empty() {
        return Err(IoError::InvalidLayout(
            "one gene map per file, and at least one file".into(),
        ));
    }
    let block_rows = block_rows.max(1);
    let mut n_rows = 0_usize;
    let mut row: Vec<(u32, f32)> = Vec::new();
    let (mut data, mut indices, mut lens) = (Vec::new(), Vec::new(), Vec::new());
    for (path, map) in paths.iter().zip(gene_maps) {
        let rows_here = {
            let stream = H5adStream::open(path)?;
            if map.len() != stream.n_cols() {
                return Err(IoError::InvalidLayout(format!(
                    "{path}: gene map has {} entries for {} genes",
                    map.len(),
                    stream.n_cols()
                )));
            }
            stream.n_rows()
        };
        if map.iter().any(|&m| m != DROPPED && m as usize >= n_cols) {
            return Err(IoError::InvalidLayout(format!(
                "{path}: gene map points past the {n_cols} merged genes"
            )));
        }
        let mut r0 = 0;
        while r0 < rows_here {
            let r1 = (r0 + block_rows).min(rows_here);
            let blk = crate::lazy::read_rows_any(path, r0, r1)?;
            data.clear();
            indices.clear();
            lens.clear();
            for i in 0..blk.nrows {
                let (a, b) = (blk.indptr[i] as usize, blk.indptr[i + 1] as usize);
                row.clear();
                for (&c, &v) in blk.indices[a..b].iter().zip(&blk.data[a..b]) {
                    let m = map[c as usize];
                    if m != DROPPED {
                        row.push((m, v));
                    }
                }
                row.sort_unstable_by_key(|&(m, _)| m);
                for &(m, v) in &row {
                    indices.push(m);
                    data.push(v);
                }
                // A row has at most n_cols entries, which fit u32.
                lens.push(row.len() as u32);
            }
            sink(&data, &indices, &lens)?;
            r0 = r1;
        }
        n_rows += rows_here;
    }
    Ok(n_rows)
}

/// The merged matrix, in memory. `nnz_hint` (the files' entries summed, an
/// upper bound) reserves the arrays once; pages beyond the entries kept are
/// never touched.
///
/// # Errors
/// As [`for_each_block`], or if the merged matrix exceeds the u32
/// row-pointer limit.
pub fn read_many(
    paths: &[String],
    gene_maps: &[Vec<u32>],
    n_cols: usize,
    block_rows: usize,
    nnz_hint: usize,
) -> Result<CsrMatrix> {
    let (mut data, mut indices) = (Vec::with_capacity(nnz_hint), Vec::with_capacity(nnz_hint));
    let mut indptr: Vec<u32> = vec![0];
    let mut running: u64 = 0;
    let n_rows = for_each_block(paths, gene_maps, n_cols, block_rows, |d, ix, lens| {
        data.extend_from_slice(d);
        indices.extend_from_slice(ix);
        for &l in lens {
            running += u64::from(l);
            indptr.push(u32::try_from(running).map_err(|_| {
                IoError::InvalidLayout(format!(
                    "merged matrix exceeds the u32 indptr limit at row {}",
                    indptr.len() - 1
                ))
            })?);
        }
        Ok(())
    })?;
    Ok(CsrMatrix::new(indptr, indices, data, n_rows, n_cols)?)
}

/// The files' raw counts summed per unit, streamed with their gene maps (see
/// [`polariseq_core::pseudobulk::unit_sums_blocks`]): `unit[i]` is the unit
/// of row `i` of the files read one after another, or `u32::MAX` to leave it
/// out. This is how an analysis whose matrix is already normalized in memory
/// gets back its cells' raw counts, without holding a second copy.
///
/// # Errors
/// As [`for_each_block`], or if `unit` does not have one entry per row.
pub fn unit_sums(
    paths: &[String],
    gene_maps: &[Vec<u32>],
    n_cols: usize,
    unit: &[u32],
    n_units: usize,
    block_rows: usize,
) -> Result<polariseq_core::pseudobulk::UnitSums> {
    let mut sums = vec![0.0_f64; n_units * n_cols];
    let mut expressing = vec![0_u32; n_units * n_cols];
    let mut cells = vec![0_u32; n_units];
    let mut row = 0_usize;
    let mut too_few = false;
    let n_rows = for_each_block(paths, gene_maps, n_cols, block_rows, |d, ix, lens| {
        let mut at = 0_usize;
        for &l in lens {
            let l = l as usize;
            let Some(&u) = unit.get(row) else {
                too_few = true;
                return Ok(());
            };
            if (u as usize) < n_units {
                cells[u as usize] += 1;
                let base = u as usize * n_cols;
                for (&c, &x) in ix[at..at + l].iter().zip(&d[at..at + l]) {
                    sums[base + c as usize] += f64::from(x);
                    expressing[base + c as usize] += u32::from(x > 0.0);
                }
            }
            at += l;
            row += 1;
        }
        Ok(())
    })?;
    if too_few || n_rows != unit.len() {
        return Err(IoError::InvalidLayout(format!(
            "{} units for the files' {n_rows} rows",
            unit.len()
        )));
    }
    Ok(polariseq_core::pseudobulk::UnitSums {
        n_units,
        n_genes: n_cols,
        sums,
        expressing,
        cells,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> String {
        format!(
            "{}/../../tests/data/sparse_100x200.h5ad",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    /// The fixture read twice as one matrix: once as it is, once with its
    /// genes reversed and the first ten dropped. Each half must hold the
    /// file's rows with their entries moved to the merged columns, sorted.
    #[test]
    fn the_merged_matrix_is_each_file_remapped_and_sorted() {
        let path = fixture();
        let file = crate::lazy::read_rows_any(&path, 0, 100).unwrap();
        let g = file.ncols;
        let identity: Vec<u32> = (0..g as u32).collect();
        let reversed: Vec<u32> = (0..g)
            .map(|j| if j < 10 { DROPPED } else { (g - 1 - j) as u32 })
            .collect();
        let paths = vec![path.clone(), path];
        let m = read_many(&paths, &[identity, reversed.clone()], g, 7, 0).unwrap();
        assert_eq!((m.nrows, m.ncols), (200, g));
        for i in 0..100 {
            let (a, b) = (file.indptr[i] as usize, file.indptr[i + 1] as usize);
            let (c, d) = (m.indptr[i] as usize, m.indptr[i + 1] as usize);
            assert_eq!(&m.indices[c..d], &file.indices[a..b]);
            assert_eq!(&m.data[c..d], &file.data[a..b]);
            let mut want: Vec<(u32, f32)> = file.indices[a..b]
                .iter()
                .zip(&file.data[a..b])
                .filter(|&(&j, _)| reversed[j as usize] != DROPPED)
                .map(|(&j, &v)| (reversed[j as usize], v))
                .collect();
            want.sort_unstable_by_key(|&(j, _)| j);
            let (c, d) = (m.indptr[100 + i] as usize, m.indptr[101 + i] as usize);
            let got: Vec<(u32, f32)> = m.indices[c..d]
                .iter()
                .copied()
                .zip(m.data[c..d].iter().copied())
                .collect();
            assert_eq!(got, want, "row {i} of the second file");
        }
    }

    /// Sums streamed from the files equal sums over the merged matrix.
    #[test]
    fn unit_sums_from_the_files_equal_those_of_the_merged_matrix() {
        let path = fixture();
        let g = crate::lazy::read_rows_any(&path, 0, 100).unwrap().ncols;
        let identity: Vec<u32> = (0..g as u32).collect();
        let paths = vec![path.clone(), path];
        let maps = [identity.clone(), identity];
        let unit: Vec<u32> = (0..200)
            .map(|i| if i % 7 == 0 { u32::MAX } else { (i % 3) as u32 })
            .collect();
        let got = unit_sums(&paths, &maps, g, &unit, 3, 16).unwrap();
        let m = read_many(&paths, &maps, g, 16, 0).unwrap();
        let want = polariseq_core::pseudobulk::unit_sums_blocks(&m, &unit, 3);
        assert_eq!(got, want);
        assert!(unit_sums(&paths, &maps, g, &unit[..150], 3, 16).is_err());
    }

    #[test]
    fn a_map_that_does_not_fit_its_file_is_refused() {
        let path = fixture();
        assert!(read_many(std::slice::from_ref(&path), &[vec![0; 3]], 200, 16, 0).is_err());
        assert!(read_many(&[path], &[vec![500; 200]], 200, 16, 0).is_err());
    }
}
