//! `H5Writer`: the HDF5 primitives `write_h5ad` lays the anndata format out
//! with (groups, attributes, arrays, text, scalars), and a CSR matrix written
//! from row blocks, so an out-of-core matrix goes to the file a few blocks at a
//! time and never whole into memory.

use std::sync::mpsc::sync_channel;
use std::sync::Mutex;

use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::{Dataset, File, H5Type, Location};
use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use polariseq_core::rowblocks::RowBlocks;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::disk::PyDiskMatrix;
use crate::sparse::PySparseMatrix;

fn h5err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(format!("writing the file failed: {e}"))
}

fn text(s: &str) -> PyResult<VarLenUnicode> {
    s.parse::<VarLenUnicode>()
        .map_err(|e| PyValueError::new_err(format!("text {s:?} cannot be stored: {e}")))
}

#[pyclass(name = "H5Writer", module = "polariseq._polariseq", unsendable)]
pub struct PyH5Writer {
    file: Option<File>,
}

impl PyH5Writer {
    fn file(&self) -> PyResult<&File> {
        self.file
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("the file is closed"))
    }

    /// The group or dataset at `path`, for its attributes.
    fn location(&self, path: &str) -> PyResult<Location> {
        let f = self.file()?;
        if path == "/" || path.is_empty() {
            return Ok((***f).clone());
        }
        if let Ok(g) = f.group(path) {
            return Ok((*g).clone());
        }
        f.dataset(path).map(|d| (**d).clone()).map_err(h5err)
    }

    fn put<T: H5Type + numpy::Element + Copy>(
        &self,
        path: &str,
        arr: &PyReadonlyArrayDyn<'_, T>,
    ) -> PyResult<()> {
        let shape: Vec<usize> = arr.shape().to_vec();
        let data = arr
            .as_slice()
            .map_err(|_| PyValueError::new_err("array is not contiguous"))?;
        let ds = self
            .file()?
            .new_dataset::<T>()
            .shape(shape)
            .create(path)
            .map_err(h5err)?;
        ds.write_raw(data).map_err(h5err)
    }
}

/// Write `src` as a CSR group's `data`, `indices`, `indptr` under `group`:
/// one pass counts each row's values (the row pointers), a second hands the
/// blocks, in any order, to this thread, which writes each where it belongs.
fn write_csr(
    file: &File,
    group: &str,
    src: &dyn RowBlocks,
    block_rows: usize,
) -> Result<(), String> {
    let (n, g) = (src.n_rows(), src.n_cols());
    let rpb = block_rows.max(1);
    let counts = Mutex::new(vec![0_u32; n]);
    src.for_each_block(rpb, &|off, blk| {
        let c: Vec<u32> = (0..blk.n_rows)
            .map(|i| blk.indptr[i + 1] - blk.indptr[i])
            .collect();
        if let Ok(mut all) = counts.lock() {
            all[off..off + c.len()].copy_from_slice(&c);
        }
    });
    let counts = counts.into_inner().map_err(|e| e.to_string())?;
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0_i64);
    for c in &counts {
        indptr.push(indptr[indptr.len() - 1] + i64::from(*c));
    }
    drop(counts);
    let nnz = usize::try_from(indptr[n]).map_err(|e| e.to_string())?;
    // anndata writes 32-bit indices and pointers where they fit
    let wide = nnz >= i32::MAX as usize || g >= i32::MAX as usize;
    let grp = file.group(group).map_err(|e| e.to_string())?;
    let data_ds = grp
        .new_dataset::<f32>()
        .shape(nnz)
        .create("data")
        .map_err(|e| e.to_string())?;
    let idx_ds: Dataset = if wide {
        grp.new_dataset::<i64>().shape(nnz).create("indices")
    } else {
        grp.new_dataset::<i32>().shape(nnz).create("indices")
    }
    .map_err(|e| e.to_string())?;
    if wide {
        let ds = grp
            .new_dataset::<i64>()
            .shape(n + 1)
            .create("indptr")
            .map_err(|e| e.to_string())?;
        ds.write_raw(&indptr).map_err(|e| e.to_string())?;
    } else {
        let ip: Vec<i32> = indptr.iter().map(|&v| v as i32).collect();
        let ds = grp
            .new_dataset::<i32>()
            .shape(n + 1)
            .create("indptr")
            .map_err(|e| e.to_string())?;
        ds.write_raw(&ip).map_err(|e| e.to_string())?;
    }
    if nnz == 0 {
        return Ok(());
    }
    let mut failed: Option<String> = None;
    std::thread::scope(|sc| {
        let (tx, rx) = sync_channel::<(usize, Vec<f32>, Vec<u32>)>(4);
        sc.spawn(move || {
            src.for_each_block(rpb, &|off, blk| {
                // a closed channel means the writer stopped on an error; stop sending
                let _ = tx.send((off, blk.data.to_vec(), blk.indices.to_vec()));
            });
        });
        for (off, data, indices) in rx {
            if failed.is_some() || data.is_empty() {
                continue;
            }
            let a = indptr[off] as usize;
            let b = a + data.len();
            let r = data_ds.write_slice(&data, a..b).and_then(|()| {
                if wide {
                    let ix: Vec<i64> = indices.iter().map(|&v| i64::from(v)).collect();
                    idx_ds.write_slice(&ix, a..b)
                } else {
                    let ix: Vec<i32> = indices.iter().map(|&v| v as i32).collect();
                    idx_ds.write_slice(&ix, a..b)
                }
            });
            if let Err(e) = r {
                failed = Some(e.to_string());
            }
        }
    });
    failed.map_or(Ok(()), Err)
}

#[pymethods]
impl PyH5Writer {
    /// Create (or replace) the file at `path`.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        Ok(Self {
            file: Some(File::create(path).map_err(h5err)?),
        })
    }

    /// A group at `path` (its parent must exist).
    fn group(&self, path: &str) -> PyResult<()> {
        self.file()?.create_group(path).map(|_| ()).map_err(h5err)
    }

    /// A text attribute.
    fn attr_str(&self, path: &str, name: &str, value: &str) -> PyResult<()> {
        let v = text(value)?;
        self.location(path)?
            .new_attr::<VarLenUnicode>()
            .create(name)
            .and_then(|a| a.write_scalar(&v))
            .map_err(h5err)
    }

    /// An attribute holding a list of texts.
    fn attr_strs(&self, path: &str, name: &str, values: Vec<String>) -> PyResult<()> {
        let v = values
            .iter()
            .map(|s| text(s))
            .collect::<PyResult<Vec<_>>>()?;
        self.location(path)?
            .new_attr::<VarLenUnicode>()
            .shape(v.len())
            .create(name)
            .and_then(|a| a.write_raw(&v))
            .map_err(h5err)
    }

    /// An attribute holding integers (one or several).
    fn attr_ints(&self, path: &str, name: &str, values: Vec<i64>) -> PyResult<()> {
        self.location(path)?
            .new_attr::<i64>()
            .shape(values.len())
            .create(name)
            .and_then(|a| a.write_raw(&values))
            .map_err(h5err)
    }

    /// A boolean attribute.
    fn attr_bool(&self, path: &str, name: &str, value: bool) -> PyResult<()> {
        self.location(path)?
            .new_attr::<bool>()
            .create(name)
            .and_then(|a| a.write_scalar(&value))
            .map_err(h5err)
    }

    /// A numeric or boolean array (any number of dimensions, C order).
    fn array(&self, path: &str, arr: &Bound<'_, PyAny>) -> PyResult<()> {
        macro_rules! try_put {
            ($($t:ty),*) => {$(
                if let Ok(a) = arr.extract::<PyReadonlyArrayDyn<'_, $t>>() {
                    return self.put::<$t>(path, &a);
                }
            )*};
        }
        try_put!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, bool);
        Err(PyTypeError::new_err(format!(
            "{path}: an array of numbers or booleans is expected"
        )))
    }

    /// A one-dimensional array of texts.
    fn strings(&self, path: &str, values: Vec<String>) -> PyResult<()> {
        let v = values
            .iter()
            .map(|s| text(s))
            .collect::<PyResult<Vec<_>>>()?;
        let ds = self
            .file()?
            .new_dataset::<VarLenUnicode>()
            .shape(v.len())
            .create(path)
            .map_err(h5err)?;
        ds.write_raw(&v).map_err(h5err)
    }

    /// A text scalar.
    fn scalar_str(&self, path: &str, value: &str) -> PyResult<()> {
        let v = text(value)?;
        let ds = self
            .file()?
            .new_dataset::<VarLenUnicode>()
            .create(path)
            .map_err(h5err)?;
        ds.write_scalar(&v).map_err(h5err)
    }

    /// A number scalar (integer, float or boolean).
    fn scalar(&self, path: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let f = self.file()?;
        if let Ok(b) = value.extract::<bool>() {
            if value.is_instance_of::<pyo3::types::PyBool>() {
                let ds = f.new_dataset::<bool>().create(path).map_err(h5err)?;
                return ds.write_scalar(&b).map_err(h5err);
            }
        }
        if let Ok(i) = value.extract::<i64>() {
            let ds = f.new_dataset::<i64>().create(path).map_err(h5err)?;
            return ds.write_scalar(&i).map_err(h5err);
        }
        let x: f64 = value.extract()?;
        let ds = f.new_dataset::<f64>().create(path).map_err(h5err)?;
        ds.write_scalar(&x).map_err(h5err)
    }

    /// The current values of `matrix` (a `SparseMatrix`, or a `DiskMatrix`
    /// through its filters and normalization) as CSR under the existing group
    /// `group`, `block_rows` rows at a time.
    #[pyo3(signature = (group, matrix, block_rows=65536))]
    fn csr(
        &self,
        py: Python<'_>,
        group: &str,
        matrix: &Bound<'_, PyAny>,
        block_rows: usize,
    ) -> PyResult<()> {
        let file = self.file()?;
        let r = if let Ok(m) = matrix.extract::<PyRef<'_, PySparseMatrix>>() {
            let inner = &m.inner;
            py.detach(|| write_csr(file, group, inner, block_rows))
        } else if let Ok(m) = matrix.extract::<PyRef<'_, PyDiskMatrix>>() {
            let dm: &PyDiskMatrix = &m;
            py.detach(|| dm.current_view(|v| write_csr(file, group, v, block_rows)))
        } else {
            return Err(PyTypeError::new_err(
                "csr takes a SparseMatrix or a DiskMatrix",
            ));
        };
        r.map_err(h5err)
    }

    /// Flush and close the file.
    fn close(&mut self) -> PyResult<()> {
        if let Some(f) = self.file.take() {
            f.close().map_err(h5err)?;
        }
        Ok(())
    }
}
