//! `read_h5ad_frame`: the `obs`/`var` columns of an `.h5ad`, handed to Python
//! in their narrowest form. Numbers move into numpy without a copy; text
//! arrives as codes into its distinct values, which Python wraps as a
//! categorical, so no string object is made per row.

use numpy::IntoPyArray;
use polariseq_io::frame::{Categories, Column, Numbers};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn numbers<'py>(py: Python<'py>, v: Numbers) -> Bound<'py, PyAny> {
    match v {
        Numbers::I8(v) => v.into_pyarray(py).into_any(),
        Numbers::I16(v) => v.into_pyarray(py).into_any(),
        Numbers::I32(v) => v.into_pyarray(py).into_any(),
        Numbers::I64(v) => v.into_pyarray(py).into_any(),
        Numbers::U8(v) => v.into_pyarray(py).into_any(),
        Numbers::U16(v) => v.into_pyarray(py).into_any(),
        Numbers::U32(v) => v.into_pyarray(py).into_any(),
        Numbers::U64(v) => v.into_pyarray(py).into_any(),
        Numbers::F32(v) => v.into_pyarray(py).into_any(),
        Numbers::F64(v) => v.into_pyarray(py).into_any(),
        Numbers::Bool(v) => v.into_pyarray(py).into_any(),
    }
}

fn column<'py>(py: Python<'py>, col: Column) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match col {
        Column::Categorical {
            codes,
            categories,
            ordered,
        } => {
            d.set_item("kind", "categorical")?;
            d.set_item("codes", numbers(py, codes))?;
            match categories {
                Categories::Text(t) => d.set_item("categories", t)?,
                Categories::Numbers(n) => d.set_item("categories", numbers(py, n))?,
            }
            d.set_item("ordered", ordered)?;
        }
        Column::Text { codes, uniques } => {
            d.set_item("kind", "text")?;
            d.set_item("codes", codes.into_pyarray(py))?;
            d.set_item("categories", uniques)?;
        }
        Column::Numbers(n) => {
            d.set_item("kind", "numbers")?;
            d.set_item("values", numbers(py, n))?;
        }
        Column::Nullable { values, mask } => {
            d.set_item("kind", "nullable")?;
            d.set_item("values", numbers(py, values))?;
            d.set_item("mask", mask.into_pyarray(py))?;
        }
    }
    Ok(d)
}

/// The columns of an `.h5ad`'s `obs` (or `var`), or only those named.
///
/// Returns `{"columns": [(name, column), ...], "skipped": [(name, encoding)]}`
/// where each column is a dict: `kind` "categorical" (`codes`, `categories`,
/// `ordered`), "text" (`codes`, `categories`: text as codes into its
/// distinct values), "numbers" (`values`) or "nullable" (`values`, `mask`,
/// true where missing). Codes of -1 are missing values. A named column the
/// file lacks is an error unless `missing_ok`.
#[pyfunction]
#[pyo3(signature = (path, axis="obs", columns=None, missing_ok=false))]
pub fn read_h5ad_frame<'py>(
    py: Python<'py>,
    path: &str,
    axis: &str,
    columns: Option<Vec<String>>,
    missing_ok: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let frame = py
        .detach(|| {
            polariseq_io::AnnDataFile::open(path)
                .and_then(|f| f.read_frame(axis, columns.as_deref(), missing_ok))
        })
        .map_err(|e| PyRuntimeError::new_err(format!("{path}: {e}")))?;
    let cols = PyList::empty(py);
    for (name, col) in frame.columns {
        cols.append((name, column(py, col)?))?;
    }
    let d = PyDict::new(py);
    d.set_item("columns", cols)?;
    d.set_item("skipped", frame.skipped)?;
    Ok(d)
}
