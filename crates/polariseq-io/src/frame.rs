//! The `obs`/`var` columns of an `.h5ad`, each read in its narrowest form.
//!
//! Every column encoding anndata writes is read: categorical (codes and
//! categories), numeric and boolean arrays in the dtype the file stores, text,
//! and the nullable integer, boolean and text arrays (values and a mask).
//! Text is returned deduplicated, as codes into its distinct values, so a
//! column of millions of repeated labels costs one small integer per cell
//! rather than one string each. Text is read a slice at a time, so a column
//! is never held twice. A column in an encoding this reader does not know is
//! listed as skipped, never guessed at.

use std::collections::HashMap;

use hdf5_metno::types::{FloatSize, IntSize, TypeDescriptor, VarLenAscii, VarLenUnicode};
use hdf5_metno::{Dataset, Group, Location};

use crate::h5ad::read_attr_string;
use crate::{IoError, Result};

/// Rows of text read per slice.
const TEXT_SLICE: usize = 1 << 20;

/// Numbers or booleans, in the width the file stores.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum Numbers {
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Bool(Vec<bool>),
}

impl Numbers {
    /// Number of values.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::I8(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::U8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::Bool(v) => v.len(),
        }
    }

    /// Whether there are no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The distinct values of a categorical column.
#[derive(Debug, Clone, PartialEq)]
pub enum Categories {
    /// Text categories.
    Text(Vec<String>),
    /// Numeric categories.
    Numbers(Numbers),
}

/// One column of a dataframe.
#[derive(Debug, Clone, PartialEq)]
pub enum Column {
    /// Codes into `categories`, in the file's integer width; -1 is missing.
    Categorical {
        /// One code per row.
        codes: Numbers,
        /// The distinct values.
        categories: Categories,
        /// Whether the categories are ordered.
        ordered: bool,
    },
    /// Text, as codes into its distinct values in order of first
    /// appearance; -1 is missing.
    Text {
        /// One code per row.
        codes: Vec<i32>,
        /// The distinct values.
        uniques: Vec<String>,
    },
    /// A plain numeric or boolean array.
    Numbers(Numbers),
    /// A nullable integer or boolean array; `mask[i]` is true where the value
    /// is missing.
    Nullable {
        /// The values (arbitrary where masked).
        values: Numbers,
        /// True where missing.
        mask: Vec<bool>,
    },
}

impl Column {
    /// Number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Categorical { codes, .. } => codes.len(),
            Self::Text { codes, .. } => codes.len(),
            Self::Numbers(v) => v.len(),
            Self::Nullable { values, .. } => values.len(),
        }
    }

    /// Whether there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The columns read from one dataframe group.
#[derive(Debug, Clone, Default)]
pub struct Frame {
    /// Columns in the file's order.
    pub columns: Vec<(String, Column)>,
    /// Columns present in an encoding this reader does not read, with the
    /// encoding.
    pub skipped: Vec<(String, String)>,
}

fn hdf5(e: hdf5_metno::Error) -> IoError {
    IoError::Hdf5(e)
}

/// Read a numeric or boolean dataset in the width the file stores.
fn read_numbers(ds: &Dataset) -> Result<Numbers> {
    let desc = ds.dtype().map_err(hdf5)?.to_descriptor().map_err(hdf5)?;
    Ok(match desc {
        TypeDescriptor::Integer(IntSize::U1) => Numbers::I8(ds.read_raw().map_err(hdf5)?),
        TypeDescriptor::Integer(IntSize::U2) => Numbers::I16(ds.read_raw().map_err(hdf5)?),
        TypeDescriptor::Integer(IntSize::U4) => Numbers::I32(ds.read_raw().map_err(hdf5)?),
        TypeDescriptor::Integer(IntSize::U8) => Numbers::I64(ds.read_raw().map_err(hdf5)?),
        TypeDescriptor::Unsigned(IntSize::U1) => Numbers::U8(ds.read_raw().map_err(hdf5)?),
        TypeDescriptor::Unsigned(IntSize::U2) => Numbers::U16(ds.read_raw().map_err(hdf5)?),
        TypeDescriptor::Unsigned(IntSize::U4) => Numbers::U32(ds.read_raw().map_err(hdf5)?),
        TypeDescriptor::Unsigned(IntSize::U8) => Numbers::U64(ds.read_raw().map_err(hdf5)?),
        TypeDescriptor::Float(FloatSize::U4) => Numbers::F32(ds.read_raw().map_err(hdf5)?),
        TypeDescriptor::Float(FloatSize::U8) => Numbers::F64(ds.read_raw().map_err(hdf5)?),
        // h5py writes booleans as an 8-bit enum {FALSE, TRUE}.
        TypeDescriptor::Boolean | TypeDescriptor::Enum(_) => {
            Numbers::Bool(ds.read_raw().map_err(hdf5)?)
        }
        other => {
            return Err(IoError::InvalidLayout(format!(
                "{}: values of type {other:?} are not numbers",
                ds.name()
            )))
        }
    })
}

fn is_text(ds: &Dataset) -> bool {
    matches!(
        ds.dtype().and_then(|t| t.to_descriptor()),
        Ok(TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii)
    )
}

/// Rows `[a, b)` of a variable-length text dataset.
fn read_text_slice(ds: &Dataset, a: usize, b: usize) -> Result<Vec<String>> {
    let sel: hdf5_metno::Selection = (a..b).into();
    match ds.dtype().map_err(hdf5)?.to_descriptor().map_err(hdf5)? {
        TypeDescriptor::VarLenUnicode => Ok(ds
            .read_slice_1d::<VarLenUnicode, _>(sel)
            .map_err(hdf5)?
            .iter()
            .map(|s| s.as_str().to_owned())
            .collect()),
        TypeDescriptor::VarLenAscii => Ok(ds
            .read_slice_1d::<VarLenAscii, _>(sel)
            .map_err(hdf5)?
            .iter()
            .map(|s| s.as_str().to_owned())
            .collect()),
        other => Err(IoError::InvalidLayout(format!(
            "{}: text of type {other:?} is not read",
            ds.name()
        ))),
    }
}

/// A whole text dataset (categories: short).
fn read_text(ds: &Dataset) -> Result<Vec<String>> {
    read_text_slice(ds, 0, ds.size())
}

/// A text dataset as codes into its distinct values, a slice at a time;
/// rows where `mask` is true are missing (-1).
fn read_text_codes(ds: &Dataset, mask: Option<&[bool]>) -> Result<(Vec<i32>, Vec<String>)> {
    let n = ds.size();
    let mut codes = Vec::with_capacity(n);
    let mut uniques: Vec<String> = Vec::new();
    let mut seen: HashMap<String, i32> = HashMap::new();
    let mut a = 0;
    while a < n {
        let b = (a + TEXT_SLICE).min(n);
        for (k, s) in read_text_slice(ds, a, b)?.into_iter().enumerate() {
            if mask.is_some_and(|m| m[a + k]) {
                codes.push(-1);
                continue;
            }
            let code = if let Some(&c) = seen.get(&s) {
                c
            } else {
                let c = i32::try_from(uniques.len()).map_err(|_| {
                    IoError::InvalidLayout(format!("{}: too many distinct values", ds.name()))
                })?;
                seen.insert(s.clone(), c);
                uniques.push(s);
                c
            };
            codes.push(code);
        }
        a = b;
    }
    Ok((codes, uniques))
}

fn read_mask(g: &Group) -> Result<Vec<bool>> {
    match read_numbers(&g.dataset("mask").map_err(hdf5)?)? {
        Numbers::Bool(m) => Ok(m),
        Numbers::U8(m) => Ok(m.into_iter().map(|x| x != 0).collect()),
        Numbers::I8(m) => Ok(m.into_iter().map(|x| x != 0).collect()),
        _ => Err(IoError::InvalidLayout(format!(
            "{}: mask is not boolean",
            g.name()
        ))),
    }
}

fn encoding(loc: &Location) -> Option<String> {
    let enc = loc.attr("encoding-type").ok()?;
    enc.read_scalar::<VarLenUnicode>()
        .map(|s| s.as_str().to_owned())
        .ok()
}

fn read_categorical(g: &Group) -> Result<Column> {
    let codes = read_numbers(&g.dataset("codes").map_err(hdf5)?)?;
    let cats = g.dataset("categories").map_err(hdf5)?;
    let categories = if is_text(&cats) {
        Categories::Text(read_text(&cats)?)
    } else {
        Categories::Numbers(read_numbers(&cats)?)
    };
    let ordered = g
        .attr("ordered")
        .ok()
        .and_then(|a| a.read_scalar::<bool>().ok())
        .unwrap_or(false);
    Ok(Column::Categorical {
        codes,
        categories,
        ordered,
    })
}

/// Read one column, `Ok(None)` when its encoding is not one read here.
fn read_column(frame: &Group, name: &str) -> Result<std::result::Result<Column, String>> {
    if let Ok(ds) = frame.dataset(name) {
        // Plain arrays; files from anndata before 0.7 carry no encoding.
        let enc = encoding(&ds).unwrap_or_default();
        return Ok(match enc.as_str() {
            "string-array" => {
                let (codes, uniques) = read_text_codes(&ds, None)?;
                Ok(Column::Text { codes, uniques })
            }
            "array" | "" if is_text(&ds) => {
                let (codes, uniques) = read_text_codes(&ds, None)?;
                Ok(Column::Text { codes, uniques })
            }
            "array" | "" => match read_numbers(&ds) {
                Ok(v) => Ok(Column::Numbers(v)),
                Err(_) => Err("array".to_owned()),
            },
            other => Err(other.to_owned()),
        });
    }
    let g = frame.group(name).map_err(hdf5)?;
    let enc = encoding(&g).unwrap_or_default();
    Ok(match enc.as_str() {
        "categorical" => Ok(read_categorical(&g)?),
        "nullable-integer" | "nullable-boolean" => {
            let values = read_numbers(&g.dataset("values").map_err(hdf5)?)?;
            Ok(Column::Nullable {
                values,
                mask: read_mask(&g)?,
            })
        }
        "nullable-string-array" => {
            let mask = read_mask(&g)?;
            let (codes, uniques) =
                read_text_codes(&g.dataset("values").map_err(hdf5)?, Some(&mask))?;
            Ok(Column::Text { codes, uniques })
        }
        other => Err(other.to_owned()),
    })
}

/// The column names of a dataframe group, in the file's order.
fn column_order(frame: &Group) -> Result<Vec<String>> {
    // An empty `column-order` is written as a float array, which reads as no
    // text: no columns.
    if let Ok(attr) = frame.attr("column-order") {
        if let Ok(v) = attr.read_raw::<VarLenUnicode>() {
            return Ok(v.iter().map(|s| s.as_str().to_owned()).collect());
        }
        return Ok(Vec::new());
    }
    // Before anndata 0.7 there is no order: every member but the index.
    let index = read_attr_string(frame, "_index")?.unwrap_or_default();
    let mut names = frame.member_names().map_err(hdf5)?;
    names.retain(|n| *n != index && !n.starts_with("__"));
    Ok(names)
}

/// Read the columns of the dataframe group `frame` (`obs` or `var`), or only
/// those named in `want`. A named column the frame lacks is an error, unless
/// `missing_ok` (several files read as one need not all have it).
///
/// # Errors
/// Returns [`IoError`] if a named column is absent, if a column's rows do not
/// match the others', or on a read failure.
pub fn read_frame(frame: &Group, want: Option<&[String]>, missing_ok: bool) -> Result<Frame> {
    let order = column_order(frame)?;
    let names: Vec<String> = match want {
        None => order,
        Some(w) => {
            if let Some(missing) = w.iter().find(|n| !order.contains(n)) {
                if !missing_ok {
                    return Err(IoError::InvalidLayout(format!(
                        "no column {missing:?} in {}; it has {order:?}",
                        frame.name()
                    )));
                }
            }
            w.iter().filter(|n| order.contains(n)).cloned().collect()
        }
    };
    let mut out = Frame::default();
    let mut rows: Option<usize> = None;
    for name in names {
        match read_column(frame, &name)? {
            Ok(col) => {
                let n = col.len();
                if *rows.get_or_insert(n) != n {
                    return Err(IoError::InvalidLayout(format!(
                        "{}/{name} has {n} rows, other columns {}",
                        frame.name(),
                        rows.unwrap_or(0)
                    )));
                }
                out.columns.push((name, col));
            }
            Err(enc) => out.skipped.push((name, enc)),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("polariseq_frame_{name}_{}.h5", std::process::id()))
    }

    fn text_attr(loc: &Location, name: &str, value: &str) {
        let v: VarLenUnicode = value.parse().unwrap();
        loc.new_attr::<VarLenUnicode>()
            .create(name)
            .unwrap()
            .write_scalar(&v)
            .unwrap();
    }

    fn text_ds(g: &Group, name: &str, values: &[&str]) -> Dataset {
        let v: Vec<VarLenUnicode> = values.iter().map(|s| s.parse().unwrap()).collect();
        let ds = g
            .new_dataset::<VarLenUnicode>()
            .shape(v.len())
            .create(name)
            .unwrap();
        ds.write_raw(&v).unwrap();
        ds
    }

    /// One column of each encoding, written the way anndata lays them out.
    #[test]
    fn every_encoding_is_read_in_its_own_width() {
        let path = tmp("encodings");
        {
            let f = hdf5_metno::File::create(&path).unwrap();
            let obs = f.create_group("obs").unwrap();
            text_attr(&obs, "_index", "_index");
            text_ds(&obs, "_index", &["a", "b", "c", "d"]);
            let order: Vec<VarLenUnicode> = ["cat", "txt", "num", "flag", "nint", "ntxt"]
                .iter()
                .map(|s| s.parse().unwrap())
                .collect();
            obs.new_attr::<VarLenUnicode>()
                .shape(order.len())
                .create("column-order")
                .unwrap()
                .write_raw(&order)
                .unwrap();

            let cat = obs.create_group("cat").unwrap();
            text_attr(&cat, "encoding-type", "categorical");
            let codes = cat.new_dataset::<i8>().shape(4).create("codes").unwrap();
            codes.write_raw(&[1_i8, 0, -1, 1]).unwrap();
            text_ds(&cat, "categories", &["ctrl", "stim"]);
            cat.new_attr::<bool>()
                .create("ordered")
                .unwrap()
                .write_scalar(&true)
                .unwrap();

            let txt = text_ds(&obs, "txt", &["x", "y", "x", "x"]);
            text_attr(&txt, "encoding-type", "string-array");

            let num = obs.new_dataset::<f32>().shape(4).create("num").unwrap();
            num.write_raw(&[1.5_f32, 2.0, -3.0, 0.0]).unwrap();
            text_attr(&num, "encoding-type", "array");

            let flag = obs.new_dataset::<bool>().shape(4).create("flag").unwrap();
            flag.write_raw(&[true, false, false, true]).unwrap();
            text_attr(&flag, "encoding-type", "array");

            let nint = obs.create_group("nint").unwrap();
            text_attr(&nint, "encoding-type", "nullable-integer");
            let v = nint.new_dataset::<i64>().shape(4).create("values").unwrap();
            v.write_raw(&[7_i64, 0, 9, 10]).unwrap();
            let m = nint.new_dataset::<bool>().shape(4).create("mask").unwrap();
            m.write_raw(&[false, true, false, false]).unwrap();

            let ntxt = obs.create_group("ntxt").unwrap();
            text_attr(&ntxt, "encoding-type", "nullable-string-array");
            text_ds(&ntxt, "values", &["p", "", "q", "p"]);
            let m = ntxt.new_dataset::<bool>().shape(4).create("mask").unwrap();
            m.write_raw(&[false, true, false, false]).unwrap();
        }
        let f = hdf5_metno::File::open(&path).unwrap();
        let frame = read_frame(&f.group("obs").unwrap(), None, false).unwrap();
        let cols: HashMap<_, _> = frame.columns.into_iter().collect();
        assert_eq!(
            cols["cat"],
            Column::Categorical {
                codes: Numbers::I8(vec![1, 0, -1, 1]),
                categories: Categories::Text(vec!["ctrl".into(), "stim".into()]),
                ordered: true,
            }
        );
        assert_eq!(
            cols["txt"],
            Column::Text {
                codes: vec![0, 1, 0, 0],
                uniques: vec!["x".into(), "y".into()]
            }
        );
        assert_eq!(
            cols["num"],
            Column::Numbers(Numbers::F32(vec![1.5, 2.0, -3.0, 0.0]))
        );
        assert_eq!(
            cols["flag"],
            Column::Numbers(Numbers::Bool(vec![true, false, false, true]))
        );
        assert_eq!(
            cols["nint"],
            Column::Nullable {
                values: Numbers::I64(vec![7, 0, 9, 10]),
                mask: vec![false, true, false, false]
            }
        );
        assert_eq!(
            cols["ntxt"],
            Column::Text {
                codes: vec![0, -1, 1, 0],
                uniques: vec!["p".into(), "q".into()]
            }
        );
        assert_eq!(frame.skipped, Vec::<(String, String)>::new());

        let only = read_frame(&f.group("obs").unwrap(), Some(&["num".to_owned()]), false).unwrap();
        assert_eq!(only.columns.len(), 1);
        assert!(read_frame(&f.group("obs").unwrap(), Some(&["nope".to_owned()]), false).is_err());
        let some = read_frame(
            &f.group("obs").unwrap(),
            Some(&["nope".to_owned(), "cat".to_owned()]),
            true,
        )
        .unwrap();
        assert_eq!(some.columns.len(), 1);
        let _ = std::fs::remove_file(&path);
    }

    /// Text longer than one slice keeps its codes across the slice edges.
    #[test]
    fn text_codes_span_slices() {
        let path = tmp("slices");
        let n = TEXT_SLICE + 5;
        {
            let f = hdf5_metno::File::create(&path).unwrap();
            let g = f.create_group("obs").unwrap();
            let names: Vec<String> = (0..n).map(|i| format!("d{}", i % 3)).collect();
            let refs: Vec<&str> = names.iter().map(String::as_str).collect();
            text_ds(&g, "donor", &refs);
        }
        let f = hdf5_metno::File::open(&path).unwrap();
        let ds = f.group("obs").unwrap().dataset("donor").unwrap();
        let (codes, uniques) = read_text_codes(&ds, None).unwrap();
        assert_eq!(uniques, vec!["d0", "d1", "d2"]);
        assert!(codes.iter().enumerate().all(|(i, &c)| c == (i % 3) as i32));
        let _ = std::fs::remove_file(&path);
    }

    /// The fixture anndata wrote: a categorical and a float column.
    #[test]
    fn reads_the_fixture_written_by_anndata() {
        let p = format!(
            "{}/../../tests/data/sparse_100x200.h5ad",
            env!("CARGO_MANIFEST_DIR")
        );
        let f = hdf5_metno::File::open(p).unwrap();
        let frame = read_frame(&f.group("obs").unwrap(), None, false).unwrap();
        let names: Vec<&str> = frame.columns.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["batch", "n_counts"]);
        assert!(frame.columns.iter().all(|(_, c)| c.len() == 100));
        assert!(matches!(frame.columns[0].1, Column::Categorical { .. }));
    }
}
