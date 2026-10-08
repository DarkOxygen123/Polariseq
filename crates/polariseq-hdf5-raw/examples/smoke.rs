//! Manual smoke: chunk FFI against a real file.
//! Usage: cargo run -p polariseq-hdf5-raw --example smoke -- <file.h5ad> [dataset]

use hdf5_metno::File;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: smoke <file> [dataset]");
    let name = std::env::args().nth(2).unwrap_or_else(|| "X/data".into());
    let file = File::open(&path).expect("open");
    let ds = file.dataset(&name).expect("dataset");

    let filters = polariseq_hdf5_raw::filters(&ds);
    println!("filters: {:?}", filters);

    let n = polariseq_hdf5_raw::num_chunks(&ds).expect("num_chunks");
    println!("num_chunks: {n}");
    match polariseq_hdf5_raw::chunk_meta_by_coord(&ds, 0) {
        Some(m) => println!("by-coord(0): {m:?}"),
        None => println!("by-coord(0): FAILED"),
    }

    let m0 = polariseq_hdf5_raw::chunk_meta(&ds, 0).expect("chunk 0");
    println!("chunk 0: {m0:?}");

    let mut buf = vec![0_u8; m0.nbytes as usize];
    assert!(polariseq_hdf5_raw::read_chunk(
        &ds,
        m0.elem_offset,
        &mut buf
    ));
    println!("read {} raw bytes; first 8: {:02x?}", buf.len(), &buf[..8]);

    if filters
        .iter()
        .any(|f| f.id == polariseq_hdf5_raw::filter_ids::DEFLATE)
    {
        use std::io::Read;
        let mut out = Vec::new();
        flate2::read::ZlibDecoder::new(&buf[..])
            .read_to_end(&mut out)
            .expect("inflate");
        println!("inflated to {} bytes; as f32 LE:", out.len());
        let v = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
        println!("  first value: {v}");
    }
}
