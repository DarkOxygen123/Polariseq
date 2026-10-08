//! Vectorisable `f32` kernels shared by PCA, kNN and UMAP.
//!
//! Rust (without `-ffast-math`) will not reassociate a floating-point
//! reduction, so a plain `s += d * d` loop compiles to a serial dependency
//! chain — one lane of the vector unit. Splitting the accumulation into
//! eight independent partial sums lets LLVM vectorise it and hides the FMA
//! latency; the partials are combined in a fixed order, so results are
//! deterministic run to run.
//!
//! These are plain safe Rust: no intrinsics, no nightly, portable to every
//! target the wheels are built for.

const LANES: usize = 8;

/// Dot product `Σ a[i]·b[i]`.
///
/// # Panics
/// Panics if the slices differ in length.
#[inline]
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    let mut acc = [0.0_f32; LANES];
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for l in 0..LANES {
            acc[l] += x[l] * y[l];
        }
    }
    let mut tail = 0.0_f32;
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        tail += x * y;
    }
    reduce(&acc) + tail
}

/// Squared Euclidean distance `Σ (a[i] − b[i])²`.
///
/// # Panics
/// Panics if the slices differ in length.
#[inline]
#[must_use]
pub fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "sq_dist: length mismatch");
    let mut acc = [0.0_f32; LANES];
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for l in 0..LANES {
            let d = x[l] - y[l];
            acc[l] += d * d;
        }
    }
    let mut tail = 0.0_f32;
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        let d = x - y;
        tail += d * d;
    }
    reduce(&acc) + tail
}

/// `y[i] += alpha · x[i]` for every `i`.
///
/// # Panics
/// Panics if the slices differ in length.
#[inline]
pub fn axpy(y: &mut [f32], alpha: f32, x: &[f32]) {
    assert_eq!(y.len(), x.len(), "axpy: length mismatch");
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}

/// Fixed-order horizontal sum: pairwise so the order is independent of the
/// input length.
#[inline]
fn reduce(acc: &[f32; LANES]) -> f32 {
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum()
    }

    fn vecs(n: usize) -> (Vec<f32>, Vec<f32>) {
        let a: Vec<f32> = (0..n).map(|i| ((i * 7) % 13) as f32 * 0.25 - 1.0).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i * 5) % 11) as f32 * 0.5 - 2.0).collect();
        (a, b)
    }

    #[test]
    fn dot_and_sq_dist_match_naive_for_all_tail_lengths() {
        for n in [0, 1, 7, 8, 9, 15, 16, 17, 50, 64, 101] {
            let (a, b) = vecs(n);
            let d = naive_dot(&a, &b);
            assert!(
                (f64::from(dot(&a, &b)) - d).abs() < 1e-3 * d.abs().max(1.0),
                "dot n={n}"
            );
            let sq: f64 = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
                .sum();
            assert!(
                (f64::from(sq_dist(&a, &b)) - sq).abs() < 1e-3 * sq.max(1.0),
                "sq n={n}"
            );
        }
    }

    #[test]
    fn axpy_accumulates() {
        let (mut y, x) = vecs(21);
        let y0 = y.clone();
        axpy(&mut y, 0.5, &x);
        for i in 0..21 {
            assert!((y[i] - (y0[i] + 0.5 * x[i])).abs() < 1e-6);
        }
    }

    #[test]
    fn results_are_bit_identical_across_calls() {
        let (a, b) = vecs(1000);
        assert_eq!(dot(&a, &b).to_bits(), dot(&a, &b).to_bits());
        assert_eq!(sq_dist(&a, &b).to_bits(), sq_dist(&a, &b).to_bits());
    }
}

#[cfg(test)]
mod throughput {
    //! Run with: `cargo test --release -p polariseq-core --lib -- simd::throughput --ignored --nocapture`
    use super::*;
    use rayon::prelude::*;

    fn naive(a: &[f32], b: &[f32]) -> f32 {
        let mut s = 0.0_f32;
        for (x, y) in a.iter().zip(b) {
            let d = x - y;
            s += d * d;
        }
        s
    }

    #[test]
    #[ignore = "manual microbenchmark; run with --ignored --nocapture"]
    fn sq_dist_gflops() {
        let (n, d) = (20_000usize, 50usize);
        let data: Vec<f32> = (0..n * d)
            .map(|i| ((i * 7919) % 1000) as f32 * 0.01)
            .collect();
        for (name, f) in [
            ("simd::sq_dist", sq_dist as fn(&[f32], &[f32]) -> f32),
            ("naive", naive),
        ] {
            let t = std::time::Instant::now();
            let total: f64 = (0..n)
                .into_par_iter()
                .map(|i| {
                    let a = &data[i * d..(i + 1) * d];
                    let mut acc = 0.0_f32;
                    for j in 0..n {
                        acc += f(a, &data[j * d..(j + 1) * d]);
                    }
                    f64::from(acc)
                })
                .sum();
            let secs = t.elapsed().as_secs_f64();
            let gflops = (n as f64 * n as f64 * d as f64 * 3.0) / secs / 1e9;
            println!("{name}: {secs:.2}s  {gflops:.1} GFLOP/s  (checksum {total:e})");
        }
    }
}
