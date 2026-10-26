//! Phase 0 smoke benchmark.
//!
//! This is *not* a real PCA benchmark — it exists to prove the `criterion`
//! harness is wired up and to produce a baseline number for `double_in_place`.
//! Phase 2 replaces this with a randomized-SVD-through-`faer` PCA benchmark on
//! the real 10x 1.3M mouse-neuron dataset.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use polariseq_core::double_in_place;

fn bench_double_in_place(c: &mut Criterion) {
    let mut group = c.benchmark_group("double_in_place");
    for &n in &[1_000usize, 100_000, 1_000_000] {
        group.bench_function(n.to_string(), |b| {
            b.iter_batched(
                || vec![1.0_f64; n],
                |mut v| {
                    double_in_place(black_box(&mut v));
                    v
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_double_in_place);
criterion_main!(benches);
