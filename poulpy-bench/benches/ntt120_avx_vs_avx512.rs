//! Three-way comparison of NTT120 backends:
//!  - `NTT120Avx`     — AVX2/256-bit, 4 prime lanes per coefficient (Primes30, 4×~30-bit).
//!  - `NTT120Avx512`  — AVX-512F/512-bit, 2 coefficients pair-packed (Primes30, 4×~30-bit).
//!  - `NTT120Ifma`    — AVX-512F + AVX-512-IFMA + AVX-512VL (Primes40, 3×~40-bit).
//!
//! All three produce mathematically equivalent NTTs (each verified against `NTT120Ref`
//! via `cross_backend_test_suite!`); this bench measures wall-clock timing per backend.
//! The Avx vs Avx512 pair is apples-to-apples (same prime set, same buffer layout); the
//! IFMA backend is included as a reference for AVX-512 with IFMA available.

use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(not(all(
    feature = "enable-avx",
    feature = "enable-avx512f",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma",
    target_feature = "avx512f"
)))]
fn bench_ntt120_avx_vs_avx512(_c: &mut Criterion) {
    eprintln!("Skipping: requires enable-avx + enable-avx512f, AVX2+FMA+AVX512F at compile time");
}

#[cfg(all(
    feature = "enable-avx",
    feature = "enable-avx512f",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma",
    target_feature = "avx512f"
))]
fn bench_ntt120_avx_vs_avx512(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use poulpy_cpu_avx::NTT120Avx;
    use poulpy_cpu_avx512::NTT120Avx512;
    use poulpy_cpu_ref::reference::ntt120::{
        I128BigOps, NttAdd, NttDFTExecute, NttNegate, NttSub, NttTable, NttTableInv, Primes30,
    };
    use std::hint::black_box;

    #[cfg(all(feature = "enable-ifma", target_feature = "avx512ifma", target_feature = "avx512vl"))]
    use poulpy_cpu_avx512::NTT120Ifma;
    #[cfg(all(feature = "enable-ifma", target_feature = "avx512ifma", target_feature = "avx512vl"))]
    use poulpy_cpu_ref::reference::ntt_ifma::{
        NttIfmaAdd, NttIfmaDFTExecute, NttIfmaNegate, NttIfmaSub, ntt::NttIfmaTable, ntt::NttIfmaTableInv, primes::Primes40,
    };

    for log_n in [10usize, 12, 14] {
        let n = 1usize << log_n;
        let q120b_len = 4 * n;

        // ── Lazy ops: ntt_add / ntt_sub / ntt_negate ──
        {
            let a: Vec<u64> = (0..q120b_len).map(|i| i as u64 + 1).collect();
            let b: Vec<u64> = (0..q120b_len).map(|i| (i + 7) as u64).collect();
            let mut res = vec![0u64; q120b_len];

            let mut group = c.benchmark_group("ntt_add");
            group.bench_with_input(BenchmarkId::new("avx2_256", format!("n={}", n)), &(), |bcr, _| {
                bcr.iter(|| {
                    NTT120Avx::ntt_add(&mut res, &a, &b);
                    black_box(());
                });
            });
            group.bench_with_input(BenchmarkId::new("avx512_pair", format!("n={}", n)), &(), |bcr, _| {
                bcr.iter(|| {
                    NTT120Avx512::ntt_add(&mut res, &a, &b);
                    black_box(());
                });
            });
            #[cfg(all(feature = "enable-ifma", target_feature = "avx512ifma", target_feature = "avx512vl"))]
            group.bench_with_input(BenchmarkId::new("ifma_avx512", format!("n={}", n)), &(), |bcr, _| {
                bcr.iter(|| {
                    NTT120Ifma::ntt_ifma_add(&mut res, &a, &b);
                    black_box(());
                });
            });
            group.finish();

            let mut group = c.benchmark_group("ntt_sub");
            group.bench_with_input(BenchmarkId::new("avx2_256", format!("n={}", n)), &(), |bcr, _| {
                bcr.iter(|| {
                    NTT120Avx::ntt_sub(&mut res, &a, &b);
                    black_box(());
                });
            });
            group.bench_with_input(BenchmarkId::new("avx512_pair", format!("n={}", n)), &(), |bcr, _| {
                bcr.iter(|| {
                    NTT120Avx512::ntt_sub(&mut res, &a, &b);
                    black_box(());
                });
            });
            #[cfg(all(feature = "enable-ifma", target_feature = "avx512ifma", target_feature = "avx512vl"))]
            group.bench_with_input(BenchmarkId::new("ifma_avx512", format!("n={}", n)), &(), |bcr, _| {
                bcr.iter(|| {
                    NTT120Ifma::ntt_ifma_sub(&mut res, &a, &b);
                    black_box(());
                });
            });
            group.finish();

            let mut group = c.benchmark_group("ntt_negate");
            group.bench_with_input(BenchmarkId::new("avx2_256", format!("n={}", n)), &(), |bcr, _| {
                bcr.iter(|| {
                    NTT120Avx::ntt_negate(&mut res, &a);
                    black_box(());
                });
            });
            group.bench_with_input(BenchmarkId::new("avx512_pair", format!("n={}", n)), &(), |bcr, _| {
                bcr.iter(|| {
                    NTT120Avx512::ntt_negate(&mut res, &a);
                    black_box(());
                });
            });
            #[cfg(all(feature = "enable-ifma", target_feature = "avx512ifma", target_feature = "avx512vl"))]
            group.bench_with_input(BenchmarkId::new("ifma_avx512", format!("n={}", n)), &(), |bcr, _| {
                bcr.iter(|| {
                    NTT120Ifma::ntt_ifma_negate(&mut res, &a);
                    black_box(());
                });
            });
            group.finish();
        }

        // ── Forward NTT (full transform) ──
        {
            let table_fwd = NttTable::<Primes30>::new(n);
            let init: Vec<u64> = (0..q120b_len).map(|i| i as u64 + 1).collect();
            let mut data_avx = init.clone();
            let mut data_avx512 = init.clone();

            let mut group = c.benchmark_group("ntt_forward");
            group.bench_with_input(BenchmarkId::new("avx2_256", format!("n={}", n)), &(), |bcr, _| {
                bcr.iter(|| {
                    NTT120Avx::ntt_dft_execute(&table_fwd, &mut data_avx);
                    black_box(());
                });
            });
            group.bench_with_input(BenchmarkId::new("avx512_pair", format!("n={}", n)), &(), |bcr, _| {
                bcr.iter(|| {
                    NTT120Avx512::ntt_dft_execute(&table_fwd, &mut data_avx512);
                    black_box(());
                });
            });
            #[cfg(all(feature = "enable-ifma", target_feature = "avx512ifma", target_feature = "avx512vl"))]
            {
                let table_ifma = NttIfmaTable::<Primes40>::new(n);
                let mut data_ifma: Vec<u64> = (0..q120b_len).map(|i| i as u64 + 1).collect();
                group.bench_with_input(BenchmarkId::new("ifma_avx512", format!("n={}", n)), &(), |bcr, _| {
                    bcr.iter(|| {
                        NTT120Ifma::ntt_ifma_dft_execute(&table_ifma, &mut data_ifma);
                        black_box(());
                    });
                });
            }
            group.finish();
        }

        // ── Inverse NTT (full transform) ──
        {
            let table_inv = NttTableInv::<Primes30>::new(n);
            let init: Vec<u64> = (0..q120b_len).map(|i| i as u64 + 1).collect();
            let mut data_avx = init.clone();
            let mut data_avx512 = init.clone();

            let mut group = c.benchmark_group("intt_inverse");
            group.bench_with_input(BenchmarkId::new("avx2_256", format!("n={}", n)), &(), |bcr, _| {
                bcr.iter(|| {
                    NTT120Avx::ntt_dft_execute(&table_inv, &mut data_avx);
                    black_box(());
                });
            });
            group.bench_with_input(BenchmarkId::new("avx512_pair", format!("n={}", n)), &(), |bcr, _| {
                bcr.iter(|| {
                    NTT120Avx512::ntt_dft_execute(&table_inv, &mut data_avx512);
                    black_box(());
                });
            });
            #[cfg(all(feature = "enable-ifma", target_feature = "avx512ifma", target_feature = "avx512vl"))]
            {
                let table_inv_ifma = NttIfmaTableInv::<Primes40>::new(n);
                let mut data_ifma: Vec<u64> = (0..q120b_len).map(|i| i as u64 + 1).collect();
                group.bench_with_input(BenchmarkId::new("ifma_avx512", format!("n={}", n)), &(), |bcr, _| {
                    bcr.iter(|| {
                        NTT120Ifma::ntt_ifma_dft_execute(&table_inv_ifma, &mut data_ifma);
                        black_box(());
                    });
                });
            }
            group.finish();
        }

        // ── i128 vector ops (vec_znx_big paths, shared across all three backends) ──
        {
            let a: Vec<i128> = (0..n).map(|i| ((i as i128) << 100) - (1i128 << 80)).collect();
            let b: Vec<i128> = (0..n).map(|i| ((i as i128) << 90) + 7).collect();
            let mut res = vec![0i128; n];

            let mut group = c.benchmark_group("i128_add");
            group.bench_with_input(BenchmarkId::new("avx2_2per_reg", format!("n={}", n)), &(), |bcr, _| {
                bcr.iter(|| {
                    NTT120Avx::i128_add(&mut res, &a, &b);
                    black_box(());
                });
            });
            group.bench_with_input(BenchmarkId::new("avx512_4per_reg", format!("n={}", n)), &(), |bcr, _| {
                bcr.iter(|| {
                    NTT120Avx512::i128_add(&mut res, &a, &b);
                    black_box(());
                });
            });
            #[cfg(all(feature = "enable-ifma", target_feature = "avx512ifma", target_feature = "avx512vl"))]
            group.bench_with_input(BenchmarkId::new("ifma_avx512", format!("n={}", n)), &(), |bcr, _| {
                bcr.iter(|| {
                    NTT120Ifma::i128_add(&mut res, &a, &b);
                    black_box(());
                });
            });
            group.finish();
        }
    }
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_ntt120_avx_vs_avx512
}
criterion_main!(benches);
