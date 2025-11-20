#![cfg(target_arch = "x86_64")]
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use poulpy_cpu_avx::{ReimFFTAvx, ReimIFFTAvx};
use poulpy_hal::reference::fft64::reim::{ReimDFTExecute, ReimFFTTable, ReimIFFTTable};

pub fn bench_ifft_avx2_fma(c: &mut Criterion) {
    let group_name: String = "ifft_avx2_fma".to_string();

    let mut group = c.benchmark_group(group_name);

    if std::is_x86_feature_detected!("avx2") {
        fn runner(m: usize) -> impl FnMut() {
            let mut values: Vec<f64> = vec![0f64; m << 1];

            let scale = 1.0f64 / (2 * m) as f64;
            values
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x = (i + 1) as f64 * scale);

            let table: ReimIFFTTable<f64> = ReimIFFTTable::<f64>::new(m);
            move || {
                ReimIFFTAvx::reim_dft_execute(&table, &mut values);
                black_box(());
            }
        }

        for log_m in [9, 10, 11, 12, 13, 14, 15] {
            let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 2 << log_m));

            let mut runner = runner(1 << log_m);
            group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
        }
    } else {
        eprintln!("skipping: CPU lacks avx2");
        return;
    }

    group.finish();
}

pub fn bench_fft_avx2_fma(c: &mut Criterion) {
    let group_name: String = "fft_avx2_fma".to_string();

    let mut group = c.benchmark_group(group_name);

    if std::is_x86_feature_detected!("avx2") {
        fn runner(m: usize) -> impl FnMut() {
            let mut values: Vec<f64> = vec![0f64; m << 1];

            let scale = 1.0f64 / (2 * m) as f64;
            values
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x = (i + 1) as f64 * scale);

            let table: ReimFFTTable<f64> = ReimFFTTable::<f64>::new(m);
            move || {
                ReimFFTAvx::reim_dft_execute(&table, &mut values);
                black_box(());
            }
        }

        for log_m in [9, 10, 11, 12, 13, 14, 15] {
            let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 2 << log_m));

            let mut runner = runner(1 << log_m);
            group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
        }
    } else {
        eprintln!("skipping: CPU lacks avx2");
        return;
    }

    group.finish();
}

criterion_group!(benches_x86, bench_fft_avx2_fma, bench_ifft_avx2_fma,);
criterion_main!(benches_x86);
