use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use poulpy_hal::reference::fft64::reim::{ReimDFTExecute, ReimFFTRef, ReimFFTTable, ReimIFFTRef, ReimIFFTTable};

pub fn bench_fft_ref(c: &mut Criterion) {
    let group_name: String = "fft_ref".to_string();

    let mut group = c.benchmark_group(group_name);

    fn runner(m: usize) -> impl FnMut() {
        let mut values: Vec<f64> = vec![0f64; m << 1];
        let scale: f64 = 1.0f64 / (2 * m) as f64;
        values
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = (i + 1) as f64 * scale);
        let table: ReimFFTTable<f64> = ReimFFTTable::<f64>::new(m);
        move || {
            ReimFFTRef::reim_dft_execute(&table, &mut values);
            black_box(());
        }
    }

    for log_m in [9, 10, 11, 12, 13, 14, 15] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 2 << log_m));
        let mut runner = runner(1 << log_m);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_ifft_ref(c: &mut Criterion) {
    let group_name: String = "ifft_ref".to_string();

    let mut group = c.benchmark_group(group_name);

    fn runner(m: usize) -> impl FnMut() {
        let mut values: Vec<f64> = vec![0f64; m << 1];
        let scale: f64 = 1.0f64 / (2 * m) as f64;
        values
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = (i + 1) as f64 * scale);
        let table: ReimIFFTTable<f64> = ReimIFFTTable::<f64>::new(m);
        move || {
            ReimIFFTRef::reim_dft_execute(&table, &mut values);
            black_box(());
        }
    }

    for log_m in [9, 10, 11, 12, 13, 14, 15] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 2 << log_m));
        let mut runner = runner(1 << log_m);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

criterion_group!(benches, bench_fft_ref, bench_ifft_ref,);

criterion_main!(benches);
