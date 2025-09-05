use std::{ffi::c_void, hint::black_box};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use poulpy_backend::cpu_spqlios::reim;
use poulpy_hal::reference::reim::{TableFFT, TableIFFT};

pub fn bench_fft_ref(c: &mut Criterion) {
    let group_name: String = format!("fft_ref");

    let mut group = c.benchmark_group(group_name);

    fn runner(m: usize) -> impl FnMut() {
        let mut values: Vec<f64> = vec![0f64; m << 1];
        let scale: f64 = f64::from(1.0f64 / (2 * m) as f64);
        values
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = (i + 1) as f64 * scale);
        let table: TableFFT<f64> = TableFFT::<f64>::new(m as usize);
        move || {
            table.execute(&mut values);
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

pub fn bench_fft_avx2_fma(c: &mut Criterion) {
    let group_name: String = format!("fft_avx2_fma");

    let mut group = c.benchmark_group(group_name);

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "avx2,fma")]
    fn runner(m: usize) -> impl FnMut() {
        let mut values: Vec<f64> = vec![0f64; m << 1];

        let scale = f64::from(1.0f64 / (2 * m) as f64);
        values
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = (i + 1) as f64 * scale);

        let table: TableFFT<f64> = TableFFT::<f64>::new(m as usize);
        move || {
            table.execute_avx2_fma(&mut values);
            black_box(());
        }
    }

    if std::is_x86_feature_detected!("avx2") {
        for log_m in [9, 10, 11, 12, 13, 14, 15] {
            let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 2 << log_m));
            unsafe {
                let mut runner = runner(1 << log_m);
                group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
            }
        }
    } else {
        eprintln!("skipping: CPU lacks avx2");
        return;
    }

    group.finish();
}

pub fn bench_fft_spqlios(c: &mut Criterion) {
    let group_name: String = format!("fft_spqlios");

    let mut group = c.benchmark_group(group_name);

    fn runner(m: usize) -> impl FnMut() {
        let mut values: Vec<f64> = vec![0f64; m << 1];

        let scale = f64::from(1.0f64 / (2 * m) as f64);
        values
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = (i + 1) as f64 * scale);

        unsafe {
            reim::reim_fft_simple(m as u32, values.as_mut_ptr() as *mut c_void);
        }

        move || {
            unsafe {
                reim::reim_fft_simple(m as u32, values.as_mut_ptr() as *mut c_void);
            }
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
    let group_name: String = format!("ifft_ref");

    let mut group = c.benchmark_group(group_name);

    fn runner(m: usize) -> impl FnMut() {
        let mut values: Vec<f64> = vec![0f64; m << 1];
        let scale: f64 = f64::from(1.0f64 / (2 * m) as f64);
        values
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = (i + 1) as f64 * scale);
        let table: TableIFFT<f64> = TableIFFT::<f64>::new(m as usize);
        move || {
            table.execute(&mut values);
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

pub fn bench_ifft_avx2_fma(c: &mut Criterion) {
    let group_name: String = format!("ifft_avx2_fma");

    let mut group = c.benchmark_group(group_name);

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "avx2,fma")]
    fn runner(m: usize) -> impl FnMut() {
        let mut values: Vec<f64> = vec![0f64; m << 1];

        let scale = f64::from(1.0f64 / (2 * m) as f64);
        values
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = (i + 1) as f64 * scale);

        let table: TableIFFT<f64> = TableIFFT::<f64>::new(m as usize);
        move || {
            table.execute_avx2_fma(&mut values);
            black_box(());
        }
    }

    if std::is_x86_feature_detected!("avx2") {
        for log_m in [9, 10, 11, 12, 13, 14, 15] {
            let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 2 << log_m));
            unsafe {
                let mut runner = runner(1 << log_m);
                group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
            }
        }
    } else {
        eprintln!("skipping: CPU lacks avx2");
        return;
    }

    group.finish();
}

pub fn bench_ifft_spqlios(c: &mut Criterion) {
    let group_name: String = format!("ifft_spqlios");

    let mut group = c.benchmark_group(group_name);

    fn runner(m: usize) -> impl FnMut() {
        let mut values: Vec<f64> = vec![0f64; m << 1];

        let scale = f64::from(1.0f64 / (2 * m) as f64);
        values
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = (i + 1) as f64 * scale);

        unsafe {
            reim::reim_ifft_simple(m as u32, values.as_mut_ptr() as *mut c_void);
        }

        move || {
            unsafe {
                reim::reim_ifft_simple(m as u32, values.as_mut_ptr() as *mut c_void);
            }
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

criterion_group!(
    benches,
    bench_fft_ref,
    bench_fft_avx2_fma,
    bench_fft_spqlios,
    bench_ifft_ref,
    bench_ifft_avx2_fma,
    bench_ifft_spqlios
);
criterion_main!(benches);
