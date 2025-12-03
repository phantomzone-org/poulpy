use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(not(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
)))]
fn bench_ifft_avx2_fma(_c: &mut Criterion) {
    eprintln!("Skipping: AVX IFft benchmark requires x86_64 + AVX2 + FMA");
}

#[cfg(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
pub fn bench_ifft_avx2_fma(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use poulpy_cpu_avx::ReimIFFTAvx;
    use poulpy_hal::reference::fft64::reim::{ReimDFTExecute, ReimIFFTTable};
    use std::hint::black_box;

    let group_name: String = "ifft_avx2_fma".to_string();

    let mut group = c.benchmark_group(group_name);

    if std::is_x86_feature_detected!("avx2") {
        fn runner(m: usize) -> impl FnMut() {
            let mut values: Vec<f64> = vec![0f64; m << 1];

            let scale = 1.0f64 / (2 * m) as f64;
            values.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as f64 * scale);

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

#[cfg(not(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
)))]
fn bench_fft_avx2_fma(_c: &mut Criterion) {
    eprintln!("Skipping: AVX FFT benchmark requires x86_64 + AVX2 + FMA");
}

#[cfg(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
pub fn bench_fft_avx2_fma(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use poulpy_cpu_avx::ReimFFTAvx;
    use poulpy_hal::reference::fft64::reim::{ReimDFTExecute, ReimFFTTable};
    use std::hint::black_box;

    let group_name: String = "fft_avx2_fma".to_string();

    let mut group = c.benchmark_group(group_name);

    if std::is_x86_feature_detected!("avx2") {
        fn runner(m: usize) -> impl FnMut() {
            let mut values: Vec<f64> = vec![0f64; m << 1];

            let scale = 1.0f64 / (2 * m) as f64;
            values.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as f64 * scale);

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
