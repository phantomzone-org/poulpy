use criterion::{Criterion, criterion_group, criterion_main};





pub fn bench_intt_ref(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use poulpy_cpu_ref::NTT120Ref;
    use poulpy_hal::reference::ntt120::{NttDFTExecute, NttTableInv, Primes30};
    use std::hint::black_box;

    let group_name: String = "intt_ref".to_string();

    let mut group = c.benchmark_group(group_name);

    fn runner(n: usize) -> impl FnMut() {
        let mut values: Vec<u64> = vec![0u64; 4*n];

        
        values.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as u64);

        let table: NttTableInv<Primes30> = NttTableInv::<Primes30>::new(n);
        move || {
            NTT120Ref::ntt_dft_execute(&table, &mut values);
            black_box(());
        }
    }

    for log_n in [10, 11, 12, 13, 14, 15, 16] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 1 << log_n));

        let mut runner = runner(1 << log_n);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}


pub fn bench_ntt_ref(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use poulpy_cpu_ref::NTT120Ref;
    use poulpy_hal::reference::ntt120::{NttDFTExecute, NttTable, Primes30};
    use std::hint::black_box;

    let group_name: String = "ntt_ref".to_string();

    let mut group = c.benchmark_group(group_name);

    fn runner(n: usize) -> impl FnMut() {
        let mut values: Vec<u64> = vec![0u64; 4*n];

        
        values.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as u64);

        let table: NttTable<Primes30> = NttTable::<Primes30>::new(n);
        move || {
            NTT120Ref::ntt_dft_execute(&table, &mut values);
            black_box(());
        }
    }

    for log_n in [10, 11, 12, 13, 14, 15, 16] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 1 << log_n));

        let mut runner = runner(1 << log_n);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

criterion_group!(benches_x86, bench_ntt_ref, bench_intt_ref,);
criterion_main!(benches_x86);
