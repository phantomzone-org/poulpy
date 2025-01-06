use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use math::dft::DFT;
use math::{dft::ntt::Table, modulus::prime::Prime};

fn forward_inplace(c: &mut Criterion) {
    fn runner(prime_instance: Prime<u64>, nth_root: u64) -> Box<dyn FnMut()> {
        let ntt_table: Table<u64> = Table::<u64>::new(prime_instance, nth_root);
        let mut a: Vec<u64> = vec![0; (nth_root >> 1) as usize];
        for i in 0..a.len() {
            a[i] = i as u64;
        }
        Box::new(move || ntt_table.forward_inplace::<false>(&mut a))
    }

    let mut b: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("forward_inplace");
    for log_nth_root in 11..18 {
        let prime_instance: Prime<u64> = Prime::<u64>::new(0x1fffffffffe00001, 1);

        let runners = [("prime", { runner(prime_instance, 1 << log_nth_root) })];
        for (name, mut runner) in runners {
            let id = BenchmarkId::new(name, 1 << (log_nth_root - 1));
            b.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
        }
    }
}

fn forward_inplace_lazy(c: &mut Criterion) {
    fn runner(prime_instance: Prime<u64>, nth_root: u64) -> Box<dyn FnMut()> {
        let ntt_table: Table<u64> = Table::<u64>::new(prime_instance, nth_root);
        let mut a: Vec<u64> = vec![0; (nth_root >> 1) as usize];
        for i in 0..a.len() {
            a[i] = i as u64;
        }
        Box::new(move || ntt_table.forward_inplace_lazy(&mut a))
    }

    let mut b: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("forward_inplace_lazy");
    for log_nth_root in 11..17 {
        let prime_instance: Prime<u64> = Prime::<u64>::new(0x1fffffffffe00001, 1);

        let runners = [("prime", { runner(prime_instance, 1 << log_nth_root) })];
        for (name, mut runner) in runners {
            let id = BenchmarkId::new(name, 1 << (log_nth_root - 1));
            b.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
        }
    }
}

fn backward_inplace(c: &mut Criterion) {
    fn runner(prime_instance: Prime<u64>, nth_root: u64) -> Box<dyn FnMut()> {
        let ntt_table: Table<u64> = Table::<u64>::new(prime_instance, nth_root);
        let mut a: Vec<u64> = vec![0; (nth_root >> 1) as usize];
        for i in 0..a.len() {
            a[i] = i as u64;
        }
        Box::new(move || ntt_table.backward_inplace::<false>(&mut a))
    }

    let mut b: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("backward_inplace");
    for log_nth_root in 11..18 {
        let prime_instance: Prime<u64> = Prime::<u64>::new(0x1fffffffffe00001, 1);

        let runners = [("prime", { runner(prime_instance, 1 << log_nth_root) })];
        for (name, mut runner) in runners {
            let id = BenchmarkId::new(name, 1 << (log_nth_root - 1));
            b.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
        }
    }
}

fn backward_inplace_lazy(c: &mut Criterion) {
    fn runner(prime_instance: Prime<u64>, nth_root: u64) -> Box<dyn FnMut()> {
        let ntt_table: Table<u64> = Table::<u64>::new(prime_instance, nth_root);
        let mut a: Vec<u64> = vec![0; (nth_root >> 1) as usize];
        for i in 0..a.len() {
            a[i] = i as u64;
        }
        Box::new(move || ntt_table.backward_inplace::<true>(&mut a))
    }

    let mut b: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("backward_inplace_lazy");
    for log_nth_root in 11..17 {
        let prime_instance: Prime<u64> = Prime::<u64>::new(0x1fffffffffe00001, 1);

        let runners = [("prime", { runner(prime_instance, 1 << log_nth_root) })];
        for (name, mut runner) in runners {
            let id = BenchmarkId::new(name, 1 << (log_nth_root - 1));
            b.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
        }
    }
}

criterion_group!(
    benches,
    forward_inplace,
    forward_inplace_lazy,
    backward_inplace,
    backward_inplace_lazy
);
criterion_main!(benches);
