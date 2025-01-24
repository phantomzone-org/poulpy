use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use math::modulus::WordOps;
use math::ring::Ring;
use math::poly::Poly;

fn ntt(c: &mut Criterion) {

    fn runner<'a, const INPLACE: bool, const LAZY:bool>(ring: &'a Ring<u64>) -> Box<dyn FnMut() + 'a > {
        let mut a: Poly<u64> = ring.new_poly();
        for i in 0..a.n() {
            a.0[i] = i as u64;
        }
        if INPLACE{
            Box::new(move || ring.ntt_inplace::<LAZY>(&mut a))
        }else{
            let mut b: Poly<u64> = ring.new_poly();
            Box::new(move || ring.ntt::<LAZY>(&a, &mut b))
        }
    }

    let q: u64 = 0x1fffffffffe00001u64;

    let mut b: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("ntt");

    for log_n in 10..17 {

        let ring = Ring::new(1<<log_n, q, 1);

        let runners: [(String, Box<dyn FnMut()>); 4] = [
            (format!("inplace=true/LAZY=true/q={}", q.log2()), { runner::<true, true>(&ring) }),
            (format!("inplace=true/LAZY=false/q={}", q.log2()), { runner::<true, false>(&ring) }),
            (format!("inplace=false/LAZY=true/q={}", q.log2()), { runner::<false, true>(&ring) }),
            (format!("inplace=false/LAZY=false/q={}", q.log2()), { runner::<false, false>(&ring) }),
            ];

        for (name, mut runner) in runners {
            let id: BenchmarkId = BenchmarkId::new(name, format!("n={}", 1 << log_n));
            b.bench_with_input(id, &(), |b: &mut criterion::Bencher<'_>, _| b.iter(&mut runner));
        }
    }
}

fn intt(c: &mut Criterion) {

    fn runner<'a, const INPLACE: bool, const LAZY:bool>(ring: &'a Ring<u64>) -> Box<dyn FnMut() + 'a > {
        let mut a: Poly<u64> = ring.new_poly();
        for i in 0..a.n() {
            a.0[i] = i as u64;
        }
        if INPLACE{
            Box::new(move || ring.intt_inplace::<LAZY>(&mut a))
        }else{
            let mut b: Poly<u64> = ring.new_poly();
            Box::new(move || ring.intt::<LAZY>(&a, &mut b))
        }
    }

    let q: u64 = 0x1fffffffffe00001u64;

    let mut b: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("intt");

    for log_n in 10..17 {

        let ring = Ring::new(1<<log_n, q, 1);

        let runners: [(String, Box<dyn FnMut()>); 4] = [
            (format!("inplace=true/LAZY=true/q={}", q.log2()), { runner::<true, true>(&ring) }),
            (format!("inplace=true/LAZY=false/q={}", q.log2()), { runner::<true, false>(&ring) }),
            (format!("inplace=false/LAZY=true/q={}", q.log2()), { runner::<false, true>(&ring) }),
            (format!("inplace=false/LAZY=false/q={}", q.log2()), { runner::<false, false>(&ring) }),
            ];

        for (name, mut runner) in runners {
            let id: BenchmarkId = BenchmarkId::new(name, format!("n={}", 1 << log_n));
            b.bench_with_input(id, &(), |b: &mut criterion::Bencher<'_>, _| b.iter(&mut runner));
        }
    }
}

criterion_group!(
    benches,
    ntt,
    intt,
);
criterion_main!(benches);
