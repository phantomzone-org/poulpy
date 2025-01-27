use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use math::modulus::montgomery::Montgomery;
use math::modulus::{WordOps, ONCE};
use math::poly::Poly;
use math::ring::Ring;

fn a_add_b_into_b(c: &mut Criterion) {
    fn runner(ring: Ring<u64>) -> Box<dyn FnMut()> {
        let mut a: Poly<u64> = ring.new_poly();
        let mut b: Poly<u64> = ring.new_poly();
        for i in 0..ring.n() {
            a.0[i] = i as u64;
            b.0[i] = i as u64;
        }
        Box::new(move || {
            ring.a_add_b_into_b::<ONCE>(&a, &mut b);
        })
    }

    let mut b: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("a_add_b_into_b");
    for log_n in 11..17 {
        let n: usize = 1 << log_n as usize;
        let q_base: u64 = 0x1fffffffffe00001u64;
        let q_power: usize = 1usize;
        let r: Ring<u64> = Ring::<u64>::new(n, q_base, q_power);
        let runners = [("prime", { runner(r) })];
        for (name, mut runner) in runners {
            let id = BenchmarkId::new(name, n);
            b.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
        }
    }
}

fn a_mul_b_montgomery_into_a(c: &mut Criterion) {
    fn runner(ring: Ring<u64>) -> Box<dyn FnMut()> {
        let mut a: Poly<Montgomery<u64>> = ring.new_poly();
        let mut b: Poly<u64> = ring.new_poly();
        for i in 0..ring.n() {
            a.0[i] = ring.modulus.montgomery.prepare::<ONCE>(i as u64);
            b.0[i] = i as u64;
        }
        Box::new(move || {
            ring.a_mul_b_montgomery_into_a::<ONCE>(&a, &mut b);
        })
    }

    let mut b: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("a_mul_b_montgomery_into_a");
    for log_n in 11..17 {
        let n: usize = 1 << log_n as usize;
        let q_base: u64 = 0x1fffffffffe00001u64;
        let q_power: usize = 1usize;
        let r: Ring<u64> = Ring::<u64>::new(n, q_base, q_power);
        let runners = [("prime", { runner(r) })];
        for (name, mut runner) in runners {
            let id = BenchmarkId::new(name, n);
            b.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
        }
    }
}

fn a_mul_b_montgomery_into_c(c: &mut Criterion) {
    fn runner(ring: Ring<u64>) -> Box<dyn FnMut()> {
        let mut a: Poly<Montgomery<u64>> = ring.new_poly();
        let mut b: Poly<u64> = ring.new_poly();
        let mut c: Poly<u64> = ring.new_poly();
        for i in 0..ring.n() {
            a.0[i] = ring.modulus.montgomery.prepare::<ONCE>(i as u64);
            b.0[i] = i as u64;
        }
        Box::new(move || {
            ring.a_mul_b_montgomery_into_c::<ONCE>(&a, &b, &mut c);
        })
    }

    let mut b: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("a_mul_b_montgomery_into_c");
    for log_n in 11..17 {
        let n: usize = 1 << log_n as usize;
        let q_base: u64 = 0x1fffffffffe00001u64;
        let q_power: usize = 1usize;
        let r: Ring<u64> = Ring::<u64>::new(n, q_base, q_power);
        let runners = [("prime", { runner(r) })];
        for (name, mut runner) in runners {
            let id = BenchmarkId::new(name, n);
            b.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
        }
    }
}

fn a_ith_digit_unsigned_base_scalar_b_into_c(c: &mut Criterion) {
    fn runner(ring: Ring<u64>, base: usize, d: usize) -> Box<dyn FnMut()> {
        let mut a: Poly<Montgomery<u64>> = ring.new_poly();
        let mut b: Poly<u64> = ring.new_poly();
        for i in 0..ring.n() {
            a.0[i] = i as u64;
        }
        Box::new(move || {
            (0..d).for_each(|i| {
                ring.a_ith_digit_unsigned_base_scalar_b_into_c(i, &a, &base, &mut b);
            })
        })
    }

    let mut b: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("a_ith_digit_unsigned_base_scalar_b_into_c");
    for log_n in 11..12 {
        let n: usize = 1 << log_n as usize;
        let q_base: u64 = 0x1fffffffffe00001u64;
        let q_power: usize = 1usize;
        let ring: Ring<u64> = Ring::<u64>::new(n, q_base, q_power);
        let base: usize = 7;
        let logq: usize = ring.modulus.q.log2();
        let d: usize = (logq + base - 1) / base;
        let runners = [(format!("prime/logq={}/w={}/d={}", logq, base, d), {
            runner(ring, base, d)
        })];
        for (name, mut runner) in runners {
            let id = BenchmarkId::new(name, n);
            b.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
        }
    }
}

fn a_ith_digit_signed_base_scalar_b_into_c_balanced_false(c: &mut Criterion) {
    fn runner(ring: Ring<u64>, base: usize, d: usize) -> Box<dyn FnMut()> {
        let mut a: Poly<Montgomery<u64>> = ring.new_poly();
        let mut carry: Poly<u64> = ring.new_poly();
        let mut b: Poly<u64> = ring.new_poly();
        for i in 0..ring.n() {
            a.0[i] = i as u64;
        }
        Box::new(move || {
            (0..d).for_each(|i| {
                ring.a_ith_digit_signed_base_scalar_b_into_c::<false>(
                    i, &a, &base, &mut carry, &mut b,
                );
            })
        })
    }

    let mut b: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("a_ith_digit_signed_base_scalar_b_into_c::<BALANCED=false>");
    for log_n in 11..12 {
        let n: usize = 1 << log_n as usize;
        let q_base: u64 = 0x1fffffffffe00001u64;
        let q_power: usize = 1usize;
        let ring: Ring<u64> = Ring::<u64>::new(n, q_base, q_power);
        let base: usize = 7;
        let logq: usize = ring.modulus.q.log2();
        let d: usize = (logq + base - 1) / base;
        let runners = [(format!("prime/logq={}/w={}/d={}", logq, base, d), {
            runner(ring, base, d)
        })];
        for (name, mut runner) in runners {
            let id = BenchmarkId::new(name, n);
            b.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
        }
    }
}

fn a_ith_digit_signed_base_scalar_b_into_c_balanced_true(c: &mut Criterion) {
    fn runner(ring: Ring<u64>, base: usize, d: usize) -> Box<dyn FnMut()> {
        let mut a: Poly<Montgomery<u64>> = ring.new_poly();
        let mut carry: Poly<u64> = ring.new_poly();
        let mut b: Poly<u64> = ring.new_poly();
        for i in 0..ring.n() {
            a.0[i] = i as u64;
        }
        Box::new(move || {
            (0..d).for_each(|i| {
                ring.a_ith_digit_signed_base_scalar_b_into_c::<true>(
                    i, &a, &base, &mut carry, &mut b,
                );
            })
        })
    }

    let mut b: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("a_ith_digit_signed_base_scalar_b_into_c::<BALANCED=true>");
    for log_n in 11..12 {
        let n: usize = 1 << log_n as usize;
        let q_base: u64 = 0x1fffffffffe00001u64;
        let q_power: usize = 1usize;
        let ring: Ring<u64> = Ring::<u64>::new(n, q_base, q_power);
        let base: usize = 7;
        let logq: usize = ring.modulus.q.log2();
        let d: usize = (logq + base - 1) / base;
        let runners = [(format!("prime/logq={}/w={}/d={}", logq, base, d), {
            runner(ring, base, d)
        })];
        for (name, mut runner) in runners {
            let id = BenchmarkId::new(name, n);
            b.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
        }
    }
}

criterion_group!(
    benches,
    a_add_b_into_b,
    a_mul_b_montgomery_into_a,
    a_mul_b_montgomery_into_c,
    a_ith_digit_unsigned_base_scalar_b_into_c,
    a_ith_digit_signed_base_scalar_b_into_c_balanced_false,
    a_ith_digit_signed_base_scalar_b_into_c_balanced_true,
);
criterion_main!(benches);
