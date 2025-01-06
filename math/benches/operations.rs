use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use math::ring::Ring;
use math::modulus::VectorOperations;
use math::modulus::montgomery::Montgomery;
use math::modulus::ONCE;
use math::CHUNK;

fn va_add_vb_into_vb(c: &mut Criterion) {
    fn runner(r: Ring<u64>) -> Box<dyn FnMut()> {
        
        let mut p0: math::poly::Poly<u64> = r.new_poly();
        let mut p1: math::poly::Poly<u64> = r.new_poly();
        for i in 0..p0.n(){
            p0.0[i] = i as u64;
            p1.0[i] = i as u64;
        }
        Box::new(move || {
            r.modulus.va_add_vb_into_vb::<CHUNK, ONCE>(&p0.0, &mut p1.0);
        })
    }

    let mut b: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> = c.benchmark_group("va_add_vb_into_vb");
    for log_n in 11..17 {

        let n: usize = 1<<log_n as usize;
        let q_base: u64 = 0x1fffffffffe00001u64;
        let q_power: usize = 1usize;
        let r: Ring<u64> = Ring::<u64>::new(n, q_base, q_power);
        let runners = [
            ("prime", {
                runner(r)
            }),
        ];
        for (name, mut runner) in runners {
            let id = BenchmarkId::new(name, n);
            b.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
        }
    }
}

fn va_mont_mul_vb_into_vb(c: &mut Criterion) {
    fn runner(r: Ring<u64>) -> Box<dyn FnMut()> {
        
        let mut p0: math::poly::Poly<Montgomery<u64>> = r.new_poly();
        let mut p1: math::poly::Poly<u64> = r.new_poly();
        for i in 0..p0.n(){
            p0.0[i] = r.modulus.montgomery.prepare::<ONCE>(i as u64);
            p1.0[i] = i as u64;
        }
        Box::new(move || {
            r.modulus.va_mont_mul_vb_into_vb::<CHUNK, ONCE>(&p0.0, &mut p1.0);
        })
    }

    let mut b: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> = c.benchmark_group("va_mont_mul_vb_into_vb");
    for log_n in 11..17 {

        let n: usize = 1<<log_n as usize;
        let q_base: u64 = 0x1fffffffffe00001u64;
        let q_power: usize = 1usize;
        let r: Ring<u64> = Ring::<u64>::new(n, q_base, q_power);
        let runners = [
            ("prime", {
                runner(r)
            }),
        ];
        for (name, mut runner) in runners {
            let id = BenchmarkId::new(name, n);
            b.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
        }
    }
}

fn va_mont_mul_vb_into_vc(c: &mut Criterion) {
    fn runner(r: Ring<u64>) -> Box<dyn FnMut()> {
        
        let mut p0: math::poly::Poly<Montgomery<u64>> = r.new_poly();
        let mut p1: math::poly::Poly<u64> = r.new_poly();
        let mut p2: math::poly::Poly<u64> = r.new_poly();
        for i in 0..p0.n(){
            p0.0[i] = r.modulus.montgomery.prepare::<ONCE>(i as u64);
            p1.0[i] = i as u64;
        }
        Box::new(move || {
            r.modulus.va_mont_mul_vb_into_vc::<CHUNK,ONCE>(&p0.0, & p1.0, &mut p2.0);
        })
    }

    let mut b: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> = c.benchmark_group("va_mont_mul_vb_into_vc");
    for log_n in 11..17 {

        let n: usize = 1<<log_n as usize;
        let q_base: u64 = 0x1fffffffffe00001u64;
        let q_power: usize = 1usize;
        let r: Ring<u64> = Ring::<u64>::new(n, q_base, q_power);
        let runners = [
            ("prime", {
                runner(r)
            }),
        ];
        for (name, mut runner) in runners {
            let id = BenchmarkId::new(name, n);
            b.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
        }
    }
}

criterion_group!(benches, va_add_vb_into_vb, va_mont_mul_vb_into_vb, va_mont_mul_vb_into_vc);
criterion_main!(benches);
