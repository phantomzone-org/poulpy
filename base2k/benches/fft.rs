use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use spqlios::bindings::{
    new_reim_fft_precomp, new_reim_ifft_precomp, reim_fft, reim_fft_precomp,
    reim_fft_precomp_get_buffer, reim_from_znx64_simple, reim_ifft, reim_ifft_precomp,
    reim_ifft_precomp_get_buffer,
};
use std::ffi::c_void;

fn fft(c: &mut Criterion) {
    fn forward<'a>(
        m: u32,
        log_bound: u32,
        reim_fft_precomp: *mut reim_fft_precomp,
        a: &'a [i64],
    ) -> Box<dyn FnMut() + 'a> {
        unsafe {
            let buf_a: *mut f64 = reim_fft_precomp_get_buffer(reim_fft_precomp, 0);
            reim_from_znx64_simple(m as u32, log_bound as u32, buf_a as *mut c_void, a.as_ptr());
            Box::new(move || reim_fft(reim_fft_precomp, buf_a))
        }
    }

    fn backward<'a>(
        m: u32,
        log_bound: u32,
        reim_ifft_precomp: *mut reim_ifft_precomp,
        a: &'a [i64],
    ) -> Box<dyn FnMut() + 'a> {
        Box::new(move || unsafe {
            let buf_a: *mut f64 = reim_ifft_precomp_get_buffer(reim_ifft_precomp, 0);
            reim_from_znx64_simple(m as u32, log_bound as u32, buf_a as *mut c_void, a.as_ptr());
            reim_ifft(reim_ifft_precomp, buf_a);
        })
    }

    let mut b: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("fft");

    for log_n in 10..17 {
        let n: usize = 1 << log_n;
        let m: usize = n >> 1;
        let log_bound: u32 = 19;

        let mut a: Vec<i64> = vec![i64::default(); n];
        a.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);

        unsafe {
            let reim_fft_precomp: *mut reim_fft_precomp = new_reim_fft_precomp(m as u32, 1);
            let reim_ifft_precomp: *mut reim_ifft_precomp = new_reim_ifft_precomp(m as u32, 1);

            let runners: [(String, Box<dyn FnMut()>); 2] = [
                (format!("forward"), {
                    forward(m as u32, log_bound, reim_fft_precomp, &a)
                }),
                (format!("backward"), {
                    backward(m as u32, log_bound, reim_ifft_precomp, &a)
                }),
            ];

            for (name, mut runner) in runners {
                let id: BenchmarkId = BenchmarkId::new(name, format!("n={}", 1 << log_n));
                b.bench_with_input(id, &(), |b: &mut criterion::Bencher<'_>, _| {
                    b.iter(&mut runner)
                });
            }
        }
    }
}

criterion_group!(benches, fft,);
criterion_main!(benches);
