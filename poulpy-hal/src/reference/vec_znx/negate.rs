use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use crate::{
    api::{ModuleNew, VecZnxNegate, VecZnxNegateInplace},
    layouts::{Backend, FillUniform, Module, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::znx::{znx_negate_i64_ref, znx_negate_inplace_i64_ref, znx_zero_ref},
    source::Source,
};

pub fn vec_znx_negate_ref<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    let a: VecZnx<&[u8]> = a.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let min_size: usize = res.size().min(a.size());

    for j in 0..min_size {
        znx_negate_i64_ref(res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res.size() {
        znx_zero_ref(res.at_mut(res_col, j));
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn vec_znx_negate_avx<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    use crate::reference::znx::znx_negate_i64_avx;

    let a: VecZnx<&[u8]> = a.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let min_size: usize = res.size().min(a.size());

    for j in 0..min_size {
        znx_negate_i64_avx(res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res.size() {
        znx_zero_ref(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_negate_inplace_ref<R>(res: &mut R, res_col: usize)
where
    R: VecZnxToMut,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    for j in 0..res.size() {
        znx_negate_inplace_i64_ref(res.at_mut(res_col, j));
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn vec_znx_negate_inplace_avx<R>(res: &mut R, res_col: usize)
where
    R: VecZnxToMut,
{
    use crate::reference::znx::znx_negate_inplace_i64_avx;
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    for j in 0..res.size() {
        znx_negate_inplace_i64_avx(res.at_mut(res_col, j));
    }
}

pub fn test_vec_znx_negate<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxNegate,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, a_size);
        a.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        for res_size in [1, 2, 6, 11] {
            let mut res_0: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);
            let mut res_1: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);

            res_0
                .raw_mut()
                .iter_mut()
                .for_each(|x| *x = source.next_i32() as i64);

            res_1.raw_mut().copy_from_slice(res_0.raw());

            for i in 0..cols {
                vec_znx_negate_ref(&mut res_0, i, &a, i);
                module.vec_znx_negate(&mut res_1, i, &a, i);
            }

            assert_eq!(res_0.raw(), res_1.raw());
        }
    }
}

pub fn test_vec_znx_negate_inplace<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxNegateInplace,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for res_size in [1, 2, 6, 11] {
        let mut res_0: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);
        let mut res_1: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);

        res_0
            .raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        res_1.raw_mut().copy_from_slice(res_0.raw());

        for i in 0..cols {
            vec_znx_negate_inplace_ref(&mut res_0, i);
            module.vec_znx_negate_inplace(&mut res_1, i);
        }

        assert_eq!(res_0.raw(), res_1.raw());
    }
}

pub fn bench_vec_znx_negate<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxNegate + ModuleNew<B>,
{
    let group_name: String = format!("vec_znx_negate::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxNegate + ModuleNew<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);

        // Fill a with random i64
        a.fill_uniform(&mut source);
        b.fill_uniform(&mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_negate(&mut b, i, &a, i);
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2],));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_negate_inplace<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxNegateInplace + ModuleNew<B>,
{
    let group_name: String = format!("vec_znx_negate_inplace::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxNegateInplace + ModuleNew<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);

        // Fill a with random i64
        a.fill_uniform(&mut source);
        move || {
            for i in 0..cols {
                module.vec_znx_negate_inplace(&mut a, i);
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2]));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

#[cfg(all(test, any(target_arch = "x86_64", target_arch = "x86")))]
mod tests {
    use super::*;

    #[target_feature(enable = "avx2")]
    fn test_znx_negate_avx_internal() {
        let cols: usize = 2;
        let mut source: Source = Source::new([0u8; 32]);

        for a_size in [1, 2, 6, 11] {
            let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(32, cols, a_size);
            a.raw_mut()
                .iter_mut()
                .for_each(|x| *x = source.next_i32() as i64);

            for res_size in [1, 2, 6, 11] {
                let mut res_0: VecZnx<Vec<u8>> = VecZnx::alloc(32, cols, res_size);
                let mut res_1: VecZnx<Vec<u8>> = VecZnx::alloc(32, cols, res_size);

                res_0
                    .raw_mut()
                    .iter_mut()
                    .for_each(|x| *x = source.next_i32() as i64);
                res_1
                    .raw_mut()
                    .iter_mut()
                    .for_each(|x| *x = source.next_i32() as i64);

                for i in 0..cols {
                    vec_znx_negate_ref(&mut res_0, i, &a, i);
                    vec_znx_negate_avx(&mut res_1, i, &a, i);
                }

                assert_eq!(res_0.raw(), res_1.raw());
            }
        }
    }

    #[test]
    fn test_znx_negate_avx() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: CPU lacks avx2");
            return;
        };
        unsafe {
            test_znx_negate_avx_internal();
        }
    }

    #[target_feature(enable = "avx2")]
    fn test_znx_negate_inplace_avx_internal() {
        let cols: usize = 2;
        let mut source: Source = Source::new([0u8; 32]);

        for res_size in [1, 2, 6, 11] {
            let mut res_0: VecZnx<Vec<u8>> = VecZnx::alloc(32, cols, res_size);
            let mut res_1: VecZnx<Vec<u8>> = VecZnx::alloc(32, cols, res_size);

            res_0
                .raw_mut()
                .iter_mut()
                .for_each(|x| *x = source.next_i32() as i64);
            res_1.raw_mut().copy_from_slice(res_0.raw());

            for i in 0..cols {
                vec_znx_negate_inplace_ref(&mut res_0, i);
                vec_znx_negate_inplace_avx(&mut res_1, i);
            }

            assert_eq!(res_0.raw(), res_1.raw());
        }
    }

    #[test]
    fn test_znx_negate_inplace_avx() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: CPU lacks avx2");
            return;
        };
        unsafe {
            test_znx_negate_inplace_avx_internal();
        }
    }
}
