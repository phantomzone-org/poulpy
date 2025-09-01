use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use crate::{
    api::{ModuleNew, VecZnxAdd, VecZnxAddInplace, VecZnxAddScalar, VecZnxAddScalarInplace},
    layouts::{
        Backend, FillUniform, Module, ScalarZnx, ScalarZnxToRef, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut,
        ZnxZero,
    },
    oep::{ModuleNewImpl, VecZnxAddImpl, VecZnxAddInplaceImpl},
    reference::znx::{znx_add_i64_ref, znx_add_inplace_i64_ref, znx_copy_ref},
    source::Source,
};

pub fn vec_znx_add_ref<R, A, B>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    B: VecZnxToRef,
{
    let a: VecZnx<&[u8]> = a.to_ref();
    let b: VecZnx<&[u8]> = b.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
        assert_eq!(b.n(), res.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let b_size: usize = b.size();

    if a_size <= b_size {
        let sum_size: usize = a_size.min(res_size);
        let cpy_size: usize = b_size.min(res_size);

        for j in 0..sum_size {
            znx_add_i64_ref(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            znx_copy_ref(res.at_mut(res_col, j), b.at(b_col, j));
        }

        for j in cpy_size..res_size {
            res.zero_at(res_col, j);
        }
    } else {
        let sum_size: usize = b_size.min(res_size);
        let cpy_size: usize = a_size.min(res_size);

        for j in 0..sum_size {
            znx_add_i64_ref(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            znx_copy_ref(res.at_mut(res_col, j), a.at(a_col, j));
        }

        for j in cpy_size..res_size {
            res.zero_at(res_col, j);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn vec_znx_add_avx<R, A, B>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    B: VecZnxToRef,
{
    use crate::reference::znx::znx_add_i64_avx;

    let a: VecZnx<&[u8]> = a.to_ref();
    let b: VecZnx<&[u8]> = b.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
        assert_eq!(b.n(), res.n());
        assert_ne!(a.as_ptr(), b.as_ptr());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let b_size: usize = b.size();

    if a_size <= b_size {
        let sum_size: usize = a_size.min(res_size);
        let cpy_size: usize = b_size.min(res_size);

        for j in 0..sum_size {
            znx_add_i64_avx(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            znx_copy_ref(res.at_mut(res_col, j), b.at(b_col, j));
        }

        for j in cpy_size..res_size {
            res.zero_at(res_col, j);
        }
    } else {
        let sum_size: usize = b_size.min(res_size);
        let cpy_size: usize = a_size.min(res_size);

        for j in 0..sum_size {
            znx_add_i64_avx(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            znx_copy_ref(res.at_mut(res_col, j), a.at(a_col, j));
        }

        for j in cpy_size..res_size {
            res.zero_at(res_col, j);
        }
    }
}

pub fn vec_znx_add_inplace_ref<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
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

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let sum_size: usize = a_size.min(res_size);

    for j in 0..sum_size {
        znx_add_inplace_i64_ref(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn vec_znx_add_inplace_avx<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    use crate::reference::znx::znx_add_inplace_i64_avx;

    let a: VecZnx<&[u8]> = a.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let sum_size: usize = a_size.min(res_size);

    for j in 0..sum_size {
        znx_add_inplace_i64_avx(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

pub fn vec_znx_add_scalar_ref<R, A, B>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize, b_limb: usize)
where
    R: VecZnxToMut,
    A: ScalarZnxToRef,
    B: VecZnxToRef,
{
    let a: ScalarZnx<&[u8]> = a.to_ref();
    let b: VecZnx<&[u8]> = b.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    let min_size: usize = b.size().min(res.size());

    #[cfg(debug_assertions)]
    {
        assert!(
            b_limb < min_size,
            "b_limb: {} > min_size: {}",
            b_limb,
            min_size
        );
    }

    for j in 0..min_size {
        if j == b_limb {
            znx_add_i64_ref(res.at_mut(res_col, j), a.at(a_col, 0), b.at(b_col, j));
        } else {
            res.at_mut(res_col, j).copy_from_slice(b.at(b_col, j));
        }
    }

    for j in min_size..res.size() {
        res.zero_at(res_col, j);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn vec_znx_add_scalar_avx<R, A, B>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize, b_limb: usize)
where
    R: VecZnxToMut,
    A: ScalarZnxToRef,
    B: VecZnxToRef,
{
    use crate::reference::znx::znx_add_i64_avx;

    let a: ScalarZnx<&[u8]> = a.to_ref();
    let b: VecZnx<&[u8]> = b.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    let min_size: usize = a.size().min(res.size());

    #[cfg(debug_assertions)]
    {
        assert!(b_limb < min_size);
    }

    for j in 0..min_size {
        if j == b_limb {
            znx_add_i64_avx(res.at_mut(res_col, j), a.at(a_col, 0), b.at(b_col, j));
        } else {
            znx_copy_ref(res.at_mut(res_col, j), b.at(b_col, j));
        }
    }
}

pub fn vec_znx_add_scalar_inplace_ref<R, A>(res: &mut R, res_col: usize, res_limb: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: ScalarZnxToRef,
{
    let a: ScalarZnx<&[u8]> = a.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert!(res_limb < res.size());
    }

    znx_add_inplace_i64_ref(res.at_mut(res_col, res_limb), a.at(a_col, 0));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn vec_znx_add_scalar_inplace_avx<R, A>(res: &mut R, res_col: usize, res_limb: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: ScalarZnxToRef,
{
    use crate::reference::znx::znx_add_inplace_i64_avx;

    let a: ScalarZnx<&[u8]> = a.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert!(res_limb < res.size());
    }

    znx_add_inplace_i64_avx(res.at_mut(res_col, res_limb), a.at(a_col, 0));
}

pub fn test_vec_znx_add<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxAdd,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, a_size);
        a.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        for b_size in [1, 2, 6, 11] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, b_size);
            b.raw_mut()
                .iter_mut()
                .for_each(|x| *x = source.next_i32() as i64);

            for res_size in [1, 2, 6, 11] {
                let mut res_0: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);
                let mut res_1: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);

                // Set d to garbage
                res_0.fill_uniform(&mut source);
                res_1.fill_uniform(&mut source);

                // Reference
                for i in 0..cols {
                    vec_znx_add_ref(&mut res_0, i, &a, i, &b, i);
                    module.vec_znx_add(&mut res_1, i, &a, i, &b, i);
                }

                assert_eq!(res_0.raw(), res_1.raw());
            }
        }
    }
}

pub fn test_vec_znx_add_inplace<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxAddInplace,
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

            res_1.raw_mut().copy_from_slice(&res_0.raw());

            for i in 0..cols {
                vec_znx_add_inplace_ref(&mut res_0, i, &a, i);
                module.vec_znx_add_inplace(&mut res_1, i, &a, i);
            }

            assert_eq!(res_0.raw(), res_1.raw());
        }
    }
}

pub fn test_vec_znx_add_scalar<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxAddScalar,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut a: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(module.n(), cols);
    a.raw_mut()
        .iter_mut()
        .for_each(|x| *x = source.next_i32() as i64);

    for a_size in [1, 2, 6, 11] {
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, a_size);
        b.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        for res_size in [1, 2, 6, 11] {
            let mut res_0: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);
            let mut res_1: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);

            // Set d to garbage
            res_0.fill_uniform(&mut source);
            res_1.fill_uniform(&mut source);

            // Reference
            for i in 0..cols {
                vec_znx_add_scalar_ref(&mut res_0, i, &a, i, &b, i, (res_size.min(a_size)) - 1);
                module.vec_znx_add_scalar(&mut res_1, i, &a, i, &b, i, (res_size.min(a_size)) - 1);
            }

            assert_eq!(res_0.raw(), res_1.raw());
        }
    }
}

pub fn test_vec_znx_add_scalar_inplace<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxAddScalarInplace,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for res_size in [1, 2, 6, 11] {
        let mut res_0: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);
        let mut res_1: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);

        let mut b: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(module.n(), cols);
        b.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        res_0
            .raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        res_1.raw_mut().copy_from_slice(&res_0.raw());

        for i in 0..cols {
            vec_znx_add_scalar_inplace_ref(&mut res_0, i, res_size - 1, &b, i);
            module.vec_znx_add_scalar_inplace(&mut res_1, i, res_size - 1, &b, i);
        }

        assert_eq!(res_0.raw(), res_1.raw());
    }
}

pub fn bench_vec_znx_add<B: Backend>(c: &mut Criterion, label: &str)
where
    B: ModuleNewImpl<B> + VecZnxAddImpl<B>,
{
    let group_name: String = format!("vec_znx_add::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxAdd + ModuleNew<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut c: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);

        // Fill a with random i64
        a.fill_uniform(&mut source);
        b.fill_uniform(&mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_add(&mut c, i, &a, i, &b, i);
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

pub fn bench_vec_znx_add_inplace<B: Backend>(c: &mut Criterion, label: &str)
where
    B: ModuleNewImpl<B> + VecZnxAddInplaceImpl<B>,
{
    let group_name: String = format!("vec_znx_add_inplace::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxAddInplace + ModuleNew<B>,
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
                module.vec_znx_add_inplace(&mut b, i, &a, i);
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
    fn test_znx_add_avx_internal() {
        let cols: usize = 2;
        let mut source: Source = Source::new([0u8; 32]);

        for a_size in [1, 2, 6, 11] {
            let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(32, cols, a_size);
            a.raw_mut()
                .iter_mut()
                .for_each(|x| *x = source.next_i32() as i64);

            for b_size in [1, 2, 6, 11] {
                let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(32, cols, b_size);
                b.raw_mut()
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
                        vec_znx_add_ref(&mut res_0, i, &a, i, &b, i);
                        vec_znx_add_avx(&mut res_1, i, &a, i, &b, i);
                    }

                    assert_eq!(res_0.raw(), res_1.raw());
                }
            }
        }
    }

    #[test]
    fn test_znx_add_avx() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: CPU lacks avx2");
            return;
        };
        unsafe {
            test_znx_add_avx_internal();
        }
    }

    #[target_feature(enable = "avx2")]
    fn test_znx_add_inplace_avx_internal() {
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
                res_1.raw_mut().copy_from_slice(&res_0.raw());

                for i in 0..cols {
                    vec_znx_add_inplace_ref(&mut res_0, i, &a, i);
                    vec_znx_add_inplace_avx(&mut res_1, i, &a, i);
                }

                assert_eq!(res_0.raw(), res_1.raw());
            }
        }
    }

    #[test]
    fn test_znx_add_inplace_avx() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: CPU lacks avx2");
            return;
        };
        unsafe {
            test_znx_add_inplace_avx_internal();
        }
    }
}
