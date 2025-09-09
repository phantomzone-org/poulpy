use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use crate::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRotate, VecZnxRotateInplace, VecZnxRotateInplaceTmpBytes},
    layouts::{Backend, FillUniform, Module, ScratchOwned, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::znx::{znx_copy_ref, znx_rotate_i64_avx, znx_rotate_i64_ref, znx_zero_ref},
    source::Source,
};

pub fn vec_znx_rotate_inplace_tmp_bytes_ref(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_rotate_ref<R, A>(p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), a.n())
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let min_size: usize = res_size.min(a_size);

    for j in 0..min_size {
        znx_rotate_i64_ref(p, res.at_mut(res_col, j), a.at(a_col, j))
    }

    for j in min_size..res_size {
        znx_zero_ref(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_rotate_inplace_ref<R>(p: i64, res: &mut R, res_col: usize, tmp: &mut [i64])
where
    R: VecZnxToMut,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), tmp.len());
    }
    for j in 0..res.size() {
        znx_rotate_i64_ref(p, tmp, res.at(res_col, j));
        znx_copy_ref(res.at_mut(res_col, j), tmp);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn vec_znx_rotate_avx<R, A>(p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    use crate::reference::znx::znx_rotate_i64_avx;

    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), a.n())
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let min_size: usize = res_size.min(a_size);

    for j in 0..min_size {
        znx_rotate_i64_avx(p, res.at_mut(res_col, j), a.at(a_col, j))
    }

    for j in min_size..res_size {
        znx_zero_ref(res.at_mut(res_col, j));
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn vec_znx_rotate_inplace_avx<R, A>(p: i64, res: &mut R, res_col: usize, tmp: &mut [i64])
where
    R: VecZnxToMut,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), tmp.len());
    }
    for j in 0..res.size() {
        znx_rotate_i64_avx(p, tmp, res.at(res_col, j));
        znx_copy_ref(res.at_mut(res_col, j), tmp);
    }
}

pub fn test_vec_znx_rotate<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxRotate,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, a_size);

        // Fill a with random i64
        a.fill_uniform(&mut source);

        for res_size in [1, 2, 6, 11] {
            let mut r0: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);
            let mut r1: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);

            let p: i64 = -1;

            // Normalize on c
            for i in 0..cols {
                module.vec_znx_rotate(p, &mut r0, i, &a, i);
                vec_znx_rotate_ref(p, &mut r1, i, &a, i);
            }

            for i in 0..cols {
                for j in 0..res_size {
                    assert_eq!(r0.at(i, j), r1.at(i, j));
                }
            }
        }
    }
}

pub fn test_vec_znx_rotate_inplace<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxRotateInplace<B> + VecZnxRotateInplaceTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_rotate_inplace_tmp_bytes());

    let mut tmp = vec![0i64; module.n()];

    for size in [1, 2, 6, 11] {
        let mut r0: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut r1: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);

        // Fill a with random i64
        r0.fill_uniform(&mut source);
        znx_copy_ref(r1.raw_mut(), r0.raw());

        let p: i64 = -7;

        // Normalize on c
        for i in 0..cols {
            module.vec_znx_rotate_inplace(p, &mut r0, i, scratch.borrow());
            vec_znx_rotate_inplace_ref(p, &mut r1, i, &mut tmp);
        }

        for i in 0..cols {
            for j in 0..size {
                assert_eq!(r0.at(i, j), r1.at(i, j));
            }
        }
    }
}

pub fn bench_vec_znx_rotate<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxRotate + ModuleNew<B>,
{
    let group_name: String = format!("vec_znx_rotate::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxRotate + ModuleNew<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);

        // Fill a with random i64
        a.fill_uniform(&mut source);
        res.fill_uniform(&mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_rotate(-7, &mut res, i, &a, i);
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

pub fn bench_vec_znx_rotate_inplace<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxRotateInplace<B> + VecZnxRotateInplaceTmpBytes + ModuleNew<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_rotate_inplace::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxRotateInplace<B> + ModuleNew<B> + VecZnxRotateInplaceTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);

        let mut scratch = ScratchOwned::alloc(module.vec_znx_rotate_inplace_tmp_bytes());

        // Fill a with random i64
        res.fill_uniform(&mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_rotate_inplace(-7, &mut res, i, scratch.borrow());
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
