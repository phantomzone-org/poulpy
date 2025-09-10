use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rand::RngCore;

use crate::{
    api::{ModuleNew, VecZnxDftAdd, VecZnxDftAddInplace},
    layouts::{Backend, DataViewMut, Module, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, ZnxInfos, ZnxView, ZnxViewMut},
    oep::VecZnxDftAllocBytesImpl,
    reference::{
        reim::{ReimArithmetic, ReimArithmeticRef},
        vec_znx_dft::fft64::assert_approx_eq_slice,
    },
    source::Source,
};

pub fn vec_znx_dft_add<R, A, B, BE, REIMARI>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarPrep = f64>,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
    B: VecZnxDftToRef<BE>,
    REIMARI: ReimArithmetic,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();
    let b: VecZnxDft<&[u8], BE> = b.to_ref();

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
            REIMARI::reim_add(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            REIMARI::reim_copy(res.at_mut(res_col, j), b.at(b_col, j));
        }

        for j in cpy_size..res_size {
            REIMARI::reim_zero(res.at_mut(res_col, j));
        }
    } else {
        let sum_size: usize = b_size.min(res_size);
        let cpy_size: usize = a_size.min(res_size);

        for j in 0..sum_size {
            REIMARI::reim_add(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            REIMARI::reim_copy(res.at_mut(res_col, j), a.at(a_col, j));
        }

        for j in cpy_size..res_size {
            REIMARI::reim_zero(res.at_mut(res_col, j));
        }
    }
}

pub fn vec_znx_dft_add_inplace<R, A, BE, REIMARI>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = f64>,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
    REIMARI: ReimArithmetic,
{
    let a: VecZnxDft<&[u8], BE> = a.to_ref();
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let sum_size: usize = a_size.min(res_size);

    for j in 0..sum_size {
        REIMARI::reim_add_inplace(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

pub fn test_vec_znx_dft_add<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAdd<B>,
    B: Backend<ScalarPrep = f64> + VecZnxDftAllocBytesImpl<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, a_size);
        a.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as f64);

        for b_size in [1, 2, 6, 11] {
            let mut b: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, b_size);
            b.raw_mut()
                .iter_mut()
                .for_each(|x| *x = source.next_i32() as f64);

            for res_size in [1, 2, 6, 11] {
                let mut res_0: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, res_size);
                let mut res_1: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, res_size);

                // Set d to garbage
                source.fill_bytes(res_0.data_mut());
                source.fill_bytes(res_1.data_mut());

                // Reference
                for i in 0..cols {
                    vec_znx_dft_add::<_, _, _, _, ReimArithmeticRef>(&mut res_0, i, &a, i, &b, i);
                    module.vec_znx_dft_add(&mut res_1, i, &a, i, &b, i);
                }

                assert_approx_eq_slice(res_0.raw(), res_1.raw(), 1e-15);
            }
        }
    }
}

pub fn test_vec_znx_dft_add_inplace<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAddInplace<B>,
    B: Backend<ScalarPrep = f64> + VecZnxDftAllocBytesImpl<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, a_size);
        a.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as f64);

        for res_size in [1, 2, 6, 11] {
            let mut res_0: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, res_size);
            let mut res_1: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, res_size);

            res_0
                .raw_mut()
                .iter_mut()
                .for_each(|x| *x = source.next_i32() as f64);

            res_1.raw_mut().copy_from_slice(res_0.raw());

            for i in 0..cols {
                vec_znx_dft_add_inplace::<_, _, _, ReimArithmeticRef>(&mut res_0, i, &a, i);
                module.vec_znx_dft_add_inplace(&mut res_1, i, &a, i);
            }

            assert_approx_eq_slice(res_0.raw(), res_1.raw(), 1e-15);
        }
    }
}

pub fn bench_vec_znx_dft_add<B>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxDftAdd<B> + ModuleNew<B>,
    B: Backend + VecZnxDftAllocBytesImpl<B>,
{
    let group_name: String = format!("vec_znx_dft_add::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxDftAdd<B> + ModuleNew<B>,
        B: Backend + VecZnxDftAllocBytesImpl<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, size);
        let mut b: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, size);
        let mut c: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, size);

        source.fill_bytes(a.data_mut());
        source.fill_bytes(b.data_mut());
        source.fill_bytes(c.data_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_dft_add(&mut c, i, &a, i, &b, i);
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

pub fn bench_vec_znx_dft_add_inplace<B>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxDftAddInplace<B> + ModuleNew<B>,
    B: Backend + VecZnxDftAllocBytesImpl<B>,
{
    let group_name: String = format!("vec_znx_dft_add_inplace::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxDftAddInplace<B> + ModuleNew<B>,
        B: Backend + VecZnxDftAllocBytesImpl<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, size);
        let mut c: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut());
        source.fill_bytes(c.data_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_dft_add_inplace(&mut c, i, &a, i);
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
