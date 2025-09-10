use std::hint::black_box;

use bytemuck::cast_slice_mut;
use criterion::{BenchmarkId, Criterion};
use rand::RngCore;

use crate::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftApply, VecZnxIdftApply, VecZnxIdftApplyConsume,
        VecZnxIdftApplyTmpA, VecZnxIdftApplyTmpBytes,
    },
    layouts::{
        Backend, Data, DataViewMut, Module, ScratchOwned, VecZnx, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftToMut,
        VecZnxDftToRef, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut,
    },
    oep::{VecZnxBigAllocBytesImpl, VecZnxDftAllocBytesImpl},
    reference::{
        reim::{
            ReimArithmetic, ReimArithmeticRef, ReimConv, ReimConvRef, ReimDFTExecute, ReimFFTRef, ReimFFTTable, ReimIFFTRef,
            ReimIFFTTable,
        },
        vec_znx_dft::fft64::assert_approx_eq_slice,
        znx::{ZnxArithmetic, ZnxArithmeticRef},
    },
    source::Source,
};

pub fn vec_znx_dft_apply<R, A, BE, ARI, CONV, FFT>(
    table: &ReimFFTTable<f64>,
    step: usize,
    offset: usize,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = f64>,
    R: VecZnxDftToMut<BE>,
    A: VecZnxToRef,
    ARI: ReimArithmetic,
    CONV: ReimConv,
    FFT: ReimDFTExecute<ReimFFTTable<f64>, f64>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert!(step > 0);
        assert_eq!(table.m() << 1, res.n());
        assert_eq!(a.n(), res.n());
    }

    let a_size: usize = a.size();
    let res_size: usize = res.size();

    let steps: usize = a_size.div_ceil(step);
    let min_steps: usize = res_size.min(steps);

    for j in 0..min_steps {
        let limb = offset + j * step;
        if limb < a_size {
            CONV::reim_from_znx_i64(res.at_mut(res_col, j), a.at(a_col, limb));
            FFT::reim_dft_execute(table, res.at_mut(res_col, j));
        }
    }

    (min_steps..res.size()).for_each(|j| {
        ARI::reim_zero(res.at_mut(res_col, j));
    });
}

pub fn vec_znx_idft_apply<R, A, BE, ZNXARI, REIARI, CONV, IFFT>(
    table: &ReimIFFTTable<f64>,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = f64, ScalarBig = i64>,
    R: VecZnxBigToMut<BE>,
    A: VecZnxDftToRef<BE>,
    ZNXARI: ZnxArithmetic,
    REIARI: ReimArithmetic,
    CONV: ReimConv,
    IFFT: ReimDFTExecute<ReimIFFTTable<f64>, f64>,
{
    let mut res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(table.m() << 1, res.n());
        assert_eq!(a.n(), res.n());
    }

    let res_size: usize = res.size();
    let min_size: usize = res_size.min(a.size());

    let divisor: f64 = table.m() as f64;

    for j in 0..min_size {
        let res_slice_f64: &mut [f64] = cast_slice_mut(res.at_mut(res_col, j));
        REIARI::reim_copy(res_slice_f64, a.at(a_col, j));
        IFFT::reim_dft_execute(table, res_slice_f64);
        CONV::reim_to_znx_i64_inplace(res_slice_f64, divisor);
    }

    for j in min_size..res_size {
        ZNXARI::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_idft_apply_tmpa<R, A, BE, ZNXARI, CONV, IFFT>(
    table: &ReimIFFTTable<f64>,
    res: &mut R,
    res_col: usize,
    a: &mut A,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = f64, ScalarBig = i64>,
    R: VecZnxBigToMut<BE>,
    A: VecZnxDftToMut<BE>,
    ZNXARI: ZnxArithmetic,
    CONV: ReimConv,
    IFFT: ReimDFTExecute<ReimIFFTTable<f64>, f64>,
{
    let mut res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let mut a: VecZnxDft<&mut [u8], BE> = a.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(table.m() << 1, res.n());
        assert_eq!(a.n(), res.n());
    }

    let res_size = res.size();
    let min_size: usize = res_size.min(a.size());

    let divisor: f64 = table.m() as f64;

    for j in 0..min_size {
        IFFT::reim_dft_execute(table, a.at_mut(a_col, j));
        CONV::reim_to_znx_i64(res.at_mut(res_col, j), divisor, a.at(a_col, j));
    }

    for j in min_size..res_size {
        ZNXARI::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_idft_apply_consume<D: Data, BE, CONV, IFFT>(
    table: &ReimIFFTTable<f64>,
    mut res: VecZnxDft<D, BE>,
) -> VecZnxBig<D, BE>
where
    BE: Backend<ScalarPrep = f64, ScalarBig = i64>,
    VecZnxDft<D, BE>: VecZnxDftToMut<BE>,
    CONV: ReimConv,
    IFFT: ReimDFTExecute<ReimIFFTTable<f64>, f64>,
{
    {
        let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(table.m() << 1, res.n());
        }

        let divisor: f64 = table.m() as f64;

        for i in 0..res.cols() {
            for j in 0..res.size() {
                IFFT::reim_dft_execute(table, res.at_mut(i, j));
                CONV::reim_to_znx_i64_inplace(res.at_mut(i, j), divisor);
            }
        }
    }

    res.into_big()
}

pub fn test_vec_znx_dft_apply<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftApply<B>,
    B: Backend<ScalarPrep = f64> + VecZnxDftAllocBytesImpl<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let table: ReimFFTTable<f64> = ReimFFTTable::<f64>::new(module.n() >> 1);

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, a_size);
        a.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        for res_size in [1, 2, 6, 11] {
            let mut res_0: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, res_size);
            let mut res_1: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, res_size);

            // Set d to garbage
            source.fill_bytes(res_0.data_mut());
            source.fill_bytes(res_1.data_mut());

            for i in 0..cols {
                module.vec_znx_dft_apply(1, 0, &mut res_1, i, &a, i);
                vec_znx_dft_apply::<_, _, _, ReimArithmeticRef, ReimConvRef, ReimFFTRef>(&table, 1, 0, &mut res_0, i, &a, i);
            }
            assert_approx_eq_slice(res_0.raw(), res_1.raw(), 1e-10);

            for i in 0..cols {
                module.vec_znx_dft_apply(1, 1, &mut res_1, i, &a, i);
                vec_znx_dft_apply::<_, _, _, ReimArithmeticRef, ReimConvRef, ReimFFTRef>(&table, 1, 1, &mut res_0, i, &a, i);
            }
            assert_approx_eq_slice(res_0.raw(), res_1.raw(), 1e-10);

            for i in 0..cols {
                module.vec_znx_dft_apply(2, 1, &mut res_1, i, &a, i);
                vec_znx_dft_apply::<_, _, _, ReimArithmeticRef, ReimConvRef, ReimFFTRef>(&table, 2, 1, &mut res_0, i, &a, i);
            }
            assert_approx_eq_slice(res_0.raw(), res_1.raw(), 1e-10);
        }
    }
}

pub fn test_vec_znx_idft_apply<B>(module: &Module<B>)
where
    Module<B>: VecZnxIdftApply<B>,
    B: Backend<ScalarPrep = f64, ScalarBig = i64> + VecZnxDftAllocBytesImpl<B> + VecZnxBigAllocBytesImpl<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let table: ReimIFFTTable<f64> = ReimIFFTTable::<f64>::new(module.n() >> 1);
    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(0);

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, a_size);
        a.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as f64);

        for res_size in [1, 2, 6, 11] {
            let mut res_0: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);
            let mut res_1: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);

            // Set d to garbage
            source.fill_bytes(res_0.data_mut());
            source.fill_bytes(res_1.data_mut());

            // Reference
            for i in 0..cols {
                module.vec_znx_idft_apply(&mut res_1, i, &a, i, scratch.borrow());
                vec_znx_idft_apply::<_, _, _, ZnxArithmeticRef, ReimArithmeticRef, ReimConvRef, ReimIFFTRef>(
                    &table, &mut res_0, i, &a, i,
                );
            }

            assert_eq!(res_0.raw(), res_1.raw());
        }
    }
}

pub fn test_vec_znx_idft_apply_tmpa<B>(module: &Module<B>)
where
    Module<B>: VecZnxIdftApplyTmpA<B>,
    B: Backend<ScalarPrep = f64, ScalarBig = i64> + VecZnxDftAllocBytesImpl<B> + VecZnxBigAllocBytesImpl<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let table: ReimIFFTTable<f64> = ReimIFFTTable::<f64>::new(module.n() >> 1);

    for a_size in [1, 2, 6, 11] {
        let mut a_0: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, a_size);
        a_0.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as f64);

        let mut a_1: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, a_size);
        a_1.raw_mut().copy_from_slice(a_0.raw());

        for res_size in [1, 2, 6, 11] {
            let mut res_0: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);
            let mut res_1: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);

            // Set d to garbage
            source.fill_bytes(res_0.data_mut());
            source.fill_bytes(res_1.data_mut());

            for i in 0..cols {
                vec_znx_idft_apply_tmpa::<_, _, _, ZnxArithmeticRef, ReimConvRef, ReimIFFTRef>(
                    &table, &mut res_0, i, &mut a_0, i,
                );
                module.vec_znx_idft_apply_tmpa(&mut res_1, i, &mut a_1, i);
            }

            assert_eq!(res_0.raw(), res_1.raw());
        }
    }
}

pub fn test_vec_znx_idft_apply_consume<B>(module: &Module<B>)
where
    Module<B>: VecZnxIdftApplyConsume<B>,
    B: Backend<ScalarPrep = f64, ScalarBig = i64> + VecZnxDftAllocBytesImpl<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let table: ReimIFFTTable<f64> = ReimIFFTTable::<f64>::new(module.n() >> 1);

    for a_size in [1, 2, 6, 11] {
        let mut a_0: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, a_size);
        a_0.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as f64);

        let mut a_1: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, a_size);
        a_1.raw_mut().copy_from_slice(a_0.raw());

        let res_0: VecZnxBig<Vec<u8>, B> = vec_znx_idft_apply_consume::<_, _, ReimConvRef, ReimIFFTRef>(&table, a_0);
        let res_1: VecZnxBig<Vec<u8>, B> = module.vec_znx_idft_apply_consume(a_1);

        assert_eq!(res_0.raw(), res_1.raw());
    }
}

pub fn bench_vec_znx_dft_apply<B>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxDftApply<B> + ModuleNew<B>,
    B: Backend + VecZnxDftAllocBytesImpl<B>,
{
    let group_name: String = format!("vec_znx_dft_apply::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxDftApply<B> + ModuleNew<B>,
        B: Backend + VecZnxDftAllocBytesImpl<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut res: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, size);
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        source.fill_bytes(res.data_mut());
        source.fill_bytes(a.data_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_dft_apply(1, 0, &mut res, i, &a, i);
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

pub fn bench_vec_znx_idft_apply<B>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxIdftApply<B> + ModuleNew<B> + VecZnxIdftApplyTmpBytes,
    B: Backend + VecZnxDftAllocBytesImpl<B> + VecZnxBigAllocBytesImpl<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_idft_apply::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxIdftApply<B> + ModuleNew<B> + VecZnxIdftApplyTmpBytes,
        B: Backend + VecZnxDftAllocBytesImpl<B> + VecZnxBigAllocBytesImpl<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut res: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, size);
        let mut a: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, size);
        source.fill_bytes(res.data_mut());
        source.fill_bytes(a.data_mut());

        let mut scratch = ScratchOwned::alloc(module.vec_znx_idft_apply_tmp_bytes());

        move || {
            for i in 0..cols {
                module.vec_znx_idft_apply(&mut res, i, &a, i, scratch.borrow());
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

pub fn bench_vec_znx_idft_apply_tmpa<B>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxIdftApplyTmpA<B> + ModuleNew<B>,
    B: Backend + VecZnxDftAllocBytesImpl<B> + VecZnxBigAllocBytesImpl<B>,
{
    let group_name: String = format!("vec_znx_idft_apply_tmpa::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxIdftApplyTmpA<B> + ModuleNew<B>,
        B: Backend + VecZnxDftAllocBytesImpl<B> + VecZnxBigAllocBytesImpl<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut res: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, size);
        let mut a: VecZnxDft<Vec<u8>, B> = VecZnxDft::alloc(module.n(), cols, size);
        source.fill_bytes(res.data_mut());
        source.fill_bytes(a.data_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_idft_apply_tmpa(&mut res, i, &mut a, i);
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
