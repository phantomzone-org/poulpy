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
            ReimFFTTable, ReimIFFTTable, reim_copy_ref, reim_from_znx_i64_ref, reim_to_znx_i64_inplace_ref, reim_to_znx_i64_ref,
            reim_zero_ref,
        },
        vec_znx_dft::fft64::assert_approx_eq_slice,
        znx::znx_zero_ref,
    },
    source::Source,
};

pub fn vec_znx_dft_apply_ref<R, A, BE>(
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
            reim_from_znx_i64_ref(res.at_mut(res_col, j), a.at(a_col, limb));
            table.execute(res.at_mut(res_col, j));
        }
    }

    (min_steps..res.size()).for_each(|j| {
        reim_zero_ref(res.at_mut(res_col, j));
    });
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2,fma")`);
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub fn vec_znx_dft_apply_avx<R, A, BE>(
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

    use crate::reference::reim::reim_from_znx_i64_bnd50_fma;

    for j in 0..min_steps {
        let limb = offset + j * step;
        if limb < a_size {
            reim_from_znx_i64_bnd50_fma(res.at_mut(res_col, limb), a.at(a_col, j));
            table.execute_avx2_fma(res.at_mut(res_col, limb));
        }
    }

    (min_steps..res.size()).for_each(|j| {
        reim_zero_ref(res.at_mut(res_col, j));
    });
}

pub fn vec_znx_idft_apply_ref<R, A, BE>(table: &ReimIFFTTable<f64>, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = f64, ScalarBig = i64>,
    R: VecZnxBigToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(table.m() << 1, res.n());
        assert_eq!(a.n(), res.n());
    }

    let res_size = res.size();
    let min_size: usize = res_size.min(a.size());

    let divisor: f64 = table.m() as f64;

    for j in 0..min_size {
        let res_slice_f64: &mut [f64] = cast_slice_mut(res.at_mut(res_col, j));
        reim_copy_ref(res_slice_f64, a.at(a_col, j));
        table.execute(res_slice_f64);
        reim_to_znx_i64_inplace_ref(res_slice_f64, divisor);
    }

    for j in min_size..res_size {
        znx_zero_ref(res.at_mut(res_col, j));
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2,fma")`);
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub fn vec_znx_idft_apply_avx<R, A, BE>(table: &ReimIFFTTable<f64>, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = f64, ScalarBig = i64>,
    R: VecZnxBigToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(table.m() << 1, res.n());
        assert_eq!(a.n(), res.n());
    }

    let res_size = res.size();
    let min_size: usize = res_size.min(a.size());

    let divisor: f64 = table.m() as f64;

    use crate::reference::reim::reim_to_znx_i64_inplace_bnd63_avx2_fma;

    for j in 0..min_size {
        let res_slice_f64: &mut [f64] = cast_slice_mut(res.at_mut(res_col, j));
        reim_copy_ref(res_slice_f64, a.at(a_col, j));
        table.execute_avx2_fma(res_slice_f64);
        reim_to_znx_i64_inplace_bnd63_avx2_fma(res_slice_f64, divisor);
    }

    for j in min_size..res_size {
        znx_zero_ref(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_idft_apply_tmpa_ref<R, A, BE>(table: &ReimIFFTTable<f64>, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
where
    BE: Backend<ScalarPrep = f64, ScalarBig = i64>,
    R: VecZnxBigToMut<BE>,
    A: VecZnxDftToMut<BE>,
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
        table.execute(a.at_mut(a_col, j));
        reim_to_znx_i64_ref(res.at_mut(res_col, j), divisor, a.at(a_col, j));
    }

    for j in min_size..res_size {
        znx_zero_ref(res.at_mut(res_col, j));
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2,fma")`);
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub fn vec_znx_idft_apply_tmpa_avx<R, A, BE>(table: &ReimIFFTTable<f64>, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
where
    BE: Backend<ScalarPrep = f64, ScalarBig = i64>,
    R: VecZnxBigToMut<BE>,
    A: VecZnxDftToMut<BE>,
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

    use crate::reference::reim::reim_to_znx_i64_bnd63_avx2_fma;

    for j in 0..min_size {
        table.execute_avx2_fma(a.at_mut(a_col, j));
        reim_to_znx_i64_bnd63_avx2_fma(res.at_mut(res_col, j), divisor, a.at(a_col, j));
    }

    for j in min_size..res_size {
        znx_zero_ref(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_idft_apply_consume_ref<D: Data, BE>(table: &ReimIFFTTable<f64>, mut res: VecZnxDft<D, BE>) -> VecZnxBig<D, BE>
where
    BE: Backend<ScalarPrep = f64, ScalarBig = i64>,
    VecZnxDft<D, BE>: VecZnxDftToMut<BE>,
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
                table.execute(res.at_mut(i, j));
                reim_to_znx_i64_inplace_ref(res.at_mut(i, j), divisor);
            }
        }
    }

    res.into_big()
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2,fma")`);
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub fn vec_znx_idft_apply_consume_avx<D: Data, BE>(table: &ReimIFFTTable<f64>, mut res: VecZnxDft<D, BE>) -> VecZnxBig<D, BE>
where
    BE: Backend<ScalarPrep = f64, ScalarBig = i64>,
    VecZnxDft<D, BE>: VecZnxDftToMut<BE>,
{
    {
        let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
        use crate::reference::reim::reim_to_znx_i64_inplace_bnd63_avx2_fma;

        #[cfg(debug_assertions)]
        {
            assert_eq!(table.m() << 1, res.n());
        }

        let divisor: f64 = table.m() as f64;

        for i in 0..res.cols() {
            for j in 0..res.size() {
                table.execute_avx2_fma(res.at_mut(i, j));
                reim_to_znx_i64_inplace_bnd63_avx2_fma(res.at_mut(i, j), divisor);
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
                vec_znx_dft_apply_ref(&table, 1, 0, &mut res_0, i, &a, i);
            }
            assert_approx_eq_slice(res_0.raw(), res_1.raw(), 1e-10);

            for i in 0..cols {
                module.vec_znx_dft_apply(1, 1, &mut res_1, i, &a, i);
                vec_znx_dft_apply_ref(&table, 1, 1, &mut res_0, i, &a, i);
            }
            assert_approx_eq_slice(res_0.raw(), res_1.raw(), 1e-10);

            for i in 0..cols {
                module.vec_znx_dft_apply(2, 1, &mut res_1, i, &a, i);
                vec_znx_dft_apply_ref(&table, 2, 1, &mut res_0, i, &a, i);
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
                vec_znx_idft_apply_ref(&table, &mut res_0, i, &a, i);
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
                vec_znx_idft_apply_tmpa_ref(&table, &mut res_0, i, &mut a_0, i);
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

        let res_0: VecZnxBig<Vec<u8>, B> = vec_znx_idft_apply_consume_ref(&table, a_0);
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
