use std::hint::black_box;

use criterion::Criterion;
use poulpy_core::{
    GLWETensoring,
    layouts::{GLWE, GLWEInfos, GLWETensor, LWEInfos},
};
use poulpy_hal::{
    api::{
        CnvPVecAlloc, Convolution, ModuleNew, ScratchArenaTakeBasic, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigNormalize,
        VecZnxIdftApplyConsume, VecZnxSubInplaceBackend,
    },
    layouts::{
        Backend, CnvPVecLToBackendRef, CnvPVecRToBackendRef, HostDataMut, Module, ScratchOwned, VecZnx,
        VecZnxReborrowBackendMut, VecZnxToBackendRef, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut,
        vec_znx_big_backend_ref_from_mut,
    },
};

fn vec_znx_copy<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    let mut res = res.to_mut();
    let a = a.to_ref();
    let min_size = res.size().min(a.size());
    for j in 0..min_size {
        res.at_mut(res_col, j).copy_from_slice(a.at(a_col, j));
    }
    for j in min_size..res.size() {
        res.at_mut(res_col, j).fill(0);
    }
}

#[inline]
fn msb_mask_bottom_limb(base2k: usize, k: usize) -> i64 {
    match k % base2k {
        0 => !0i64,
        r => (!0i64) << (base2k - r),
    }
}

#[inline]
fn normalize_input_limb_bound_with_offset(
    full_size: usize,
    res_size: usize,
    res_base2k: usize,
    in_base2k: usize,
    res_offset: i64,
) -> usize {
    let mut offset_bits = res_offset % in_base2k as i64;
    if res_offset < 0 && offset_bits != 0 {
        offset_bits += in_base2k as i64;
    }

    full_size.min((res_size * res_base2k + offset_bits as usize).div_ceil(in_base2k))
}

pub fn bench_glwe_tensor_apply<BE: Backend<OwnedBuf = Vec<u8>>>(glwe_infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWETensoring<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
{
    let n: usize = glwe_infos.n().into();
    let module = Module::<BE>::new(n as u64);

    let a = GLWE::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let b = GLWE::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let mut tensor = GLWETensor::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_tensor_apply_tmp_bytes(&tensor, &a, &b));

    let group_name = format!("glwe_tensor_apply::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_tensor_apply(
                0,
                &mut tensor,
                &a,
                a.max_k().as_usize(),
                &b,
                b.max_k().as_usize(),
                &mut scratch.borrow(),
            );
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_tensor_prepare_left<BE: Backend<OwnedBuf = Vec<u8>>>(
    glwe_infos: &impl GLWEInfos,
    c: &mut Criterion,
    label: &str,
) where
    Module<BE>: ModuleNew<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: AsRef<[u8]> + AsMut<[u8]> + Sync,
{
    let n: usize = glwe_infos.n().into();
    let cols: usize = (glwe_infos.rank() + 1).into();
    let module = Module::<BE>::new(n as u64);

    let a = GLWE::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let a_mask = msb_mask_bottom_limb(glwe_infos.base2k().as_usize(), a.max_k().as_usize());
    let mut a_prep = module.cnv_pvec_left_alloc(cols, a.size());
    let mut scratch = ScratchOwned::<BE>::alloc(module.cnv_prepare_left_tmp_bytes(a.size(), a.size()));

    let group_name = format!("glwe_tensor_prepare_left::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.cnv_prepare_left(
                &mut a_prep,
                &<VecZnx<Vec<u8>> as VecZnxToBackendRef<BE>>::to_backend_ref(a.data()),
                a_mask,
                &mut scratch.borrow(),
            );
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_tensor_prepare_right<BE: Backend<OwnedBuf = Vec<u8>>>(
    glwe_infos: &impl GLWEInfos,
    c: &mut Criterion,
    label: &str,
) where
    Module<BE>: ModuleNew<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: AsRef<[u8]> + AsMut<[u8]> + Sync,
{
    let n: usize = glwe_infos.n().into();
    let cols: usize = (glwe_infos.rank() + 1).into();
    let module = Module::<BE>::new(n as u64);

    let b = GLWE::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let b_mask = msb_mask_bottom_limb(glwe_infos.base2k().as_usize(), b.max_k().as_usize());
    let mut b_prep = module.cnv_pvec_right_alloc(cols, b.size());
    let mut scratch = ScratchOwned::<BE>::alloc(module.cnv_prepare_right_tmp_bytes(b.size(), b.size()));

    let group_name = format!("glwe_tensor_prepare_right::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.cnv_prepare_right(
                &mut b_prep,
                &<VecZnx<Vec<u8>> as VecZnxToBackendRef<BE>>::to_backend_ref(b.data()),
                b_mask,
                &mut scratch.borrow(),
            );
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_tensor_diag_lane<BE: Backend<OwnedBuf = Vec<u8>>>(glwe_infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE>
        + GLWETensoring<BE>
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigNormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'x> BE::BufRef<'x>: AsRef<[u8]> + Send,
{
    let n: usize = glwe_infos.n().into();
    let cols: usize = (glwe_infos.rank() + 1).into();
    let module = Module::<BE>::new(n as u64);

    let a = GLWE::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let b = GLWE::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let tensor = GLWETensor::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let base2k = glwe_infos.base2k().as_usize();
    let (cnv_offset_hi, cnv_offset_lo) = (0, -(base2k as i64));
    let diag_dft_size = normalize_input_limb_bound_with_offset(
        a.size() + b.size() - cnv_offset_hi,
        tensor.size(),
        base2k,
        base2k,
        cnv_offset_lo,
    );

    let a_mask = msb_mask_bottom_limb(base2k, a.max_k().as_usize());
    let b_mask = msb_mask_bottom_limb(base2k, b.max_k().as_usize());
    let mut a_prep = module.cnv_pvec_left_alloc(cols, a.size());
    let mut b_prep = module.cnv_pvec_right_alloc(cols, b.size());
    let mut prep_scratch = ScratchOwned::<BE>::alloc(
        module
            .cnv_prepare_left_tmp_bytes(a.size(), a.size())
            .max(module.cnv_prepare_right_tmp_bytes(b.size(), b.size())),
    );
    module.cnv_prepare_left(
        &mut a_prep,
        &<VecZnx<Vec<u8>> as VecZnxToBackendRef<BE>>::to_backend_ref(a.data()),
        a_mask,
        &mut prep_scratch.borrow(),
    );
    module.cnv_prepare_right(
        &mut b_prep,
        &<VecZnx<Vec<u8>> as VecZnxToBackendRef<BE>>::to_backend_ref(b.data()),
        b_mask,
        &mut prep_scratch.borrow(),
    );

    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_tensor_apply_tmp_bytes(&tensor, &a, &b));

    let group_name = format!("glwe_tensor_diag_lane::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            let scratch = scratch.borrow();
            let (mut res_dft, mut scratch) = scratch.take_vec_znx_dft(&module, 1, diag_dft_size);
            module.cnv_apply_dft(
                cnv_offset_hi,
                &mut res_dft,
                0,
                &a_prep.to_backend_ref(),
                0,
                &b_prep.to_backend_ref(),
                0,
                &mut scratch,
            );
            let res_big = module.vec_znx_idft_apply_consume(res_dft);
            let (mut tmp, mut scratch) = scratch.take_vec_znx(n, 1, tensor.size());
            module.vec_znx_big_normalize(
                &mut tmp,
                base2k,
                cnv_offset_lo,
                0,
                &vec_znx_big_backend_ref_from_mut(&res_big),
                base2k,
                0,
                &mut scratch,
            );
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_tensor_pairwise_lane<BE: Backend<OwnedBuf = Vec<u8>>>(
    glwe_infos: &impl GLWEInfos,
    c: &mut Criterion,
    label: &str,
) where
    Module<BE>: ModuleNew<BE>
        + GLWETensoring<BE>
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxSubInplaceBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'x> BE::BufRef<'x>: AsRef<[u8]> + Send,
{
    let n: usize = glwe_infos.n().into();
    let cols: usize = (glwe_infos.rank() + 1).into();
    if cols < 2 {
        return;
    }

    let module = Module::<BE>::new(n as u64);

    let a = GLWE::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let b = GLWE::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let tensor = GLWETensor::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let base2k = glwe_infos.base2k().as_usize();
    let (cnv_offset_hi, cnv_offset_lo) = (0, -(base2k as i64));
    let pairwise_dft_size = normalize_input_limb_bound_with_offset(
        a.size() + b.size() - cnv_offset_hi,
        tensor.size(),
        base2k,
        base2k,
        cnv_offset_lo,
    );

    let a_mask = msb_mask_bottom_limb(base2k, a.max_k().as_usize());
    let b_mask = msb_mask_bottom_limb(base2k, b.max_k().as_usize());
    let mut a_prep = module.cnv_pvec_left_alloc(cols, a.size());
    let mut b_prep = module.cnv_pvec_right_alloc(cols, b.size());
    let mut prep_scratch = ScratchOwned::<BE>::alloc(
        module
            .cnv_prepare_left_tmp_bytes(a.size(), a.size())
            .max(module.cnv_prepare_right_tmp_bytes(b.size(), b.size())),
    );
    module.cnv_prepare_left(
        &mut a_prep,
        &<VecZnx<Vec<u8>> as VecZnxToBackendRef<BE>>::to_backend_ref(a.data()),
        a_mask,
        &mut prep_scratch.borrow(),
    );
    module.cnv_prepare_right(
        &mut b_prep,
        &<VecZnx<Vec<u8>> as VecZnxToBackendRef<BE>>::to_backend_ref(b.data()),
        b_mask,
        &mut prep_scratch.borrow(),
    );

    let mut diag_terms = VecZnx::alloc(n, cols, tensor.size());
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_tensor_apply_tmp_bytes(&tensor, &a, &b));

    {
        for i in 0..cols {
            let scratch = scratch.borrow();
            let (mut res_dft, mut scratch) = scratch.take_vec_znx_dft(&module, 1, pairwise_dft_size);
            module.cnv_apply_dft(
                cnv_offset_hi,
                &mut res_dft,
                0,
                &a_prep.to_backend_ref(),
                i,
                &b_prep.to_backend_ref(),
                i,
                &mut scratch,
            );
            let res_big = module.vec_znx_idft_apply_consume(res_dft);
            let (mut tmp, mut scratch) = scratch.take_vec_znx(n, 1, tensor.size());
            module.vec_znx_big_normalize(
                &mut tmp,
                base2k,
                cnv_offset_lo,
                0,
                &vec_znx_big_backend_ref_from_mut(&res_big),
                base2k,
                0,
                &mut scratch,
            );
            vec_znx_copy(&mut diag_terms, i, &tmp, 0);
        }
    }

    let group_name = format!("glwe_tensor_pairwise_lane::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            let scratch = scratch.borrow();
            let (mut res_dft, mut scratch) = scratch.take_vec_znx_dft(&module, 1, pairwise_dft_size);
            module.cnv_pairwise_apply_dft(
                cnv_offset_hi,
                &mut res_dft,
                0,
                &a_prep.to_backend_ref(),
                &b_prep.to_backend_ref(),
                0,
                1,
                &mut scratch,
            );
            let res_big = module.vec_znx_idft_apply_consume(res_dft);
            let (mut tmp, mut scratch) = scratch.take_vec_znx(n, 1, tensor.size());
            module.vec_znx_big_normalize(
                &mut tmp,
                base2k,
                cnv_offset_lo,
                0,
                &vec_znx_big_backend_ref_from_mut(&res_big),
                base2k,
                0,
                &mut scratch,
            );
            let mut tmp_mut = <VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendMut<BE>>::reborrow_backend_mut(&mut tmp);
            let diag_terms_ref =
                <VecZnx<BE::OwnedBuf> as poulpy_hal::layouts::VecZnxToBackendRef<BE>>::to_backend_ref(&diag_terms);
            module.vec_znx_sub_inplace_backend(&mut tmp_mut, 0, &diag_terms_ref, 0);
            module.vec_znx_sub_inplace_backend(&mut tmp_mut, 0, &diag_terms_ref, 1);
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_tensor_square_apply<BE: Backend<OwnedBuf = Vec<u8>>>(
    glwe_infos: &impl GLWEInfos,
    c: &mut Criterion,
    label: &str,
) where
    Module<BE>: ModuleNew<BE> + GLWETensoring<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
{
    let n: usize = glwe_infos.n().into();
    let module = Module::<BE>::new(n as u64);

    let a = GLWE::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let mut tensor = GLWETensor::<Vec<u8>>::alloc_from_infos(glwe_infos);
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_tensor_square_apply_tmp_bytes(&tensor, &a));

    let group_name = format!("glwe_tensor_square_apply::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_tensor_square_apply(0, &mut tensor, &a, a.max_k().as_usize(), &mut scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}
