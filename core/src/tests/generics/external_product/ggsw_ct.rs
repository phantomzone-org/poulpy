use backend::hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare, VecZnxAddInplace,
        VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigAlloc,
        VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftAlloc, VecZnxDftAllocBytes,
        VecZnxDftFromVecZnx, VecZnxDftToVecZnxBigConsume, VecZnxDftToVecZnxBigTmpA, VecZnxFillUniform, VecZnxNormalize,
        VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotateInplace, VecZnxSub, VecZnxSubABInplace, VmpApply,
        VmpApplyAdd, VmpApplyTmpBytes, VmpPMatAlloc, VmpPrepare, ZnxViewMut,
    },
    layouts::{Backend, Module, ScalarZnx, ScalarZnxToMut, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl, VecZnxBigAllocBytesImpl, VecZnxDftAllocBytesImpl,
    },
};
use sampling::source::Source;

use crate::{
    layouts::{
        GGSWCiphertext, GLWESecret,
        prepared::{GGSWCiphertextPrepared, GLWESecretPrepared, PrepareAlloc},
    },
    noise::noise_ggsw_product,
};

pub fn test_ggsw_external_product<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k_in: usize,
    k_out: usize,
    k_ggsw: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftFromVecZnx<B>
        + SvpApplyInplace<B>
        + VecZnxDftToVecZnxBigConsume<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxFillUniform
        + VecZnxSubABInplace
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxAddNormal
        + VecZnxNormalize<B>
        + VecZnxSub
        + SvpPrepare<B>
        + SvpPPolAllocBytes
        + SvpPPolAlloc<B>
        + VecZnxBigAllocBytes
        + VecZnxBigAddInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxAddScalarInplace
        + VecZnxCopy
        + VmpPMatAlloc<B>
        + VecZnxRotateInplace
        + VmpApplyTmpBytes
        + VmpApply<B>
        + VmpApplyAdd<B>
        + VmpPrepare<B>
        + VecZnxBigAlloc<B>
        + VecZnxDftAlloc<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftToVecZnxBigTmpA<B>,
    B: TakeVecZnxDftImpl<B>
        + TakeVecZnxBigImpl<B>
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxImpl<B>
        + VecZnxDftAllocBytesImpl<B>
        + VecZnxBigAllocBytesImpl<B>
        + TakeSvpPPolImpl<B>,
{
    let n: usize = module.n();
    let rows: usize = k_in.div_ceil(basek * digits);
    let rows_in: usize = k_in.div_euclid(basek * digits);
    let digits_in: usize = 1;

    let mut ct_ggsw_lhs_in: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_in, rows_in, digits_in, rank);
    let mut ct_ggsw_lhs_out: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_out, rows_in, digits_in, rank);
    let mut ct_ggsw_rhs: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_ggsw, rows, digits, rank);
    let mut pt_ggsw_lhs: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
    let mut pt_ggsw_rhs: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    pt_ggsw_lhs.fill_ternary_prob(0, 0.5, &mut source_xs);

    let k: usize = 1;

    pt_ggsw_rhs.to_mut().raw_mut()[k] = 1; //X^{k}

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GGSWCiphertext::encrypt_sk_scratch_space(module, n, basek, k_ggsw, rank)
            | GGSWCiphertext::external_product_scratch_space(module, n, basek, k_out, k_in, k_ggsw, digits, rank),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_prepared: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

    ct_ggsw_rhs.encrypt_sk(
        module,
        &pt_ggsw_rhs,
        &sk_prepared,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_ggsw_lhs_in.encrypt_sk(
        module,
        &pt_ggsw_lhs,
        &sk_prepared,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let ct_rhs_prepared: GGSWCiphertextPrepared<Vec<u8>, B> = ct_ggsw_rhs.prepare_alloc(module, scratch.borrow());

    ct_ggsw_lhs_out.external_product(module, &ct_ggsw_lhs_in, &ct_rhs_prepared, scratch.borrow());

    module.vec_znx_rotate_inplace(k as i64, &mut pt_ggsw_lhs.as_vec_znx_mut(), 0);

    let var_gct_err_lhs: f64 = sigma * sigma;
    let var_gct_err_rhs: f64 = 0f64;

    let var_msg: f64 = 1f64 / n as f64; // X^{k}
    let var_a0_err: f64 = sigma * sigma;
    let var_a1_err: f64 = 1f64 / 12f64;

    let max_noise = |_col_j: usize| -> f64 {
        noise_ggsw_product(
            n as f64,
            basek * digits,
            0.5,
            var_msg,
            var_a0_err,
            var_a1_err,
            var_gct_err_lhs,
            var_gct_err_rhs,
            rank as f64,
            k_in,
            k_ggsw,
        ) + 0.5
    };

    ct_ggsw_lhs_out.assert_noise(module, &sk_prepared, &pt_ggsw_lhs, &max_noise);
}

pub fn test_ggsw_external_product_inplace<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k_ct: usize,
    k_ggsw: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftFromVecZnx<B>
        + SvpApplyInplace<B>
        + VecZnxDftToVecZnxBigConsume<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxFillUniform
        + VecZnxSubABInplace
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxAddNormal
        + VecZnxNormalize<B>
        + VecZnxSub
        + SvpPrepare<B>
        + SvpPPolAllocBytes
        + SvpPPolAlloc<B>
        + VecZnxBigAllocBytes
        + VecZnxBigAddInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxAddScalarInplace
        + VecZnxCopy
        + VmpPMatAlloc<B>
        + VecZnxRotateInplace
        + VmpApplyTmpBytes
        + VmpApply<B>
        + VmpApplyAdd<B>
        + VmpPrepare<B>
        + VecZnxBigAlloc<B>
        + VecZnxDftAlloc<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftToVecZnxBigTmpA<B>,
    B: TakeVecZnxDftImpl<B>
        + TakeVecZnxBigImpl<B>
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxImpl<B>
        + VecZnxDftAllocBytesImpl<B>
        + VecZnxBigAllocBytesImpl<B>
        + TakeSvpPPolImpl<B>,
{
    let n: usize = module.n();
    let rows: usize = k_ct.div_ceil(digits * basek);
    let rows_in: usize = k_ct.div_euclid(basek * digits);
    let digits_in: usize = 1;

    let mut ct_ggsw_lhs: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_ct, rows_in, digits_in, rank);
    let mut ct_ggsw_rhs: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_ggsw, rows, digits, rank);

    let mut pt_ggsw_lhs: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
    let mut pt_ggsw_rhs: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    pt_ggsw_lhs.fill_ternary_prob(0, 0.5, &mut source_xs);

    let k: usize = 1;

    pt_ggsw_rhs.to_mut().raw_mut()[k] = 1; //X^{k}

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GGSWCiphertext::encrypt_sk_scratch_space(module, n, basek, k_ggsw, rank)
            | GGSWCiphertext::external_product_inplace_scratch_space(module, n, basek, k_ct, k_ggsw, digits, rank),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_prepared: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

    ct_ggsw_rhs.encrypt_sk(
        module,
        &pt_ggsw_rhs,
        &sk_prepared,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_ggsw_lhs.encrypt_sk(
        module,
        &pt_ggsw_lhs,
        &sk_prepared,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let ct_rhs_prepared: GGSWCiphertextPrepared<Vec<u8>, B> = ct_ggsw_rhs.prepare_alloc(module, scratch.borrow());

    ct_ggsw_lhs.external_product_inplace(module, &ct_rhs_prepared, scratch.borrow());

    module.vec_znx_rotate_inplace(k as i64, &mut pt_ggsw_lhs.as_vec_znx_mut(), 0);

    let var_gct_err_lhs: f64 = sigma * sigma;
    let var_gct_err_rhs: f64 = 0f64;

    let var_msg: f64 = 1f64 / n as f64; // X^{k}
    let var_a0_err: f64 = sigma * sigma;
    let var_a1_err: f64 = 1f64 / 12f64;

    let max_noise = |_col_j: usize| -> f64 {
        noise_ggsw_product(
            n as f64,
            basek * digits,
            0.5,
            var_msg,
            var_a0_err,
            var_a1_err,
            var_gct_err_lhs,
            var_gct_err_rhs,
            rank as f64,
            k_ct,
            k_ggsw,
        ) + 0.5
    };

    ct_ggsw_lhs.assert_noise(module, &sk_prepared, &pt_ggsw_lhs, &max_noise);
}
