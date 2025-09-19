use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigAlloc,
        VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftAlloc, VecZnxDftAllocBytes,
        VecZnxDftApply, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA, VecZnxNormalize, VecZnxNormalizeInplace,
        VecZnxNormalizeTmpBytes, VecZnxRotateInplace, VecZnxSub, VecZnxSubABInplace, VmpApplyDftToDft, VmpApplyDftToDftAdd,
        VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare,
    },
    layouts::{Backend, Module, ScalarZnx, ScalarZnxToMut, ScratchOwned, ZnxViewMut},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl, VecZnxBigAllocBytesImpl, VecZnxDftAllocBytesImpl,
    },
    source::Source,
};

use crate::{
    encryption::SIGMA,
    layouts::{
        GGSWCiphertext, GLWESecret,
        prepared::{GGSWCiphertextPrepared, GLWESecretPrepared, PrepareAlloc},
    },
    noise::noise_ggsw_product,
};

#[allow(clippy::too_many_arguments)]
pub fn test_ggsw_external_product<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
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
        + VecZnxRotateInplace<B>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VmpPrepare<B>
        + VecZnxBigAlloc<B>
        + VecZnxDftAlloc<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpA<B>,
    B: Backend
        + TakeVecZnxDftImpl<B>
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
    let basek: usize = 12;
    let k_in: usize = 60;
    let digits: usize = k_in.div_ceil(basek);
    (1..3).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ggsw: usize = k_in + basek * di;

            let k_out: usize = k_in; // Better capture noise.

            let n: usize = module.n();
            let rows: usize = k_in.div_ceil(basek * di);
            let rows_in: usize = k_in.div_euclid(basek * di);
            let digits_in: usize = 1;

            let mut ct_ggsw_lhs_in: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_in, rows_in, digits_in, rank);
            let mut ct_ggsw_lhs_out: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_out, rows_in, digits_in, rank);
            let mut ct_ggsw_rhs: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_ggsw, rows, di, rank);
            let mut pt_ggsw_lhs: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
            let mut pt_ggsw_rhs: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            pt_ggsw_lhs.fill_ternary_prob(0, 0.5, &mut source_xs);

            let k: usize = 1;

            pt_ggsw_rhs.to_mut().raw_mut()[k] = 1; //X^{k}

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
                GGSWCiphertext::encrypt_sk_scratch_space(module, basek, k_ggsw, rank)
                    | GGSWCiphertext::external_product_scratch_space(module, basek, k_out, basek, k_in, basek, k_ggsw, di, rank),
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
                scratch.borrow(),
            );

            ct_ggsw_lhs_in.encrypt_sk(
                module,
                &pt_ggsw_lhs,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let ct_rhs_prepared: GGSWCiphertextPrepared<Vec<u8>, B> = ct_ggsw_rhs.prepare_alloc(module, scratch.borrow());

            ct_ggsw_lhs_out.external_product(module, &ct_ggsw_lhs_in, &ct_rhs_prepared, scratch.borrow());

            module.vec_znx_rotate_inplace(
                k as i64,
                &mut pt_ggsw_lhs.as_vec_znx_mut(),
                0,
                scratch.borrow(),
            );

            let var_gct_err_lhs: f64 = SIGMA * SIGMA;
            let var_gct_err_rhs: f64 = 0f64;

            let var_msg: f64 = 1f64 / n as f64; // X^{k}
            let var_a0_err: f64 = SIGMA * SIGMA;
            let var_a1_err: f64 = 1f64 / 12f64;

            let max_noise = |_col_j: usize| -> f64 {
                noise_ggsw_product(
                    n as f64,
                    basek * di,
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

            ct_ggsw_lhs_out.assert_noise(module, &sk_prepared, &pt_ggsw_lhs, max_noise);
        });
    });
}

#[allow(clippy::too_many_arguments)]
pub fn test_ggsw_external_product_inplace<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
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
        + VecZnxRotateInplace<B>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VmpPrepare<B>
        + VecZnxBigAlloc<B>
        + VecZnxDftAlloc<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpA<B>,
    B: Backend
        + TakeVecZnxDftImpl<B>
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
    let basek: usize = 12;
    let k_ct: usize = 60;
    let digits: usize = k_ct.div_ceil(basek);
    (1..3).for_each(|rank| {
        (1..digits).for_each(|di| {
            let k_ggsw: usize = k_ct + basek * di;

            let n: usize = module.n();
            let rows: usize = k_ct.div_ceil(di * basek);
            let rows_in: usize = k_ct.div_euclid(basek * di);
            let digits_in: usize = 1;

            let mut ct_ggsw_lhs: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_ct, rows_in, digits_in, rank);
            let mut ct_ggsw_rhs: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_ggsw, rows, di, rank);

            let mut pt_ggsw_lhs: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
            let mut pt_ggsw_rhs: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            pt_ggsw_lhs.fill_ternary_prob(0, 0.5, &mut source_xs);

            let k: usize = 1;

            pt_ggsw_rhs.to_mut().raw_mut()[k] = 1; //X^{k}

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
                GGSWCiphertext::encrypt_sk_scratch_space(module, basek, k_ggsw, rank)
                    | GGSWCiphertext::external_product_inplace_scratch_space(module, basek, k_ct, basek, k_ggsw, di, rank),
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
                scratch.borrow(),
            );

            ct_ggsw_lhs.encrypt_sk(
                module,
                &pt_ggsw_lhs,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let ct_rhs_prepared: GGSWCiphertextPrepared<Vec<u8>, B> = ct_ggsw_rhs.prepare_alloc(module, scratch.borrow());

            ct_ggsw_lhs.external_product_inplace(module, &ct_rhs_prepared, scratch.borrow());

            module.vec_znx_rotate_inplace(
                k as i64,
                &mut pt_ggsw_lhs.as_vec_znx_mut(),
                0,
                scratch.borrow(),
            );

            let var_gct_err_lhs: f64 = SIGMA * SIGMA;
            let var_gct_err_rhs: f64 = 0f64;

            let var_msg: f64 = 1f64 / n as f64; // X^{k}
            let var_a0_err: f64 = SIGMA * SIGMA;
            let var_a1_err: f64 = 1f64 / 12f64;

            let max_noise = |_col_j: usize| -> f64 {
                noise_ggsw_product(
                    n as f64,
                    basek * di,
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

            ct_ggsw_lhs.assert_noise(module, &sk_prepared, &pt_ggsw_lhs, max_noise);
        });
    });
}
