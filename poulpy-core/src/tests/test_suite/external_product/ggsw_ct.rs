use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigAlloc,
        VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftAlloc, VecZnxDftAllocBytes,
        VecZnxDftApply, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA, VecZnxNormalize, VecZnxNormalizeInplace,
        VecZnxNormalizeTmpBytes, VecZnxRotateInplace, VecZnxSub, VecZnxSubInplace, VmpApplyDftToDft, VmpApplyDftToDftAdd,
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
        GGSWCiphertext, GGSWCiphertextLayout, GLWESecret,
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
        + VecZnxSubInplace
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
    let base2k: usize = 12;
    let k_in: usize = 60;
    let digits: usize = k_in.div_ceil(base2k);
    for rank in 1_usize..3 {
        for di in 1..digits + 1 {
            let k_apply: usize = k_in + base2k * di;

            let k_out: usize = k_in; // Better capture noise.

            let n: usize = module.n();
            let rows: usize = k_in.div_ceil(base2k * di);
            let rows_in: usize = k_in.div_euclid(base2k * di);
            let digits_in: usize = 1;

            let ggsw_in_infos: GGSWCiphertextLayout = GGSWCiphertextLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_in.into(),
                rows: rows_in.into(),
                digits: digits_in.into(),
                rank: rank.into(),
            };

            let ggsw_out_infos: GGSWCiphertextLayout = GGSWCiphertextLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_out.into(),
                rows: rows_in.into(),
                digits: digits_in.into(),
                rank: rank.into(),
            };

            let ggsw_apply_infos: GGSWCiphertextLayout = GGSWCiphertextLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_apply.into(),
                rows: rows.into(),
                digits: di.into(),
                rank: rank.into(),
            };

            let mut ggsw_in: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(&ggsw_in_infos);
            let mut ggsw_out: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(&ggsw_out_infos);
            let mut ggsw_apply: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(&ggsw_apply_infos);
            let mut pt_in: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
            let mut pt_apply: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            pt_in.fill_ternary_prob(0, 0.5, &mut source_xs);

            let k: usize = 1;

            pt_apply.to_mut().raw_mut()[k] = 1; //X^{k}

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
                GGSWCiphertext::encrypt_sk_scratch_space(module, &ggsw_apply_infos)
                    | GGSWCiphertext::encrypt_sk_scratch_space(module, &ggsw_in_infos)
                    | GGSWCiphertext::external_product_scratch_space(module, &ggsw_out_infos, &ggsw_in_infos, &ggsw_apply_infos),
            );

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(n.into(), rank.into());
            sk.fill_ternary_prob(0.5, &mut source_xs);
            let sk_prepared: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

            ggsw_apply.encrypt_sk(
                module,
                &pt_apply,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            ggsw_in.encrypt_sk(
                module,
                &pt_in,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let ct_rhs_prepared: GGSWCiphertextPrepared<Vec<u8>, B> = ggsw_apply.prepare_alloc(module, scratch.borrow());

            ggsw_out.external_product(module, &ggsw_in, &ct_rhs_prepared, scratch.borrow());

            module.vec_znx_rotate_inplace(k as i64, &mut pt_in.as_vec_znx_mut(), 0, scratch.borrow());

            let var_gct_err_lhs: f64 = SIGMA * SIGMA;
            let var_gct_err_rhs: f64 = 0f64;

            let var_msg: f64 = 1f64 / n as f64; // X^{k}
            let var_a0_err: f64 = SIGMA * SIGMA;
            let var_a1_err: f64 = 1f64 / 12f64;

            let max_noise = |_col_j: usize| -> f64 {
                noise_ggsw_product(
                    n as f64,
                    base2k * di,
                    0.5,
                    var_msg,
                    var_a0_err,
                    var_a1_err,
                    var_gct_err_lhs,
                    var_gct_err_rhs,
                    rank as f64,
                    k_in,
                    k_apply,
                ) + 0.5
            };

            ggsw_out.assert_noise(module, &sk_prepared, &pt_in, max_noise);
        }
    }
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
        + VecZnxSubInplace
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
    let base2k: usize = 12;
    let k_out: usize = 60;
    let digits: usize = k_out.div_ceil(base2k);
    for rank in 1_usize..3 {
        for di in 1..digits + 1 {
            let k_apply: usize = k_out + base2k * di;

            let n: usize = module.n();
            let rows: usize = k_out.div_ceil(di * base2k);
            let rows_in: usize = k_out.div_euclid(base2k * di);
            let digits_in: usize = 1;

            let ggsw_out_infos: GGSWCiphertextLayout = GGSWCiphertextLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_out.into(),
                rows: rows_in.into(),
                digits: digits_in.into(),
                rank: rank.into(),
            };

            let ggsw_apply_infos: GGSWCiphertextLayout = GGSWCiphertextLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_apply.into(),
                rows: rows.into(),
                digits: di.into(),
                rank: rank.into(),
            };

            let mut ggsw_out: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(&ggsw_out_infos);
            let mut ggsw_apply: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(&ggsw_apply_infos);

            let mut pt_in: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
            let mut pt_apply: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            pt_in.fill_ternary_prob(0, 0.5, &mut source_xs);

            let k: usize = 1;

            pt_apply.to_mut().raw_mut()[k] = 1; //X^{k}

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
                GGSWCiphertext::encrypt_sk_scratch_space(module, &ggsw_apply_infos)
                    | GGSWCiphertext::encrypt_sk_scratch_space(module, &ggsw_out_infos)
                    | GGSWCiphertext::external_product_inplace_scratch_space(module, &ggsw_out_infos, &ggsw_apply_infos),
            );

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(n.into(), rank.into());
            sk.fill_ternary_prob(0.5, &mut source_xs);
            let sk_prepared: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

            ggsw_apply.encrypt_sk(
                module,
                &pt_apply,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            ggsw_out.encrypt_sk(
                module,
                &pt_in,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let ct_rhs_prepared: GGSWCiphertextPrepared<Vec<u8>, B> = ggsw_apply.prepare_alloc(module, scratch.borrow());

            ggsw_out.external_product_inplace(module, &ct_rhs_prepared, scratch.borrow());

            module.vec_znx_rotate_inplace(k as i64, &mut pt_in.as_vec_znx_mut(), 0, scratch.borrow());

            let var_gct_err_lhs: f64 = SIGMA * SIGMA;
            let var_gct_err_rhs: f64 = 0f64;

            let var_msg: f64 = 1f64 / n as f64; // X^{k}
            let var_a0_err: f64 = SIGMA * SIGMA;
            let var_a1_err: f64 = 1f64 / 12f64;

            let max_noise = |_col_j: usize| -> f64 {
                noise_ggsw_product(
                    n as f64,
                    base2k * di,
                    0.5,
                    var_msg,
                    var_a0_err,
                    var_a1_err,
                    var_gct_err_lhs,
                    var_gct_err_rhs,
                    rank as f64,
                    k_out,
                    k_apply,
                ) + 0.5
            };

            ggsw_out.assert_noise(module, &sk_prepared, &pt_in, max_noise);
        }
    }
}
