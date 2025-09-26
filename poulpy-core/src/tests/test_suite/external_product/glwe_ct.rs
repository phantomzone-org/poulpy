use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace,
        VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxFillUniform, VecZnxIdftApplyConsume,
        VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotateInplace, VecZnxSub, VecZnxSubInplace,
        VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare,
    },
    layouts::{Backend, Module, ScalarZnx, ScratchOwned, ZnxViewMut},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl,
    },
    source::Source,
};

use crate::{
    encryption::SIGMA,
    layouts::{
        GGSWCiphertext, GGSWCiphertextLayout, GLWECiphertext, GLWECiphertextLayout, GLWEPlaintext, GLWESecret,
        prepared::{GGSWCiphertextPrepared, GLWESecretPrepared, PrepareAlloc},
    },
    noise::noise_ggsw_product,
};

#[allow(clippy::too_many_arguments)]
pub fn test_glwe_external_product<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
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
        + VecZnxNormalizeTmpBytes
        + VecZnxAddScalarInplace
        + VecZnxRotateInplace<B>
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>,
    B: Backend
        + TakeVecZnxDftImpl<B>
        + TakeVecZnxBigImpl<B>
        + TakeSvpPPolImpl<B>
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxImpl<B>,
{
    let base2k: usize = 12;
    let k_in: usize = 45;
    let digits: usize = k_in.div_ceil(base2k);
    for rank in 1_usize..3 {
        for di in 1..digits + 1 {
            let k_ggsw: usize = k_in + base2k * di;
            let k_out: usize = k_ggsw; // Better capture noise

            let n: usize = module.n();
            let rows: usize = k_in.div_ceil(base2k * digits);

            let glwe_in_infos: GLWECiphertextLayout = GLWECiphertextLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_in.into(),
                rank: rank.into(),
            };

            let glwe_out_infos: GLWECiphertextLayout = GLWECiphertextLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_out.into(),
                rank: rank.into(),
            };

            let ggsw_apply_infos: GGSWCiphertextLayout = GGSWCiphertextLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_ggsw.into(),
                rows: rows.into(),
                digits: di.into(),
                rank: rank.into(),
            };

            let mut ggsw_apply: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(&ggsw_apply_infos);
            let mut glwe_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&glwe_in_infos);
            let mut glwe_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&glwe_out_infos);
            let mut pt_ggsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
            let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&glwe_in_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            // Random input plaintext
            module.vec_znx_fill_uniform(base2k, &mut pt_want.data, 0, &mut source_xa);

            pt_want.data.at_mut(0, 0)[1] = 1;

            let k: usize = 1;

            pt_ggsw.raw_mut()[k] = 1; // X^{k}

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
                GGSWCiphertext::encrypt_sk_scratch_space(module, &ggsw_apply_infos)
                    | GLWECiphertext::encrypt_sk_scratch_space(module, &glwe_in_infos)
                    | GLWECiphertext::external_product_scratch_space(module, &glwe_out_infos, &glwe_in_infos, &ggsw_apply_infos),
            );

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(n.into(), rank.into());
            sk.fill_ternary_prob(0.5, &mut source_xs);
            let sk_prepared: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

            ggsw_apply.encrypt_sk(
                module,
                &pt_ggsw,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            glwe_in.encrypt_sk(
                module,
                &pt_want,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let ct_ggsw_prepared: GGSWCiphertextPrepared<Vec<u8>, B> = ggsw_apply.prepare_alloc(module, scratch.borrow());

            glwe_out.external_product(module, &glwe_in, &ct_ggsw_prepared, scratch.borrow());

            module.vec_znx_rotate_inplace(k as i64, &mut pt_want.data, 0, scratch.borrow());

            let var_gct_err_lhs: f64 = SIGMA * SIGMA;
            let var_gct_err_rhs: f64 = 0f64;

            let var_msg: f64 = 1f64 / n as f64; // X^{k}
            let var_a0_err: f64 = SIGMA * SIGMA;
            let var_a1_err: f64 = 1f64 / 12f64;

            let max_noise: f64 = noise_ggsw_product(
                n as f64,
                base2k * digits,
                0.5,
                var_msg,
                var_a0_err,
                var_a1_err,
                var_gct_err_lhs,
                var_gct_err_rhs,
                rank as f64,
                k_in,
                k_ggsw,
            );

            glwe_out.assert_noise(module, &sk_prepared, &pt_want, max_noise + 0.5);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_glwe_external_product_inplace<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
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
        + VecZnxNormalizeTmpBytes
        + VecZnxAddScalarInplace
        + VecZnxRotateInplace<B>
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>,
    B: Backend
        + TakeVecZnxDftImpl<B>
        + TakeVecZnxBigImpl<B>
        + TakeSvpPPolImpl<B>
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxImpl<B>,
{
    let base2k: usize = 12;
    let k_out: usize = 60;
    let digits: usize = k_out.div_ceil(base2k);
    for rank in 1_usize..3 {
        for di in 1..digits + 1 {
            let k_ggsw: usize = k_out + base2k * di;

            let n: usize = module.n();
            let rows: usize = k_out.div_ceil(base2k * digits);

            let glwe_out_infos: GLWECiphertextLayout = GLWECiphertextLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_out.into(),
                rank: rank.into(),
            };

            let ggsw_apply_infos: GGSWCiphertextLayout = GGSWCiphertextLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_ggsw.into(),
                rows: rows.into(),
                digits: di.into(),
                rank: rank.into(),
            };

            let mut ggsw_apply: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(&ggsw_apply_infos);
            let mut glwe_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&glwe_out_infos);
            let mut pt_ggsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
            let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&glwe_out_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            // Random input plaintext
            module.vec_znx_fill_uniform(base2k, &mut pt_want.data, 0, &mut source_xa);

            pt_want.data.at_mut(0, 0)[1] = 1;

            let k: usize = 1;

            pt_ggsw.raw_mut()[k] = 1; // X^{k}

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
                GGSWCiphertext::encrypt_sk_scratch_space(module, &ggsw_apply_infos)
                    | GLWECiphertext::encrypt_sk_scratch_space(module, &glwe_out_infos)
                    | GLWECiphertext::external_product_inplace_scratch_space(module, &glwe_out_infos, &ggsw_apply_infos),
            );

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(n.into(), rank.into());
            sk.fill_ternary_prob(0.5, &mut source_xs);
            let sk_prepared: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

            ggsw_apply.encrypt_sk(
                module,
                &pt_ggsw,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            glwe_out.encrypt_sk(
                module,
                &pt_want,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let ct_ggsw_prepared: GGSWCiphertextPrepared<Vec<u8>, B> = ggsw_apply.prepare_alloc(module, scratch.borrow());

            glwe_out.external_product_inplace(module, &ct_ggsw_prepared, scratch.borrow());

            module.vec_znx_rotate_inplace(k as i64, &mut pt_want.data, 0, scratch.borrow());

            let var_gct_err_lhs: f64 = SIGMA * SIGMA;
            let var_gct_err_rhs: f64 = 0f64;

            let var_msg: f64 = 1f64 / n as f64; // X^{k}
            let var_a0_err: f64 = SIGMA * SIGMA;
            let var_a1_err: f64 = 1f64 / 12f64;

            let max_noise: f64 = noise_ggsw_product(
                n as f64,
                base2k * digits,
                0.5,
                var_msg,
                var_a0_err,
                var_a1_err,
                var_gct_err_lhs,
                var_gct_err_rhs,
                rank as f64,
                k_out,
                k_ggsw,
            );

            glwe_out.assert_noise(module, &sk_prepared, &pt_want, max_noise + 0.5);
        }
    }
}
