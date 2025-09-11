use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace,
        VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxCopy, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxFillUniform,
        VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotateInplace, VecZnxSub,
        VecZnxSubABInplace, VecZnxSubScalarInplace, VecZnxSwitchRing, VmpApplyDftToDft, VmpApplyDftToDftAdd,
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
        GGLWESwitchingKey, GGSWCiphertext, GLWESecret,
        prepared::{GGSWCiphertextPrepared, GLWESecretPrepared, PrepareAlloc},
    },
    noise::noise_ggsw_product,
};

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_switching_key_external_product<B>(
    module: &Module<B>,
    basek: usize,
    k_out: usize,
    k_in: usize,
    k_ggsw: usize,
    digits: usize,
    rank_in: usize,
    rank_out: usize,
) where
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
        + VecZnxSwitchRing<B>
        + VecZnxAddScalarInplace
        + VecZnxSubScalarInplace
        + VecZnxCopy
        + VmpPMatAlloc<B>
        + VecZnxRotateInplace<B>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VmpPrepare<B>,
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
    let n: usize = module.n();
    let rows: usize = k_in.div_ceil(basek * digits);
    let digits_in: usize = 1;

    let mut ct_gglwe_in: GGLWESwitchingKey<Vec<u8>> =
        GGLWESwitchingKey::alloc(n, basek, k_in, rows, digits_in, rank_in, rank_out);
    let mut ct_gglwe_out: GGLWESwitchingKey<Vec<u8>> =
        GGLWESwitchingKey::alloc(n, basek, k_out, rows, digits_in, rank_in, rank_out);
    let mut ct_rgsw: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_ggsw, rows, digits, rank_out);

    let mut pt_rgsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GGLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k_in, rank_in, rank_out)
            | GGLWESwitchingKey::external_product_scratch_space(module, basek, k_out, k_in, k_ggsw, digits, rank_out)
            | GGSWCiphertext::encrypt_sk_scratch_space(module, basek, k_ggsw, rank_out),
    );

    let r: usize = 1;

    pt_rgsw.to_mut().raw_mut()[r] = 1; // X^{r}

    let var_xs: f64 = 0.5;

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank_in);
    sk_in.fill_ternary_prob(var_xs, &mut source_xs);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank_out);
    sk_out.fill_ternary_prob(var_xs, &mut source_xs);
    let sk_out_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_out.prepare_alloc(module, scratch.borrow());

    // gglwe_{s1}(s0) = s0 -> s1
    ct_gglwe_in.encrypt_sk(
        module,
        &sk_in,
        &sk_out,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    ct_rgsw.encrypt_sk(
        module,
        &pt_rgsw,
        &sk_out_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let ct_rgsw_prepared: GGSWCiphertextPrepared<Vec<u8>, B> = ct_rgsw.prepare_alloc(module, scratch.borrow());

    // gglwe_(m) (x) RGSW_(X^k) = gglwe_(m * X^k)
    ct_gglwe_out.external_product(module, &ct_gglwe_in, &ct_rgsw_prepared, scratch.borrow());

    (0..rank_in).for_each(|i| {
        module.vec_znx_rotate_inplace(
            r as i64,
            &mut sk_in.data.as_vec_znx_mut(),
            i,
            scratch.borrow(),
        ); // * X^{r}
    });

    let var_gct_err_lhs: f64 = SIGMA * SIGMA;
    let var_gct_err_rhs: f64 = 0f64;

    let var_msg: f64 = 1f64 / n as f64; // X^{k}
    let var_a0_err: f64 = SIGMA * SIGMA;
    let var_a1_err: f64 = 1f64 / 12f64;

    let max_noise: f64 = noise_ggsw_product(
        n as f64,
        basek * digits,
        var_xs,
        var_msg,
        var_a0_err,
        var_a1_err,
        var_gct_err_lhs,
        var_gct_err_rhs,
        rank_out as f64,
        k_in,
        k_ggsw,
    );

    ct_gglwe_out
        .key
        .assert_noise(module, &sk_out_prepared, &sk_in.data, max_noise + 0.5);
}

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_switching_key_external_product_inplace<B>(
    module: &Module<B>,
    basek: usize,
    k_ct: usize,
    k_ggsw: usize,
    digits: usize,
    rank_in: usize,
    rank_out: usize,
) where
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
        + VecZnxSwitchRing<B>
        + VecZnxAddScalarInplace
        + VecZnxSubScalarInplace
        + VecZnxCopy
        + VmpPMatAlloc<B>
        + VecZnxRotateInplace<B>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VmpPrepare<B>,
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
    let n: usize = module.n();
    let rows: usize = k_ct.div_ceil(basek * digits);

    let digits_in: usize = 1;

    let mut ct_gglwe: GGLWESwitchingKey<Vec<u8>> = GGLWESwitchingKey::alloc(n, basek, k_ct, rows, digits_in, rank_in, rank_out);
    let mut ct_rgsw: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_ggsw, rows, digits, rank_out);

    let mut pt_rgsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GGLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k_ct, rank_in, rank_out)
            | GGLWESwitchingKey::external_product_inplace_scratch_space(module, basek, k_ct, k_ggsw, digits, rank_out)
            | GGSWCiphertext::encrypt_sk_scratch_space(module, basek, k_ggsw, rank_out),
    );

    let r: usize = 1;

    pt_rgsw.to_mut().raw_mut()[r] = 1; // X^{r}

    let var_xs: f64 = 0.5;

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank_in);
    sk_in.fill_ternary_prob(var_xs, &mut source_xs);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank_out);
    sk_out.fill_ternary_prob(var_xs, &mut source_xs);
    let sk_out_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_out.prepare_alloc(module, scratch.borrow());

    // gglwe_{s1}(s0) = s0 -> s1
    ct_gglwe.encrypt_sk(
        module,
        &sk_in,
        &sk_out,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    ct_rgsw.encrypt_sk(
        module,
        &pt_rgsw,
        &sk_out_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let ct_rgsw_prepared: GGSWCiphertextPrepared<Vec<u8>, B> = ct_rgsw.prepare_alloc(module, scratch.borrow());

    // gglwe_(m) (x) RGSW_(X^k) = gglwe_(m * X^k)
    ct_gglwe.external_product_inplace(module, &ct_rgsw_prepared, scratch.borrow());

    (0..rank_in).for_each(|i| {
        module.vec_znx_rotate_inplace(
            r as i64,
            &mut sk_in.data.as_vec_znx_mut(),
            i,
            scratch.borrow(),
        ); // * X^{r}
    });

    let var_gct_err_lhs: f64 = SIGMA * SIGMA;
    let var_gct_err_rhs: f64 = 0f64;

    let var_msg: f64 = 1f64 / n as f64; // X^{k}
    let var_a0_err: f64 = SIGMA * SIGMA;
    let var_a1_err: f64 = 1f64 / 12f64;

    let max_noise: f64 = noise_ggsw_product(
        n as f64,
        basek * digits,
        var_xs,
        var_msg,
        var_a0_err,
        var_a1_err,
        var_gct_err_lhs,
        var_gct_err_rhs,
        rank_out as f64,
        k_ct,
        k_ggsw,
    );

    ct_gglwe
        .key
        .assert_noise(module, &sk_out_prepared, &sk_in.data, max_noise + 0.5);
}
