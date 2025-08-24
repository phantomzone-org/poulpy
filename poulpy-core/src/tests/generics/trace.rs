use std::collections::HashMap;

use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare, VecZnxAddInplace,
        VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphism, VecZnxBigAddInplace, VecZnxBigAddSmallInplace,
        VecZnxBigAllocBytes, VecZnxBigAutomorphismInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxBigSubSmallBInplace, VecZnxCopy, VecZnxDftAllocBytes, VecZnxDftFromVecZnx, VecZnxDftToVecZnxBigConsume,
        VecZnxFillUniform, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotateInplace,
        VecZnxRshInplace, VecZnxSub, VecZnxSubABInplace, VecZnxSwithcDegree, VmpApply, VmpApplyAdd, VmpApplyTmpBytes,
        VmpPMatAlloc, VmpPrepare,
    },
    layouts::{Backend, Module, ScratchOwned, ZnxView, ZnxViewMut},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl,
    },
    source::Source,
};

use crate::{
    encryption::SIGMA,
    layouts::{
        GGLWEAutomorphismKey, GLWECiphertext, GLWEPlaintext, GLWESecret, Infos,
        prepared::{GGLWEAutomorphismKeyPrepared, GLWESecretPrepared, PrepareAlloc},
    },
    noise::var_noise_gglwe_product,
};

pub fn test_glwe_trace_inplace<B>(module: &Module<B>, basek: usize, k: usize, rank: usize)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxAutomorphism
        + VecZnxBigAutomorphismInplace<B>
        + VecZnxBigSubSmallBInplace<B>
        + VecZnxRshInplace
        + VecZnxRotateInplace
        + VecZnxBigNormalize<B>
        + VecZnxDftFromVecZnx<B>
        + SvpApplyInplace<B>
        + VecZnxDftToVecZnxBigConsume<B>
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
        + VecZnxNormalizeTmpBytes
        + VecZnxAddScalarInplace
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + VmpApplyTmpBytes
        + VmpApply<B>
        + VmpApplyAdd<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxSwithcDegree
        + VecZnxCopy,
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
    let n: usize = module.n();
    let k_autokey: usize = k + basek;

    let digits: usize = 1;
    let rows: usize = k.div_ceil(basek * digits);

    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GLWECiphertext::encrypt_sk_scratch_space(module, basek, ct.k())
            | GLWECiphertext::decrypt_scratch_space(module, basek, ct.k())
            | GGLWEAutomorphismKey::encrypt_sk_scratch_space(module, basek, k_autokey, rank)
            | GLWECiphertext::trace_inplace_scratch_space(module, basek, ct.k(), k_autokey, digits, rank),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_dft: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

    let mut data_want: Vec<i64> = vec![0i64; n];

    data_want
        .iter_mut()
        .for_each(|x| *x = source_xa.next_i64() & 0xFF);

    module.vec_znx_fill_uniform(basek, &mut pt_have.data, 0, k, &mut source_xa);

    ct.encrypt_sk(
        module,
        &pt_have,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut auto_keys: HashMap<i64, GGLWEAutomorphismKeyPrepared<Vec<u8>, B>> = HashMap::new();
    let gal_els: Vec<i64> = GLWECiphertext::trace_galois_elements(module);
    let mut tmp: GGLWEAutomorphismKey<Vec<u8>> = GGLWEAutomorphismKey::alloc(n, basek, k_autokey, rows, digits, rank);
    gal_els.iter().for_each(|gal_el| {
        tmp.encrypt_sk(
            module,
            *gal_el,
            &sk,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );
        let atk_prepared: GGLWEAutomorphismKeyPrepared<Vec<u8>, B> = tmp.prepare_alloc(module, scratch.borrow());
        auto_keys.insert(*gal_el, atk_prepared);
    });

    ct.trace_inplace(module, 0, 5, &auto_keys, scratch.borrow());
    ct.trace_inplace(module, 5, module.log_n(), &auto_keys, scratch.borrow());

    (0..pt_want.size()).for_each(|i| pt_want.data.at_mut(0, i)[0] = pt_have.data.at(0, i)[0]);

    ct.decrypt(module, &mut pt_have, &sk_dft, scratch.borrow());

    module.vec_znx_sub_ab_inplace(&mut pt_want.data, 0, &pt_have.data, 0);
    module.vec_znx_normalize_inplace(basek, &mut pt_want.data, 0, scratch.borrow());

    let noise_have: f64 = pt_want.std().log2();

    let mut noise_want: f64 = var_noise_gglwe_product(
        n as f64,
        basek,
        0.5,
        0.5,
        1.0 / 12.0,
        SIGMA * SIGMA,
        0.0,
        rank as f64,
        k,
        k_autokey,
    );
    noise_want += SIGMA * SIGMA * (-2.0 * (k) as f64).exp2();
    noise_want += n as f64 * 1.0 / 12.0 * 0.5 * rank as f64 * (-2.0 * (k) as f64).exp2();
    noise_want = noise_want.sqrt().log2();

    assert!(
        (noise_have - noise_want).abs() < 1.0,
        "{} > {}",
        noise_have,
        noise_want
    );
}
