use backend::hal::{
    api::{
        MatZnxAlloc, ScalarZnxAlloc, ScalarZnxAllocBytes, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarInplace,
        VecZnxAlloc, VecZnxAllocBytes, VecZnxFillUniform, VecZnxStd, VecZnxSwithcDegree,
    },
    layouts::{Backend, Module, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl,
    },
};
use sampling::source::Source;

use crate::{
    GGLWEExecLayoutFamily, GLWECiphertext, GLWEDecryptFamily, GLWEKeyswitchFamily, GLWEPlaintext, GLWESecret, GLWESecretExec,
    GLWESecretFamily, GLWESwitchingKey, GLWESwitchingKeyEncryptSkFamily, GLWESwitchingKeyExec, Infos,
    noise::log2_std_noise_gglwe_product,
};

pub(crate) trait KeySwitchTestModuleFamily<B: Backend> = GLWESecretFamily<B>
    + GLWESwitchingKeyEncryptSkFamily<B>
    + GLWEKeyswitchFamily<B>
    + GLWEDecryptFamily<B>
    + GGLWEExecLayoutFamily<B>
    + MatZnxAlloc
    + VecZnxAlloc
    + ScalarZnxAlloc
    + ScalarZnxAllocBytes
    + VecZnxAllocBytes
    + VecZnxStd
    + VecZnxSwithcDegree
    + VecZnxAddScalarInplace;

pub(crate) trait KeySwitchTestScratchFamily<B: Backend> = TakeVecZnxDftImpl<B>
    + TakeVecZnxBigImpl<B>
    + TakeSvpPPolImpl<B>
    + ScratchOwnedAllocImpl<B>
    + ScratchOwnedBorrowImpl<B>
    + ScratchAvailableImpl<B>
    + TakeScalarZnxImpl<B>
    + TakeVecZnxImpl<B>;

pub(crate) fn test_keyswitch<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k_out: usize,
    k_in: usize,
    k_ksk: usize,
    digits: usize,
    rank_in: usize,
    rank_out: usize,
    sigma: f64,
) where
    Module<B>: KeySwitchTestModuleFamily<B>,
    B: KeySwitchTestScratchFamily<B>,
{
    let rows: usize = k_in.div_ceil(basek * digits);

    let mut ksk: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc(module, basek, k_ksk, rows, digits, rank_in, rank_out);
    let mut ct_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, basek, k_in, rank_in);
    let mut ct_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, basek, k_out, rank_out);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k_in);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    module.vec_znx_fill_uniform(basek, &mut pt_want.data, 0, k_in, &mut source_xa);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GLWESwitchingKey::encrypt_sk_scratch_space(module, basek, ksk.k(), rank_in, rank_out)
            | GLWECiphertext::encrypt_sk_scratch_space(module, basek, ct_in.k())
            | GLWECiphertext::keyswitch_scratch_space(
                module,
                basek,
                ct_out.k(),
                ct_in.k(),
                ksk.k(),
                digits,
                rank_in,
                rank_out,
            ),
    );

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank_in);
    sk_in.fill_ternary_prob(0.5, &mut source_xs);
    let sk_in_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk_in);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank_out);
    sk_out.fill_ternary_prob(0.5, &mut source_xs);
    let sk_out_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk_out);

    ksk.encrypt_sk(
        module,
        &sk_in,
        &sk_out,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_in.encrypt_sk(
        module,
        &pt_want,
        &sk_in_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let ksk_exec: GLWESwitchingKeyExec<Vec<u8>, B> = GLWESwitchingKeyExec::from(module, &ksk, scratch.borrow());

    ct_out.keyswitch(module, &ct_in, &ksk_exec, scratch.borrow());

    let max_noise: f64 = log2_std_noise_gglwe_product(
        module.n() as f64,
        basek * digits,
        0.5,
        0.5,
        0f64,
        sigma * sigma,
        0f64,
        rank_in as f64,
        k_in,
        k_ksk,
    );

    ct_out.assert_noise(module, &sk_out_exec, &pt_want, max_noise + 0.5);
}

pub(crate) fn test_keyswitch_inplace<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k_ct: usize,
    k_ksk: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) where
    Module<B>: KeySwitchTestModuleFamily<B>,
    B: KeySwitchTestScratchFamily<B>,
{
    let rows: usize = k_ct.div_ceil(basek * digits);

    let mut ksk: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc(module, basek, k_ksk, rows, digits, rank, rank);
    let mut ct_glwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, basek, k_ct, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k_ct);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    module.vec_znx_fill_uniform(basek, &mut pt_want.data, 0, k_ct, &mut source_xa);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GLWESwitchingKey::encrypt_sk_scratch_space(module, basek, ksk.k(), rank, rank)
            | GLWECiphertext::encrypt_sk_scratch_space(module, basek, ct_glwe.k())
            | GLWECiphertext::keyswitch_inplace_scratch_space(module, basek, ct_glwe.k(), ksk.k(), digits, rank),
    );

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
    sk_in.fill_ternary_prob(0.5, &mut source_xs);
    let sk_in_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk_in);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
    sk_out.fill_ternary_prob(0.5, &mut source_xs);
    let sk_out_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk_out);

    ksk.encrypt_sk(
        module,
        &sk_in,
        &sk_out,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_glwe.encrypt_sk(
        module,
        &pt_want,
        &sk_in_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let ksk_exec: GLWESwitchingKeyExec<Vec<u8>, B> = GLWESwitchingKeyExec::from(module, &ksk, scratch.borrow());

    ct_glwe.keyswitch_inplace(module, &ksk_exec, scratch.borrow());

    let max_noise: f64 = log2_std_noise_gglwe_product(
        module.n() as f64,
        basek * digits,
        0.5,
        0.5,
        0f64,
        sigma * sigma,
        0f64,
        rank as f64,
        k_ct,
        k_ksk,
    );

    ct_glwe.assert_noise(module, &sk_out_exec, &pt_want, max_noise + 0.5);
}
