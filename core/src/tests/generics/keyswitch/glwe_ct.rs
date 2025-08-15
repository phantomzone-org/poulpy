use backend::hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarInplace, VecZnxFillUniform, VecZnxSwithcDegree, VmpPMatAlloc,
        VmpPMatPrepare,
    },
    layouts::{Backend, Module, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl,
    },
};
use sampling::source::Source;

use crate::{
    layouts::{
        GGLWESwitchingKey, GLWECiphertext, GLWEPlaintext, GLWESecret, Infos,
        prepared::{GGLWESwitchingKeyPrepared, GLWESecretPrepared, PrepareAlloc},
    },
    noise::log2_std_noise_gglwe_product,
    trait_families::{GLWEDecryptFamily, GLWEKeyswitchFamily},
};

use crate::trait_families::{GGLWESwitchingKeyEncryptSkFamily, GLWESecretExecModuleFamily};

pub fn test_glwe_keyswitch<B: Backend>(
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
    Module<B>: GGLWESwitchingKeyEncryptSkFamily<B>
        + GLWESecretExecModuleFamily<B>
        + GLWEKeyswitchFamily<B>
        + GLWEDecryptFamily<B>
        + VecZnxSwithcDegree
        + VecZnxAddScalarInplace
        + VmpPMatAlloc<B>
        + VmpPMatPrepare<B>,
    B: TakeVecZnxDftImpl<B>
        + TakeVecZnxBigImpl<B>
        + TakeSvpPPolImpl<B>
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxImpl<B>,
{
    let n = module.n();
    let rows: usize = k_in.div_ceil(basek * digits);

    let mut ksk: GGLWESwitchingKey<Vec<u8>> = GGLWESwitchingKey::alloc(n, basek, k_ksk, rows, digits, rank_in, rank_out);
    let mut ct_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_in, rank_in);
    let mut ct_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_out, rank_out);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_in);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    module.vec_znx_fill_uniform(basek, &mut pt_want.data, 0, k_in, &mut source_xa);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GGLWESwitchingKey::encrypt_sk_scratch_space(module, n, basek, ksk.k(), rank_in, rank_out)
            | GLWECiphertext::encrypt_sk_scratch_space(module, n, basek, ct_in.k())
            | GLWECiphertext::keyswitch_scratch_space(
                module,
                n,
                basek,
                ct_out.k(),
                ct_in.k(),
                ksk.k(),
                digits,
                rank_in,
                rank_out,
            ),
    );

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank_in);
    sk_in.fill_ternary_prob(0.5, &mut source_xs);
    let sk_in_exec: GLWESecretPrepared<Vec<u8>, B> = sk_in.prepare_alloc(module, scratch.borrow());

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank_out);
    sk_out.fill_ternary_prob(0.5, &mut source_xs);
    let sk_out_exec: GLWESecretPrepared<Vec<u8>, B> = sk_out.prepare_alloc(module, scratch.borrow());

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

    let ksk_exec: GGLWESwitchingKeyPrepared<Vec<u8>, B> = ksk.prepare_alloc(module, scratch.borrow());

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

pub fn test_glwe_keyswitch_inplace<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k_ct: usize,
    k_ksk: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) where
    Module<B>: GGLWESwitchingKeyEncryptSkFamily<B>
        + GLWESecretExecModuleFamily<B>
        + GLWEKeyswitchFamily<B>
        + GLWEDecryptFamily<B>
        + VecZnxSwithcDegree
        + VecZnxAddScalarInplace
        + VmpPMatAlloc<B>
        + VmpPMatPrepare<B>,
    B: TakeVecZnxDftImpl<B>
        + TakeVecZnxBigImpl<B>
        + TakeSvpPPolImpl<B>
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxImpl<B>,
{
    let n: usize = module.n();
    let rows: usize = k_ct.div_ceil(basek * digits);

    let mut ksk: GGLWESwitchingKey<Vec<u8>> = GGLWESwitchingKey::alloc(n, basek, k_ksk, rows, digits, rank, rank);
    let mut ct_glwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_ct, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_ct);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    module.vec_znx_fill_uniform(basek, &mut pt_want.data, 0, k_ct, &mut source_xa);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GGLWESwitchingKey::encrypt_sk_scratch_space(module, n, basek, ksk.k(), rank, rank)
            | GLWECiphertext::encrypt_sk_scratch_space(module, n, basek, ct_glwe.k())
            | GLWECiphertext::keyswitch_inplace_scratch_space(module, n, basek, ct_glwe.k(), ksk.k(), digits, rank),
    );

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk_in.fill_ternary_prob(0.5, &mut source_xs);
    let sk_in_exec: GLWESecretPrepared<Vec<u8>, B> = sk_in.prepare_alloc(module, scratch.borrow());

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk_out.fill_ternary_prob(0.5, &mut source_xs);
    let sk_out_exec: GLWESecretPrepared<Vec<u8>, B> = sk_out.prepare_alloc(module, scratch.borrow());

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

    let ksk_exec: GGLWESwitchingKeyPrepared<Vec<u8>, B> = ksk.prepare_alloc(module, scratch.borrow());

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
