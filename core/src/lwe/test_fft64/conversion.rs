use backend::{
    hal::{
        api::{
            MatZnxAlloc, ModuleNew, ScalarZnxAlloc, ScalarZnxAllocBytes, ScratchOwnedAlloc, ScratchOwnedBorrow,
            VecZnxAddScalarInplace, VecZnxAlloc, VecZnxAllocBytes, VecZnxAutomorphismInplace, VecZnxEncodeCoeffsi64,
            VecZnxSwithcDegree, ZnxView,
        },
        layouts::{Backend, Module, ScratchOwned},
        oep::{
            ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
            TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl,
        },
    },
    implementation::cpu_avx::FFT64,
};
use sampling::source::Source;

use crate::{
    GGLWEEncryptSkFamily, GGLWEExecLayoutFamily, GLWECiphertext, GLWEDecryptFamily, GLWEKeyswitchFamily, GLWEPlaintext,
    GLWESecret, GLWESecretExec, Infos, LWECiphertext, LWESecret,
    lwe::{
        LWEPlaintext,
        keyswtich::{
            GLWEToLWESwitchingKey, GLWEToLWESwitchingKeyExec, LWESwitchingKey, LWESwitchingKeyExec, LWEToGLWESwitchingKey,
            LWEToGLWESwitchingKeyExec,
        },
    },
};

#[test]
fn lwe_to_glwe() {
    let log_n: usize = 5;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    test_lwe_to_glwe(&module)
}

pub(crate) trait LWETestModuleFamily<B: Backend> = GGLWEEncryptSkFamily<B>
    + GLWEDecryptFamily<B>
    + VecZnxSwithcDegree
    + VecZnxAddScalarInplace
    + VecZnxAlloc
    + GGLWEExecLayoutFamily<B>
    + GLWEKeyswitchFamily<B>
    + ScalarZnxAllocBytes
    + VecZnxAllocBytes
    + ScalarZnxAlloc
    + VecZnxEncodeCoeffsi64
    + MatZnxAlloc
    + VecZnxAutomorphismInplace;

pub(crate) trait LWETestScratchFamily<B: Backend> = TakeScalarZnxImpl<B>
    + TakeVecZnxDftImpl<B>
    + ScratchAvailableImpl<B>
    + TakeVecZnxImpl<B>
    + TakeVecZnxBigImpl<B>
    + TakeSvpPPolImpl<B>
    + ScratchOwnedAllocImpl<B>
    + ScratchOwnedBorrowImpl<B>;

pub(crate) fn test_lwe_to_glwe<B: Backend>(module: &Module<B>)
where
    Module<B>: LWETestModuleFamily<B>,
    B: LWETestScratchFamily<B>,
{
    let basek: usize = 17;
    let sigma: f64 = 3.2;

    let rank: usize = 2;

    let n_lwe: usize = 22;
    let k_lwe_ct: usize = 2 * basek;
    let k_lwe_pt: usize = 8;

    let k_glwe_ct: usize = 3 * basek;

    let k_ksk: usize = k_lwe_ct + basek;

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        LWEToGLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k_ksk, rank)
            | GLWECiphertext::from_lwe_scratch_space(module, basek, k_lwe_ct, k_glwe_ct, k_ksk, rank)
            | GLWECiphertext::decrypt_scratch_space(module, basek, k_glwe_ct),
    );

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let sk_glwe_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk_glwe);

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
    sk_lwe.fill_ternary_prob(0.5, &mut source_xs);

    let data: i64 = 17;

    let mut lwe_pt: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(basek, k_lwe_pt);
    module.encode_coeff_i64(basek, &mut lwe_pt.data, 0, k_lwe_pt, 0, data, k_lwe_pt);

    let mut lwe_ct: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(n_lwe, basek, k_lwe_ct);
    lwe_ct.encrypt_sk(
        module,
        &lwe_pt,
        &sk_lwe,
        &mut source_xa,
        &mut source_xe,
        sigma,
    );

    let mut ksk: LWEToGLWESwitchingKey<Vec<u8>> = LWEToGLWESwitchingKey::alloc(module, basek, k_ksk, lwe_ct.size(), rank);

    ksk.encrypt_sk(
        module,
        &sk_lwe,
        &sk_glwe,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut glwe_ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, basek, k_glwe_ct, rank);

    let ksk_exec: LWEToGLWESwitchingKeyExec<Vec<u8>, B> = LWEToGLWESwitchingKeyExec::from(module, &ksk, scratch.borrow());

    glwe_ct.from_lwe(module, &lwe_ct, &ksk_exec, scratch.borrow());

    let mut glwe_pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k_glwe_ct);
    glwe_ct.decrypt(module, &mut glwe_pt, &sk_glwe_exec, scratch.borrow());

    assert_eq!(glwe_pt.data.at(0, 0)[0], lwe_pt.data.at(0, 0)[0]);
}

#[test]
fn glwe_to_lwe() {
    let log_n: usize = 5;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    test_glwe_to_lwe(&module)
}

fn test_glwe_to_lwe<B: Backend>(module: &Module<B>)
where
    Module<B>: LWETestModuleFamily<B>,
    B: LWETestScratchFamily<B>,
{
    let basek: usize = 17;
    let sigma: f64 = 3.2;

    let rank: usize = 2;

    let n_lwe: usize = 22;
    let k_lwe_ct: usize = 2 * basek;
    let k_lwe_pt: usize = 8;

    let k_glwe_ct: usize = 3 * basek;

    let k_ksk: usize = k_lwe_ct + basek;

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        LWEToGLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k_ksk, rank)
            | GLWECiphertext::from_lwe_scratch_space(module, basek, k_lwe_ct, k_glwe_ct, k_ksk, rank)
            | GLWECiphertext::decrypt_scratch_space(module, basek, k_glwe_ct),
    );

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let sk_glwe_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk_glwe);

    let mut sk_lwe = LWESecret::alloc(n_lwe);
    sk_lwe.fill_ternary_prob(0.5, &mut source_xs);

    let data: i64 = 17;
    let mut glwe_pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k_glwe_ct);
    module.encode_coeff_i64(basek, &mut glwe_pt.data, 0, k_lwe_pt, 0, data, k_lwe_pt);

    let mut glwe_ct = GLWECiphertext::alloc(module, basek, k_glwe_ct, rank);
    glwe_ct.encrypt_sk(
        module,
        &glwe_pt,
        &sk_glwe_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut ksk: GLWEToLWESwitchingKey<Vec<u8>> = GLWEToLWESwitchingKey::alloc(module, basek, k_ksk, glwe_ct.size(), rank);

    ksk.encrypt_sk(
        module,
        &sk_lwe,
        &sk_glwe,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut lwe_ct: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(n_lwe, basek, k_lwe_ct);

    let ksk_exec: GLWEToLWESwitchingKeyExec<Vec<u8>, B> = GLWEToLWESwitchingKeyExec::from(module, &ksk, scratch.borrow());

    lwe_ct.from_glwe(module, &glwe_ct, &ksk_exec, scratch.borrow());

    let mut lwe_pt: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(basek, k_lwe_ct);
    lwe_ct.decrypt(module, &mut lwe_pt, &sk_lwe);

    assert_eq!(glwe_pt.data.at(0, 0)[0], lwe_pt.data.at(0, 0)[0]);
}

#[test]
fn keyswitch() {
    let log_n: usize = 5;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    test_keyswitch(&module)
}

fn test_keyswitch<B: Backend>(module: &Module<B>)
where
    Module<B>: LWETestModuleFamily<B>,
    B: LWETestScratchFamily<B>,
{
    let basek: usize = 17;
    let sigma: f64 = 3.2;

    let n_lwe_in: usize = 22;
    let n_lwe_out: usize = 30;
    let k_lwe_ct: usize = 2 * basek;
    let k_lwe_pt: usize = 8;

    let k_ksk: usize = k_lwe_ct + basek;

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        LWESwitchingKey::encrypt_sk_scratch_space(module, basek, k_ksk)
            | LWECiphertext::keyswitch_scratch_space(module, basek, k_lwe_ct, k_lwe_ct, k_ksk),
    );

    let mut sk_lwe_in: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe_in);
    sk_lwe_in.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_lwe_out: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe_out);
    sk_lwe_out.fill_ternary_prob(0.5, &mut source_xs);

    let data: i64 = 17;

    let mut lwe_pt_in: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(basek, k_lwe_pt);
    module.encode_coeff_i64(basek, &mut lwe_pt_in.data, 0, k_lwe_pt, 0, data, k_lwe_pt);

    let mut lwe_ct_in: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(n_lwe_in, basek, k_lwe_ct);
    lwe_ct_in.encrypt_sk(
        module,
        &lwe_pt_in,
        &sk_lwe_in,
        &mut source_xa,
        &mut source_xe,
        sigma,
    );

    let mut ksk: LWESwitchingKey<Vec<u8>> = LWESwitchingKey::alloc(module, basek, k_ksk, lwe_ct_in.size());

    ksk.encrypt_sk(
        module,
        &sk_lwe_in,
        &sk_lwe_out,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut lwe_ct_out: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(n_lwe_out, basek, k_lwe_ct);

    let ksk_exec: LWESwitchingKeyExec<Vec<u8>, B> = LWESwitchingKeyExec::from(module, &ksk, scratch.borrow());

    lwe_ct_out.keyswitch(module, &lwe_ct_in, &ksk_exec, scratch.borrow());

    let mut lwe_pt_out: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(basek, k_lwe_ct);
    lwe_ct_out.decrypt(module, &mut lwe_pt_out, &sk_lwe_out);

    assert_eq!(lwe_pt_in.data.at(0, 0)[0], lwe_pt_out.data.at(0, 0)[0]);
}
