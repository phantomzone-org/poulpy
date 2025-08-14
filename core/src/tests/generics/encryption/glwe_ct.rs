use backend::hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxCopy, VecZnxDftAlloc, VecZnxFillUniform, VecZnxSubABInplace},
    layouts::{Backend, Module, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl,
    },
};
use sampling::source::Source;

use crate::{
    layouts::{
        GLWECiphertext, GLWEPlaintext, GLWEPublicKey, GLWESecret, Infos,
        compressed::GLWECiphertextCompressed,
        prepared::{GLWEPublicKeyExec, GLWESecretExec},
    },
    operations::GLWEOperations,
    trait_families::Decompress,
};

use crate::trait_families::{GLWEDecryptFamily, GLWEEncryptPkFamily, GLWEEncryptSkFamily, GLWESecretExecModuleFamily};

pub trait EncryptionTestModuleFamily<B: Backend> = GLWEDecryptFamily<B> + GLWESecretExecModuleFamily<B> + GLWEEncryptPkFamily<B>;

pub fn test_glwe_encrypt_sk<B: Backend>(module: &Module<B>, basek: usize, k_ct: usize, k_pt: usize, sigma: f64, rank: usize)
where
    Module<B>: EncryptionTestModuleFamily<B> + GLWEEncryptSkFamily<B>,
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
    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_ct, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_pt);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_pt);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GLWECiphertext::encrypt_sk_scratch_space(module, n, basek, ct.k())
            | GLWECiphertext::decrypt_scratch_space(module, n, basek, ct.k()),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk);

    module.vec_znx_fill_uniform(basek, &mut pt_want.data, 0, k_pt, &mut source_xa);

    ct.encrypt_sk(
        module,
        &pt_want,
        &sk_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct.decrypt(module, &mut pt_have, &sk_exec, scratch.borrow());

    pt_want.sub_inplace_ab(module, &pt_have);

    let noise_have: f64 = pt_want.data.std(basek, 0) * (ct.k() as f64).exp2();
    let noise_want: f64 = sigma;

    assert!(noise_have <= noise_want + 0.2);
}

pub fn test_glwe_compressed_encrypt_sk<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k_ct: usize,
    k_pt: usize,
    sigma: f64,
    rank: usize,
) where
    Module<B>: EncryptionTestModuleFamily<B> + GLWEEncryptSkFamily<B> + VecZnxCopy,
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
    let mut ct_compressed: GLWECiphertextCompressed<Vec<u8>> = GLWECiphertextCompressed::alloc(n, basek, k_ct, rank);

    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_pt);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_pt);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GLWECiphertextCompressed::encrypt_sk_scratch_space(module, n, basek, k_ct)
            | GLWECiphertext::decrypt_scratch_space(module, n, basek, k_ct),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk);

    module.vec_znx_fill_uniform(basek, &mut pt_want.data, 0, k_pt, &mut source_xa);

    let seed_xa: [u8; 32] = [1u8; 32];

    ct_compressed.encrypt_sk(
        module,
        &pt_want,
        &sk_exec,
        seed_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_ct, rank);
    ct.decompress(module, &ct_compressed);

    ct.decrypt(module, &mut pt_have, &sk_exec, scratch.borrow());

    pt_want.sub_inplace_ab(module, &pt_have);

    let noise_have: f64 = pt_want.data.std(basek, 0) * (ct.k() as f64).exp2();
    let noise_want: f64 = sigma;

    assert!(
        noise_have <= noise_want + 0.2,
        "{} <= {}",
        noise_have,
        noise_want + 0.2
    );
}

pub fn test_glwe_encrypt_zero_sk<B: Backend>(module: &Module<B>, basek: usize, k_ct: usize, sigma: f64, rank: usize)
where
    Module<B>: EncryptionTestModuleFamily<B> + GLWEEncryptSkFamily<B>,
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
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_ct);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk);

    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_ct, rank);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GLWECiphertext::decrypt_scratch_space(module, n, basek, k_ct)
            | GLWECiphertext::encrypt_sk_scratch_space(module, n, basek, k_ct),
    );

    ct.encrypt_zero_sk(
        module,
        &sk_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );
    ct.decrypt(module, &mut pt, &sk_exec, scratch.borrow());

    assert!((sigma - pt.data.std(basek, 0) * (k_ct as f64).exp2()) <= 0.2);
}

pub fn test_glwe_encrypt_pk<B: Backend>(module: &Module<B>, basek: usize, k_ct: usize, k_pk: usize, sigma: f64, rank: usize)
where
    Module<B>:
        EncryptionTestModuleFamily<B> + GLWEEncryptSkFamily<B> + VecZnxDftAlloc<B> + VecZnxFillUniform + VecZnxSubABInplace,
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
    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_ct, rank);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_ct);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_ct);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xu: Source = Source::new([0u8; 32]);

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk);

    let mut pk: GLWEPublicKey<Vec<u8>> = GLWEPublicKey::alloc(n, basek, k_pk, rank);
    pk.generate_from_sk(module, &sk_exec, &mut source_xa, &mut source_xe, sigma);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GLWECiphertext::encrypt_sk_scratch_space(module, n, basek, ct.k())
            | GLWECiphertext::decrypt_scratch_space(module, n, basek, ct.k())
            | GLWECiphertext::encrypt_pk_scratch_space(module, n, basek, pk.k()),
    );

    module.vec_znx_fill_uniform(basek, &mut pt_want.data, 0, k_ct, &mut source_xa);

    let pk_exec: GLWEPublicKeyExec<Vec<u8>, B> = GLWEPublicKeyExec::from(module, &pk, scratch.borrow());

    ct.encrypt_pk(
        module,
        &pt_want,
        &pk_exec,
        &mut source_xu,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct.decrypt(module, &mut pt_have, &sk_exec, scratch.borrow());

    pt_want.sub_inplace_ab(module, &pt_have);

    let noise_have: f64 = pt_want.data.std(basek, 0).log2();
    let noise_want: f64 = ((((rank as f64) + 1.0) * n as f64 * 0.5 * sigma * sigma).sqrt()).log2() - (k_ct as f64);

    assert!(
        noise_have <= noise_want + 0.2,
        "{} {}",
        noise_have,
        noise_want
    );
}
