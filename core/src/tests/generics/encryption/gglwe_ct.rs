use backend::hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarInplace, VecZnxCopy, VecZnxStd, VecZnxSubScalarInplace,
        VecZnxSwithcDegree, VmpPMatAlloc, VmpPMatPrepare,
    },
    layouts::{Backend, Module, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl, VecZnxBigAllocBytesImpl, VecZnxDftAllocBytesImpl,
    },
};
use sampling::source::Source;

use crate::{
    layouts::{GLWESecret, GGLWESwitchingKey, compressed::GGLWESwitchingKeyCompressed, prepared::GLWESecretExec},
    trait_families::{Decompress, GLWEDecryptFamily},
};

use crate::trait_families::{GGLWEEncryptSkFamily, GLWESecretExecModuleFamily};

pub fn test_gglwe_switching_key_encrypt_sk<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k_ksk: usize,
    digits: usize,
    rank_in: usize,
    rank_out: usize,
    sigma: f64,
) where
    Module<B>: GGLWEEncryptSkFamily<B>
        + GLWESecretExecModuleFamily<B>
        + GLWEDecryptFamily<B>
        + VecZnxSwithcDegree
        + VecZnxAddScalarInplace
        + VecZnxStd
        + VecZnxSubScalarInplace
        + VecZnxCopy
        + VmpPMatAlloc<B>
        + VmpPMatPrepare<B>,
    B: TakeVecZnxDftImpl<B>
        + TakeVecZnxBigImpl<B>
        + TakeSvpPPolImpl<B>
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
    let rows: usize = (k_ksk - digits * basek) / (digits * basek);

    let mut ksk: GGLWESwitchingKey<Vec<u8>> = GGLWESwitchingKey::alloc(n, basek, k_ksk, rows, digits, rank_in, rank_out);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GGLWESwitchingKey::encrypt_sk_scratch_space(
        module, n, basek, k_ksk, rank_in, rank_out,
    ));

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank_in);
    sk_in.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank_out);
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

    ksk.key
        .assert_noise(module, &sk_out_exec, &sk_in.data, sigma);
}

pub fn test_gglwe_switching_key_compressed_encrypt_sk<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k_ksk: usize,
    digits: usize,
    rank_in: usize,
    rank_out: usize,
    sigma: f64,
) where
    Module<B>: GGLWEEncryptSkFamily<B>
        + GLWESecretExecModuleFamily<B>
        + GLWEDecryptFamily<B>
        + VecZnxSwithcDegree
        + VecZnxAddScalarInplace
        + VecZnxStd
        + VecZnxSubScalarInplace
        + VecZnxCopy
        + VmpPMatAlloc<B>
        + VmpPMatPrepare<B>,
    B: TakeVecZnxDftImpl<B>
        + TakeVecZnxBigImpl<B>
        + TakeSvpPPolImpl<B>
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
    let rows: usize = (k_ksk - digits * basek) / (digits * basek);

    let mut ksk_compressed: GGLWESwitchingKeyCompressed<Vec<u8>> =
        GGLWESwitchingKeyCompressed::alloc(n, basek, k_ksk, rows, digits, rank_in, rank_out);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GGLWESwitchingKeyCompressed::encrypt_sk_scratch_space(
        module, n, basek, k_ksk, rank_in, rank_out,
    ));

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank_in);
    sk_in.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank_out);
    sk_out.fill_ternary_prob(0.5, &mut source_xs);
    let sk_out_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk_out);

    let seed_xa = [1u8; 32];

    ksk_compressed.encrypt_sk(
        module,
        &sk_in,
        &sk_out,
        seed_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut ksk: GGLWESwitchingKey<Vec<u8>> = GGLWESwitchingKey::alloc(n, basek, k_ksk, rows, digits, rank_in, rank_out);
    ksk.decompress(module, &ksk_compressed);

    ksk.key
        .assert_noise(module, &sk_out_exec, &sk_in.data, sigma);
}
