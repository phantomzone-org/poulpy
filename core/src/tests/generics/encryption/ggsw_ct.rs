use backend::hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarInplace, VecZnxCopy, VecZnxSubABInplace, VmpPMatAlloc,
        VmpPMatPrepare,
    },
    layouts::{Backend, Module, ScalarZnx, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl, VecZnxBigAllocBytesImpl, VecZnxDftAllocBytesImpl,
    },
};
use sampling::source::Source;

use crate::{
    layouts::{
        GGSWCiphertext, GLWESecret,
        compressed::GGSWCiphertextCompressed,
        prepared::{GLWESecretPrepared, PrepareAlloc},
    },
    trait_families::{Decompress, GGSWAssertNoiseFamily},
};

use crate::trait_families::{GGSWEncryptSkFamily, GLWESecretExecModuleFamily};

pub fn test_ggsw_encrypt_sk<B: Backend>(module: &Module<B>, basek: usize, k: usize, digits: usize, rank: usize, sigma: f64)
where
    Module<B>: GLWESecretExecModuleFamily<B>
        + GGSWEncryptSkFamily<B>
        + GGSWAssertNoiseFamily<B>
        + VecZnxAddScalarInplace
        + VecZnxSubABInplace
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
    let rows: usize = (k - digits * basek) / (digits * basek);

    let mut ct: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k, rows, digits, rank);

    let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GGSWCiphertext::encrypt_sk_scratch_space(
        module, n, basek, k, rank,
    ));

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_exec: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

    ct.encrypt_sk(
        module,
        &pt_scalar,
        &sk_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let noise_f = |_col_i: usize| -(k as f64) + sigma.log2() + 0.5;

    ct.assert_noise(module, &sk_exec, &pt_scalar, &noise_f);
}

pub fn test_ggsw_compressed_encrypt_sk<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) where
    Module<B>: GLWESecretExecModuleFamily<B>
        + GGSWEncryptSkFamily<B>
        + GGSWAssertNoiseFamily<B>
        + VecZnxAddScalarInplace
        + VecZnxSubABInplace
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
    let rows: usize = (k - digits * basek) / (digits * basek);

    let mut ct_compressed: GGSWCiphertextCompressed<Vec<u8>> = GGSWCiphertextCompressed::alloc(n, basek, k, rows, digits, rank);

    let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GGSWCiphertextCompressed::encrypt_sk_scratch_space(
        module, n, basek, k, rank,
    ));

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_exec: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

    let seed_xa: [u8; 32] = [1u8; 32];

    ct_compressed.encrypt_sk(
        module,
        &pt_scalar,
        &sk_exec,
        seed_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let noise_f = |_col_i: usize| -(k as f64) + sigma.log2() + 0.5;

    let mut ct: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k, rows, digits, rank);
    ct.decompress(module, &ct_compressed);

    ct.assert_noise(module, &sk_exec, &pt_scalar, &noise_f);
}
