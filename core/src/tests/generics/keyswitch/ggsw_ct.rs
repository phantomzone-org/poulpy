use backend::hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarInplace, VecZnxCopy, VecZnxStd, VecZnxSubABInplace,
        VecZnxSwithcDegree, VmpPMatAlloc, VmpPMatPrepare,
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
        GGLWETensorKey, GGSWCiphertext, GLWESecret, GGLWESwitchingKey,
        prepared::{GLWESecretExec, GGLWESwitchingKeyExec, GGLWETensorKeyExec},
    },
    noise::noise_ggsw_keyswitch,
    trait_families::GGSWAssertNoiseFamily,
};

use crate::trait_families::{
    GGLWESwitchingKeyEncryptSkFamily, GGLWETensorKeyEncryptSkFamily, GGSWEncryptSkFamily, GGSWKeySwitchFamily,
    GLWESecretExecModuleFamily,
};

pub fn test_ggsw_keyswitch<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k_out: usize,
    k_in: usize,
    k_ksk: usize,
    k_tsk: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) where
    Module<B>: GLWESecretExecModuleFamily<B>
        + GGSWEncryptSkFamily<B>
        + GGSWAssertNoiseFamily<B>
        + VecZnxAddScalarInplace
        + VecZnxSubABInplace
        + VecZnxStd
        + VecZnxCopy
        + VmpPMatAlloc<B>
        + VmpPMatPrepare<B>
        + GGSWAssertNoiseFamily<B>
        + GGSWKeySwitchFamily<B>
        + GGLWESwitchingKeyEncryptSkFamily<B>
        + GGLWETensorKeyEncryptSkFamily<B>
        + VecZnxSwithcDegree,
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
        + TakeSvpPPolImpl<B>
        + VecZnxDftAllocBytesImpl<B>
        + VecZnxBigAllocBytesImpl<B>
        + TakeSvpPPolImpl<B>,
{
    let n: usize = module.n();
    let rows: usize = k_in.div_ceil(digits * basek);

    let digits_in: usize = 1;

    let mut ct_in: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_in, rows, digits_in, rank);
    let mut ct_out: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_out, rows, digits_in, rank);
    let mut tsk: GGLWETensorKey<Vec<u8>> = GGLWETensorKey::alloc(n, basek, k_ksk, rows, digits, rank);
    let mut ksk: GGLWESwitchingKey<Vec<u8>> = GGLWESwitchingKey::alloc(n, basek, k_ksk, rows, digits, rank, rank);
    let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GGSWCiphertext::encrypt_sk_scratch_space(module, n, basek, k_in, rank)
            | GGLWESwitchingKey::encrypt_sk_scratch_space(module, n, basek, k_ksk, rank, rank)
            | GGLWETensorKey::encrypt_sk_scratch_space(module, n, basek, k_tsk, rank)
            | GGSWCiphertext::keyswitch_scratch_space(
                module, n, basek, k_out, k_in, k_ksk, digits, k_tsk, digits, rank,
            ),
    );

    let var_xs: f64 = 0.5;

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk_in.fill_ternary_prob(var_xs, &mut source_xs);
    let sk_in_dft: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk_in);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk_out.fill_ternary_prob(var_xs, &mut source_xs);
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
    tsk.encrypt_sk(
        module,
        &sk_out,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

    ct_in.encrypt_sk(
        module,
        &pt_scalar,
        &sk_in_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut ksk_exec: GGLWESwitchingKeyExec<Vec<u8>, B> =
        GGLWESwitchingKeyExec::alloc(module, n, basek, k_ksk, rows, digits, rank, rank);
    let mut tsk_exec: GGLWETensorKeyExec<Vec<u8>, B> = GGLWETensorKeyExec::alloc(module, n, basek, k_ksk, rows, digits, rank);

    ksk_exec.prepare(module, &ksk, scratch.borrow());
    tsk_exec.prepare(module, &tsk, scratch.borrow());

    ct_out.keyswitch(module, &ct_in, &ksk_exec, &tsk_exec, scratch.borrow());

    let max_noise = |col_j: usize| -> f64 {
        noise_ggsw_keyswitch(
            n as f64,
            basek * digits,
            col_j,
            var_xs,
            0f64,
            sigma * sigma,
            0f64,
            rank as f64,
            k_in,
            k_ksk,
            k_tsk,
        ) + 0.5
    };

    ct_out.assert_noise(module, &sk_out_exec, &pt_scalar, &max_noise);
}

pub fn test_ggsw_keyswitch_inplace<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k_ct: usize,
    k_ksk: usize,
    k_tsk: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) where
    Module<B>: GLWESecretExecModuleFamily<B>
        + GGSWEncryptSkFamily<B>
        + GGSWAssertNoiseFamily<B>
        + VecZnxAddScalarInplace
        + VecZnxSubABInplace
        + VecZnxStd
        + VecZnxCopy
        + VmpPMatAlloc<B>
        + VmpPMatPrepare<B>
        + GGSWAssertNoiseFamily<B>
        + GGSWKeySwitchFamily<B>
        + GGLWESwitchingKeyEncryptSkFamily<B>
        + GGLWETensorKeyEncryptSkFamily<B>
        + VecZnxSwithcDegree,
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
    let rows: usize = k_ct.div_ceil(digits * basek);

    let digits_in: usize = 1;

    let mut ct: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_ct, rows, digits_in, rank);
    let mut tsk: GGLWETensorKey<Vec<u8>> = GGLWETensorKey::alloc(n, basek, k_tsk, rows, digits, rank);
    let mut ksk: GGLWESwitchingKey<Vec<u8>> = GGLWESwitchingKey::alloc(n, basek, k_ksk, rows, digits, rank, rank);
    let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GGSWCiphertext::encrypt_sk_scratch_space(module, n, basek, k_ct, rank)
            | GGLWESwitchingKey::encrypt_sk_scratch_space(module, n, basek, k_ksk, rank, rank)
            | GGLWETensorKey::encrypt_sk_scratch_space(module, n, basek, k_tsk, rank)
            | GGSWCiphertext::keyswitch_inplace_scratch_space(module, n, basek, k_ct, k_ksk, digits, k_tsk, digits, rank),
    );

    let var_xs: f64 = 0.5;

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk_in.fill_ternary_prob(var_xs, &mut source_xs);
    let sk_in_dft: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk_in);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk_out.fill_ternary_prob(var_xs, &mut source_xs);
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
    tsk.encrypt_sk(
        module,
        &sk_out,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

    ct.encrypt_sk(
        module,
        &pt_scalar,
        &sk_in_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut ksk_exec: GGLWESwitchingKeyExec<Vec<u8>, B> =
        GGLWESwitchingKeyExec::alloc(module, n, basek, k_ksk, rows, digits, rank, rank);
    let mut tsk_exec: GGLWETensorKeyExec<Vec<u8>, B> = GGLWETensorKeyExec::alloc(module, n, basek, k_ksk, rows, digits, rank);

    ksk_exec.prepare(module, &ksk, scratch.borrow());
    tsk_exec.prepare(module, &tsk, scratch.borrow());

    ct.keyswitch_inplace(module, &ksk_exec, &tsk_exec, scratch.borrow());

    let max_noise = |col_j: usize| -> f64 {
        noise_ggsw_keyswitch(
            n as f64,
            basek * digits,
            col_j,
            var_xs,
            0f64,
            sigma * sigma,
            0f64,
            rank as f64,
            k_ct,
            k_ksk,
            k_tsk,
        ) + 0.5
    };

    ct.assert_noise(module, &sk_out_exec, &pt_scalar, &max_noise);
}
