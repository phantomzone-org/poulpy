use backend::hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarInplace, VecZnxAutomorphism, VecZnxAutomorphismInplace, VecZnxCopy,
        VecZnxStd, VecZnxSubABInplace, VecZnxSwithcDegree, VmpPMatAlloc, VmpPMatPrepare,
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
        GGLWEAutomorphismKey, GGLWETensorKey, GGSWCiphertext, GLWESecret,
        prepared::{GGLWEAutomorphismKeyExec, GLWESecretExec, GGLWETensorKeyExec},
    },
    noise::noise_ggsw_keyswitch,
    trait_families::GGSWAssertNoiseFamily,
};

use crate::trait_families::{
    GGLWESwitchingKeyEncryptSkFamily, GGLWETensorKeyEncryptSkFamily, GGSWKeySwitchFamily, GLWESecretExecModuleFamily,
};

pub fn test_ggsw_automorphism<B: Backend>(
    p: i64,
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
    Module<B>: GGSWAssertNoiseFamily<B>
        + GLWESecretExecModuleFamily<B>
        + VecZnxAddScalarInplace
        + VecZnxCopy
        + VecZnxStd
        + VecZnxSubABInplace
        + VmpPMatAlloc<B>
        + VmpPMatPrepare<B>
        + GGSWKeySwitchFamily<B>
        + GGLWESwitchingKeyEncryptSkFamily<B>
        + GGLWETensorKeyEncryptSkFamily<B>
        + VecZnxSwithcDegree
        + VecZnxAutomorphismInplace
        + VecZnxAutomorphismInplace
        + VecZnxAutomorphism,
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
    let rows: usize = k_in.div_ceil(basek * digits);
    let rows_in: usize = k_in.div_euclid(basek * digits);

    let digits_in: usize = 1;

    let mut ct_in: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_in, rows_in, digits_in, rank);
    let mut ct_out: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_out, rows_in, digits_in, rank);
    let mut tensor_key: GGLWETensorKey<Vec<u8>> = GGLWETensorKey::alloc(n, basek, k_tsk, rows, digits, rank);
    let mut auto_key: GGLWEAutomorphismKey<Vec<u8>> = GGLWEAutomorphismKey::alloc(n, basek, k_ksk, rows, digits, rank);
    let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GGSWCiphertext::encrypt_sk_scratch_space(module, n, basek, k_in, rank)
            | GGLWEAutomorphismKey::encrypt_sk_scratch_space(module, n, basek, k_ksk, rank)
            | GGLWETensorKey::encrypt_sk_scratch_space(module, n, basek, k_tsk, rank)
            | GGSWCiphertext::automorphism_scratch_space(
                module, n, basek, k_out, k_in, k_ksk, digits, k_tsk, digits, rank,
            ),
    );

    let var_xs: f64 = 0.5;

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(var_xs, &mut source_xs);
    let sk_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk);

    auto_key.encrypt_sk(
        module,
        p,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );
    tensor_key.encrypt_sk(
        module,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

    ct_in.encrypt_sk(
        module,
        &pt_scalar,
        &sk_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut auto_key_exec: GGLWEAutomorphismKeyExec<Vec<u8>, B> =
        GGLWEAutomorphismKeyExec::alloc(module, n, basek, k_ksk, rows, digits, rank);
    auto_key_exec.prepare(module, &auto_key, scratch.borrow());

    let mut tsk_exec: GGLWETensorKeyExec<Vec<u8>, B> = GGLWETensorKeyExec::alloc(module, n, basek, k_tsk, rows, digits, rank);
    tsk_exec.prepare(module, &tensor_key, scratch.borrow());

    ct_out.automorphism(module, &ct_in, &auto_key_exec, &tsk_exec, scratch.borrow());

    module.vec_znx_automorphism_inplace(p, &mut pt_scalar.as_vec_znx_mut(), 0);

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

    ct_out.assert_noise(module, &sk_exec, &pt_scalar, &max_noise);
}

pub fn test_ggsw_automorphism_inplace<B: Backend>(
    p: i64,
    module: &Module<B>,
    basek: usize,
    k_ct: usize,
    k_ksk: usize,
    k_tsk: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) where
    Module<B>: GGSWAssertNoiseFamily<B>
        + GLWESecretExecModuleFamily<B>
        + VecZnxAddScalarInplace
        + VecZnxCopy
        + VecZnxStd
        + VecZnxSubABInplace
        + VmpPMatAlloc<B>
        + VmpPMatPrepare<B>
        + GGSWKeySwitchFamily<B>
        + GGLWESwitchingKeyEncryptSkFamily<B>
        + GGLWETensorKeyEncryptSkFamily<B>
        + VecZnxSwithcDegree
        + VecZnxAutomorphismInplace
        + VecZnxAutomorphismInplace
        + VecZnxAutomorphism,
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
    let rows_in: usize = k_ct.div_euclid(basek * digits);
    let digits_in: usize = 1;

    let mut ct: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_ct, rows_in, digits_in, rank);
    let mut tensor_key: GGLWETensorKey<Vec<u8>> = GGLWETensorKey::alloc(n, basek, k_tsk, rows, digits, rank);
    let mut auto_key: GGLWEAutomorphismKey<Vec<u8>> = GGLWEAutomorphismKey::alloc(n, basek, k_ksk, rows, digits, rank);
    let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GGSWCiphertext::encrypt_sk_scratch_space(module, n, basek, k_ct, rank)
            | GGLWEAutomorphismKey::encrypt_sk_scratch_space(module, n, basek, k_ksk, rank)
            | GGLWETensorKey::encrypt_sk_scratch_space(module, n, basek, k_tsk, rank)
            | GGSWCiphertext::automorphism_inplace_scratch_space(module, n, basek, k_ct, k_ksk, digits, k_tsk, digits, rank),
    );

    let var_xs: f64 = 0.5;

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(var_xs, &mut source_xs);
    let sk_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk);

    auto_key.encrypt_sk(
        module,
        p,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );
    tensor_key.encrypt_sk(
        module,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

    ct.encrypt_sk(
        module,
        &pt_scalar,
        &sk_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut auto_key_exec: GGLWEAutomorphismKeyExec<Vec<u8>, B> =
        GGLWEAutomorphismKeyExec::alloc(module, n, basek, k_ksk, rows, digits, rank);
    auto_key_exec.prepare(module, &auto_key, scratch.borrow());

    let mut tsk_exec: GGLWETensorKeyExec<Vec<u8>, B> = GGLWETensorKeyExec::alloc(module, n, basek, k_tsk, rows, digits, rank);
    tsk_exec.prepare(module, &tensor_key, scratch.borrow());

    ct.automorphism_inplace(module, &auto_key_exec, &tsk_exec, scratch.borrow());

    module.vec_znx_automorphism_inplace(p, &mut pt_scalar.as_vec_znx_mut(), 0);

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

    ct.assert_noise(module, &sk_exec, &pt_scalar, &max_noise);
}
