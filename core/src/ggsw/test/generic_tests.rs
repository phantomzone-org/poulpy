use backend::{Backend, Module, ScalarZnx, ScalarZnxAlloc, ScalarZnxOps, ScratchOwned, VecZnxOps, VecZnxToMut, ZnxViewMut};
use sampling::source::Source;

use crate::{
    AutomorphismExecFamily, AutomorphismKey, AutomorphismKeyExec, GGLWEExecLayoutFamily, GGSWAssertNoiseFamily, GGSWCiphertext,
    GGSWCiphertextExec, GGSWEncryptSkFamily, GGSWKeySwitchFamily, GLWESecret, GLWESecretExec, GLWESecretFamily, GLWESwitchingKey,
    GLWESwitchingKeyEncryptSkFamily, GLWESwitchingKeyExec, GLWETensorKey, GLWETensorKeyEncryptSkFamily, GLWETensorKeyExec,
    noise::{noise_ggsw_keyswitch, noise_ggsw_product},
};

pub(crate) fn test_encrypt_sk<B: Backend>(module: &Module<B>, basek: usize, k: usize, digits: usize, rank: usize, sigma: f64)
where
    Module<B>: GLWESecretFamily<B> + GGSWEncryptSkFamily<B> + GGSWAssertNoiseFamily<B>,
{
    let rows: usize = (k - digits * basek) / (digits * basek);

    let mut ct: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(module, basek, k, rows, digits, rank);

    let mut pt_scalar: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    pt_scalar.fill_ternary_hw(0, module.n(), &mut source_xs);

    let mut scratch: ScratchOwned = ScratchOwned::new(GGSWCiphertext::encrypt_sk_scratch_space(
        module, basek, k, rank,
    ));

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let mut sk_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk);
    sk_exec.prepare(module, &sk);

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

pub(crate) fn test_keyswitch<B: Backend>(
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
        + GGSWKeySwitchFamily<B>
        + GLWESwitchingKeyEncryptSkFamily<B>
        + GLWETensorKeyEncryptSkFamily<B>
        + GGLWEExecLayoutFamily<B>,
{
    let rows: usize = k_in.div_ceil(digits * basek);

    let digits_in: usize = 1;

    let mut ct_in: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(module, basek, k_in, rows, digits_in, rank);
    let mut ct_out: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(module, basek, k_out, rows, digits_in, rank);
    let mut tsk: GLWETensorKey<Vec<u8>> = GLWETensorKey::alloc(module, basek, k_ksk, rows, digits, rank);
    let mut ksk: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc(module, basek, k_ksk, rows, digits, rank, rank);
    let mut pt_scalar: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(module, basek, k_in, rank)
            | GLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k_ksk, rank, rank)
            | GLWETensorKey::encrypt_sk_scratch_space(module, basek, k_tsk, rank)
            | GGSWCiphertext::keyswitch_scratch_space(
                module, basek, k_out, k_in, k_ksk, digits, k_tsk, digits, rank,
            ),
    );

    let var_xs: f64 = 0.5;

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
    sk_in.fill_ternary_prob(var_xs, &mut source_xs);
    let sk_in_dft: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk_in);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
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

    pt_scalar.fill_ternary_hw(0, module.n(), &mut source_xs);

    ct_in.encrypt_sk(
        module,
        &pt_scalar,
        &sk_in_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut ksk_exec: GLWESwitchingKeyExec<Vec<u8>, B> =
        GLWESwitchingKeyExec::alloc(module, basek, k_ksk, rows, digits, rank, rank);
    let mut tsk_exec: GLWETensorKeyExec<Vec<u8>, B> = GLWETensorKeyExec::alloc(module, basek, k_ksk, rows, digits, rank);

    ksk_exec.prepare(module, &ksk, scratch.borrow());
    tsk_exec.prepare(module, &tsk, scratch.borrow());

    ct_out.keyswitch(module, &ct_in, &ksk_exec, &tsk_exec, scratch.borrow());

    let max_noise = |col_j: usize| -> f64 {
        noise_ggsw_keyswitch(
            module.n() as f64,
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

pub(crate) fn test_keyswitch_inplace<B: Backend>(
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
        + GGSWKeySwitchFamily<B>
        + GLWESwitchingKeyEncryptSkFamily<B>
        + GLWETensorKeyEncryptSkFamily<B>
        + GGLWEExecLayoutFamily<B>,
{
    let rows: usize = k_ct.div_ceil(digits * basek);

    let digits_in: usize = 1;

    let mut ct: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(module, basek, k_ct, rows, digits_in, rank);
    let mut tsk: GLWETensorKey<Vec<u8>> = GLWETensorKey::alloc(module, basek, k_tsk, rows, digits, rank);
    let mut ksk: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc(module, basek, k_ksk, rows, digits, rank, rank);
    let mut pt_scalar: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(module, basek, k_ct, rank)
            | GLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k_ksk, rank, rank)
            | GLWETensorKey::encrypt_sk_scratch_space(module, basek, k_tsk, rank)
            | GGSWCiphertext::keyswitch_inplace_scratch_space(module, basek, k_ct, k_ksk, digits, k_tsk, digits, rank),
    );

    let var_xs: f64 = 0.5;

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
    sk_in.fill_ternary_prob(var_xs, &mut source_xs);
    let sk_in_dft: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk_in);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
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

    pt_scalar.fill_ternary_hw(0, module.n(), &mut source_xs);

    ct.encrypt_sk(
        module,
        &pt_scalar,
        &sk_in_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut ksk_exec: GLWESwitchingKeyExec<Vec<u8>, B> =
        GLWESwitchingKeyExec::alloc(module, basek, k_ksk, rows, digits, rank, rank);
    let mut tsk_exec: GLWETensorKeyExec<Vec<u8>, B> = GLWETensorKeyExec::alloc(module, basek, k_ksk, rows, digits, rank);

    ksk_exec.prepare(module, &ksk, scratch.borrow());
    tsk_exec.prepare(module, &tsk, scratch.borrow());

    ct.keyswitch_inplace(module, &ksk_exec, &tsk_exec, scratch.borrow());

    let max_noise = |col_j: usize| -> f64 {
        noise_ggsw_keyswitch(
            module.n() as f64,
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

pub(crate) fn test_automorphism<B: Backend>(
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
        + GGSWKeySwitchFamily<B>
        + GLWESwitchingKeyEncryptSkFamily<B>
        + GLWETensorKeyEncryptSkFamily<B>
        + GGLWEExecLayoutFamily<B>
        + AutomorphismExecFamily<B>,
{
    let rows: usize = k_in.div_ceil(basek * digits);
    let rows_in: usize = k_in.div_euclid(basek * digits);

    let digits_in: usize = 1;

    let mut ct_in: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(module, basek, k_in, rows_in, digits_in, rank);
    let mut ct_out: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(module, basek, k_out, rows_in, digits_in, rank);
    let mut tensor_key: GLWETensorKey<Vec<u8>> = GLWETensorKey::alloc(module, basek, k_tsk, rows, digits, rank);
    let mut auto_key: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc(module, basek, k_ksk, rows, digits, rank);
    let mut pt_scalar: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(module, basek, k_in, rank)
            | AutomorphismKey::encrypt_sk_scratch_space(module, basek, k_ksk, rank)
            | GLWETensorKey::encrypt_sk_scratch_space(module, basek, k_tsk, rank)
            | GGSWCiphertext::automorphism_scratch_space(
                module, basek, k_out, k_in, k_ksk, digits, k_tsk, digits, rank,
            ),
    );

    let var_xs: f64 = 0.5;

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
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

    pt_scalar.fill_ternary_hw(0, module.n(), &mut source_xs);

    ct_in.encrypt_sk(
        module,
        &pt_scalar,
        &sk_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut auto_key_exec: AutomorphismKeyExec<Vec<u8>, B> = AutomorphismKeyExec::alloc(module, basek, k_ksk, rows, digits, rank);
    auto_key_exec.prepare(module, &auto_key, scratch.borrow());

    let mut tsk_exec: GLWETensorKeyExec<Vec<u8>, B> = GLWETensorKeyExec::alloc(module, basek, k_tsk, rows, digits, rank);
    tsk_exec.prepare(module, &tensor_key, scratch.borrow());

    ct_out.automorphism(module, &ct_in, &auto_key_exec, &tsk_exec, scratch.borrow());

    module.scalar_znx_automorphism_inplace(p, &mut pt_scalar, 0);

    let max_noise = |col_j: usize| -> f64 {
        noise_ggsw_keyswitch(
            module.n() as f64,
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

pub(crate) fn test_automorphism_inplace<B: Backend>(
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
        + GGSWKeySwitchFamily<B>
        + GLWESwitchingKeyEncryptSkFamily<B>
        + GLWETensorKeyEncryptSkFamily<B>
        + GGLWEExecLayoutFamily<B>
        + AutomorphismExecFamily<B>,
{
    let rows: usize = k_ct.div_ceil(digits * basek);
    let rows_in: usize = k_ct.div_euclid(basek * digits);
    let digits_in: usize = 1;

    let mut ct: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(module, basek, k_ct, rows_in, digits_in, rank);
    let mut tensor_key: GLWETensorKey<Vec<u8>> = GLWETensorKey::alloc(module, basek, k_tsk, rows, digits, rank);
    let mut auto_key: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc(module, basek, k_ksk, rows, digits, rank);
    let mut pt_scalar: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(module, basek, k_ct, rank)
            | AutomorphismKey::encrypt_sk_scratch_space(module, basek, k_ksk, rank)
            | GLWETensorKey::encrypt_sk_scratch_space(module, basek, k_tsk, rank)
            | GGSWCiphertext::automorphism_inplace_scratch_space(module, basek, k_ct, k_ksk, digits, k_tsk, digits, rank),
    );

    let var_xs: f64 = 0.5;

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
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

    pt_scalar.fill_ternary_hw(0, module.n(), &mut source_xs);

    ct.encrypt_sk(
        module,
        &pt_scalar,
        &sk_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut auto_key_exec: AutomorphismKeyExec<Vec<u8>, B> = AutomorphismKeyExec::alloc(module, basek, k_ksk, rows, digits, rank);
    auto_key_exec.prepare(module, &auto_key, scratch.borrow());

    let mut tsk_exec: GLWETensorKeyExec<Vec<u8>, B> = GLWETensorKeyExec::alloc(module, basek, k_tsk, rows, digits, rank);
    tsk_exec.prepare(module, &tensor_key, scratch.borrow());

    ct.automorphism_inplace(module, &auto_key_exec, &tsk_exec, scratch.borrow());

    module.scalar_znx_automorphism_inplace(p, &mut pt_scalar, 0);

    let max_noise = |col_j: usize| -> f64 {
        noise_ggsw_keyswitch(
            module.n() as f64,
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

pub(crate) fn test_external_product<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k_in: usize,
    k_out: usize,
    k_ggsw: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) where
    Module<B>: GGSWAssertNoiseFamily<B>
        + GGSWKeySwitchFamily<B>
        + GLWESwitchingKeyEncryptSkFamily<B>
        + GLWETensorKeyEncryptSkFamily<B>
        + GGLWEExecLayoutFamily<B>,
{
    let rows: usize = k_in.div_ceil(basek * digits);
    let rows_in: usize = k_in.div_euclid(basek * digits);
    let digits_in: usize = 1;

    let mut ct_ggsw_lhs_in: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(module, basek, k_in, rows_in, digits_in, rank);
    let mut ct_ggsw_lhs_out: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(module, basek, k_out, rows_in, digits_in, rank);
    let mut ct_ggsw_rhs: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(module, basek, k_ggsw, rows, digits, rank);
    let mut pt_ggsw_lhs: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);
    let mut pt_ggsw_rhs: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    pt_ggsw_lhs.fill_ternary_prob(0, 0.5, &mut source_xs);

    let k: usize = 1;

    pt_ggsw_rhs.to_mut().raw_mut()[k] = 1; //X^{k}

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(module, basek, k_ggsw, rank)
            | GGSWCiphertext::external_product_scratch_space(module, basek, k_out, k_in, k_ggsw, digits, rank),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk);

    ct_ggsw_rhs.encrypt_sk(
        module,
        &pt_ggsw_rhs,
        &sk_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_ggsw_lhs_in.encrypt_sk(
        module,
        &pt_ggsw_lhs,
        &sk_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut ct_rhs_exec: GGSWCiphertextExec<Vec<u8>, B> = GGSWCiphertextExec::alloc(module, basek, k_ggsw, rows, digits, rank);
    ct_rhs_exec.prepare(module, &ct_ggsw_rhs, scratch.borrow());

    ct_ggsw_lhs_out.external_product(module, &ct_ggsw_lhs_in, &ct_rhs_exec, scratch.borrow());

    module.vec_znx_rotate_inplace(k as i64, &mut pt_ggsw_lhs, 0);

    let var_gct_err_lhs: f64 = sigma * sigma;
    let var_gct_err_rhs: f64 = 0f64;

    let var_msg: f64 = 1f64 / module.n() as f64; // X^{k}
    let var_a0_err: f64 = sigma * sigma;
    let var_a1_err: f64 = 1f64 / 12f64;

    let max_noise = |_col_j: usize| -> f64 {
        noise_ggsw_product(
            module.n() as f64,
            basek * digits,
            0.5,
            var_msg,
            var_a0_err,
            var_a1_err,
            var_gct_err_lhs,
            var_gct_err_rhs,
            rank as f64,
            k_in,
            k_ggsw,
        ) + 0.5
    };

    ct_ggsw_lhs_out.assert_noise(module, &sk_exec, &pt_ggsw_lhs, &max_noise);
}

pub(crate) fn test_external_product_inplace<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k_ct: usize,
    k_ggsw: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) where
    Module<B>: GGSWAssertNoiseFamily<B>
        + GGSWKeySwitchFamily<B>
        + GLWESwitchingKeyEncryptSkFamily<B>
        + GLWETensorKeyEncryptSkFamily<B>
        + GGLWEExecLayoutFamily<B>,
{
    let rows: usize = k_ct.div_ceil(digits * basek);
    let rows_in: usize = k_ct.div_euclid(basek * digits);
    let digits_in: usize = 1;

    let mut ct_ggsw_lhs: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(module, basek, k_ct, rows_in, digits_in, rank);
    let mut ct_ggsw_rhs: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(module, basek, k_ggsw, rows, digits, rank);

    let mut pt_ggsw_lhs: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);
    let mut pt_ggsw_rhs: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    pt_ggsw_lhs.fill_ternary_prob(0, 0.5, &mut source_xs);

    let k: usize = 1;

    pt_ggsw_rhs.to_mut().raw_mut()[k] = 1; //X^{k}

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(module, basek, k_ggsw, rank)
            | GGSWCiphertext::external_product_inplace_scratch_space(module, basek, k_ct, k_ggsw, digits, rank),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk);

    ct_ggsw_rhs.encrypt_sk(
        module,
        &pt_ggsw_rhs,
        &sk_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_ggsw_lhs.encrypt_sk(
        module,
        &pt_ggsw_lhs,
        &sk_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut ct_rhs_exec: GGSWCiphertextExec<Vec<u8>, B> = GGSWCiphertextExec::alloc(module, basek, k_ggsw, rows, digits, rank);
    ct_rhs_exec.prepare(module, &ct_ggsw_rhs, scratch.borrow());

    ct_ggsw_lhs.external_product_inplace(module, &ct_rhs_exec, scratch.borrow());

    module.vec_znx_rotate_inplace(k as i64, &mut pt_ggsw_lhs, 0);

    let var_gct_err_lhs: f64 = sigma * sigma;
    let var_gct_err_rhs: f64 = 0f64;

    let var_msg: f64 = 1f64 / module.n() as f64; // X^{k}
    let var_a0_err: f64 = sigma * sigma;
    let var_a1_err: f64 = 1f64 / 12f64;

    let max_noise = |_col_j: usize| -> f64 {
        noise_ggsw_product(
            module.n() as f64,
            basek * digits,
            0.5,
            var_msg,
            var_a0_err,
            var_a1_err,
            var_gct_err_lhs,
            var_gct_err_rhs,
            rank as f64,
            k_ct,
            k_ggsw,
        ) + 0.5
    };

    ct_ggsw_lhs.assert_noise(module, &sk_exec, &pt_ggsw_lhs, &max_noise);
}
