use backend::{Backend, Module, ScalarZnx, ScalarZnxAlloc, ScalarZnxToMut, ScratchOwned, VecZnxOps, ZnxViewMut};
use sampling::source::Source;

use crate::{
    GGLWEEncryptSkFamily, GGLWEExecLayoutFamily, GGSWCiphertext, GGSWCiphertextExec, GGSWLayoutFamily, GLWEDecryptFamily,
    GLWEExternalProductFamily, GLWEKeyswitchFamily, GLWESecret, GLWESecretExec, GLWESwitchingKey,
    GLWESwitchingKeyEncryptSkFamily, GLWESwitchingKeyExec,
    noise::{log2_std_noise_gglwe_product, noise_ggsw_product},
};

pub(crate) fn test_encrypt_sk<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k_ksk: usize,
    digits: usize,
    rank_in: usize,
    rank_out: usize,
    sigma: f64,
) where
    Module<B>: GGLWEEncryptSkFamily<B> + GLWEDecryptFamily<B>,
{
    let rows: usize = (k_ksk - digits * basek) / (digits * basek);

    let mut ksk: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc(module, basek, k_ksk, rows, digits, rank_in, rank_out);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(GLWESwitchingKey::encrypt_sk_scratch_space(
        module, basek, k_ksk, rank_in, rank_out,
    ));

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank_in);
    sk_in.fill_ternary_prob(0.5, &mut source_xs);

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

    ksk.key
        .assert_noise(module, &sk_out_exec, &sk_in.data, sigma);
}

pub(crate) fn test_key_switch<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k_out: usize,
    k_in: usize,
    k_ksk: usize,
    digits: usize,
    rank_in_s0s1: usize,
    rank_out_s0s1: usize,
    rank_out_s1s2: usize,
    sigma: f64,
) where
    Module<B>: GGLWEEncryptSkFamily<B> + GLWEDecryptFamily<B> + GLWEKeyswitchFamily<B> + GGLWEExecLayoutFamily<B>,
{
    let rows: usize = k_in.div_ceil(basek * digits);
    let digits_in: usize = 1;

    let mut ct_gglwe_s0s1: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc(
        module,
        basek,
        k_in,
        rows,
        digits_in,
        rank_in_s0s1,
        rank_out_s0s1,
    );
    let mut ct_gglwe_s1s2: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc(
        module,
        basek,
        k_ksk,
        rows,
        digits,
        rank_out_s0s1,
        rank_out_s1s2,
    );
    let mut ct_gglwe_s0s2: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc(
        module,
        basek,
        k_out,
        rows,
        digits_in,
        rank_in_s0s1,
        rank_out_s1s2,
    );

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(
            module,
            basek,
            k_ksk,
            rank_in_s0s1,
            rank_in_s0s1 | rank_out_s0s1,
        ) | GLWESwitchingKey::keyswitch_scratch_space(
            module,
            basek,
            k_out,
            k_in,
            k_ksk,
            digits,
            ct_gglwe_s1s2.rank_in(),
            ct_gglwe_s1s2.rank_out(),
        ),
    );

    let mut sk0: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank_in_s0s1);
    sk0.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk1: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank_out_s0s1);
    sk1.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk2: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank_out_s1s2);
    sk2.fill_ternary_prob(0.5, &mut source_xs);
    let sk2_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk2);

    // gglwe_{s1}(s0) = s0 -> s1
    ct_gglwe_s0s1.encrypt_sk(
        module,
        &sk0,
        &sk1,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    // gglwe_{s2}(s1) -> s1 -> s2
    ct_gglwe_s1s2.encrypt_sk(
        module,
        &sk1,
        &sk2,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut ct_gglwe_s1s2_exec: GLWESwitchingKeyExec<Vec<u8>, B> = GLWESwitchingKeyExec::alloc(
        module,
        basek,
        k_out,
        rows,
        digits_in,
        rank_in_s0s1,
        rank_out_s1s2,
    );

    ct_gglwe_s1s2_exec.prepare(module, &ct_gglwe_s1s2, scratch.borrow());

    // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
    ct_gglwe_s0s2.keyswitch(
        module,
        &ct_gglwe_s0s1,
        &ct_gglwe_s1s2_exec,
        scratch.borrow(),
    );

    let max_noise: f64 = log2_std_noise_gglwe_product(
        module.n() as f64,
        basek * digits,
        0.5,
        0.5,
        0f64,
        sigma * sigma,
        0f64,
        rank_out_s0s1 as f64,
        k_in,
        k_ksk,
    );

    ct_gglwe_s0s2
        .key
        .assert_noise(module, &sk2_exec, &sk0.data, max_noise);
}

pub(crate) fn test_key_switch_inplace<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k_ct: usize,
    k_ksk: usize,
    digits: usize,
    rank_in: usize,
    rank_out: usize,
    sigma: f64,
) where
    Module<B>: GLWESwitchingKeyEncryptSkFamily<B> + GLWEKeyswitchFamily<B> + GGLWEExecLayoutFamily<B> + GLWEDecryptFamily<B>,
{
    let rows: usize = k_ct.div_ceil(basek * digits);
    let digits_in: usize = 1;

    let mut ct_gglwe_s0s1: GLWESwitchingKey<Vec<u8>> =
        GLWESwitchingKey::alloc(module, basek, k_ct, rows, digits_in, rank_in, rank_out);
    let mut ct_gglwe_s1s2: GLWESwitchingKey<Vec<u8>> =
        GLWESwitchingKey::alloc(module, basek, k_ksk, rows, digits, rank_out, rank_out);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k_ksk, rank_in, rank_out)
            | GLWESwitchingKey::keyswitch_inplace_scratch_space(module, basek, k_ct, k_ksk, digits, rank_out),
    );

    let var_xs: f64 = 0.5;

    let mut sk0: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank_in);
    sk0.fill_ternary_prob(var_xs, &mut source_xs);

    let mut sk1: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank_out);
    sk1.fill_ternary_prob(var_xs, &mut source_xs);

    let mut sk2: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank_out);
    sk2.fill_ternary_prob(var_xs, &mut source_xs);
    let sk2_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk2);

    // gglwe_{s1}(s0) = s0 -> s1
    ct_gglwe_s0s1.encrypt_sk(
        module,
        &sk0,
        &sk1,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    // gglwe_{s2}(s1) -> s1 -> s2
    ct_gglwe_s1s2.encrypt_sk(
        module,
        &sk1,
        &sk2,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut ct_gglwe_s1s2_exec: GLWESwitchingKeyExec<Vec<u8>, B> =
        GLWESwitchingKeyExec::alloc(module, basek, k_ct, rows, digits_in, rank_in, rank_out);

    ct_gglwe_s1s2_exec.prepare(module, &ct_gglwe_s1s2, scratch.borrow());

    // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
    ct_gglwe_s0s1.keyswitch_inplace(module, &ct_gglwe_s1s2_exec, scratch.borrow());

    let ct_gglwe_s0s2: GLWESwitchingKey<Vec<u8>> = ct_gglwe_s0s1;

    let max_noise: f64 = log2_std_noise_gglwe_product(
        module.n() as f64,
        basek * digits,
        var_xs,
        var_xs,
        0f64,
        sigma * sigma,
        0f64,
        rank_out as f64,
        k_ct,
        k_ksk,
    );

    ct_gglwe_s0s2
        .key
        .assert_noise(module, &sk2_exec, &sk0.data, max_noise);
}

pub(crate) fn test_external_product<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k_out: usize,
    k_in: usize,
    k_ggsw: usize,
    digits: usize,
    rank_in: usize,
    rank_out: usize,
    sigma: f64,
) where
    Module<B>: GLWESwitchingKeyEncryptSkFamily<B> + GLWEExternalProductFamily<B> + GGSWLayoutFamily<B> + GLWEDecryptFamily<B>,
{
    let rows: usize = k_in.div_ceil(basek * digits);
    let digits_in: usize = 1;

    let mut ct_gglwe_in: GLWESwitchingKey<Vec<u8>> =
        GLWESwitchingKey::alloc(module, basek, k_in, rows, digits_in, rank_in, rank_out);
    let mut ct_gglwe_out: GLWESwitchingKey<Vec<u8>> =
        GLWESwitchingKey::alloc(module, basek, k_out, rows, digits_in, rank_in, rank_out);
    let mut ct_rgsw: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(module, basek, k_ggsw, rows, digits, rank_out);

    let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k_in, rank_in, rank_out)
            | GLWESwitchingKey::external_product_scratch_space(module, basek, k_out, k_in, k_ggsw, digits, rank_out)
            | GGSWCiphertext::encrypt_sk_scratch_space(module, basek, k_ggsw, rank_out),
    );

    let r: usize = 1;

    pt_rgsw.to_mut().raw_mut()[r] = 1; // X^{r}

    let var_xs: f64 = 0.5;

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank_in);
    sk_in.fill_ternary_prob(var_xs, &mut source_xs);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank_out);
    sk_out.fill_ternary_prob(var_xs, &mut source_xs);
    let sk_out_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk_out);

    // gglwe_{s1}(s0) = s0 -> s1
    ct_gglwe_in.encrypt_sk(
        module,
        &sk_in,
        &sk_out,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_rgsw.encrypt_sk(
        module,
        &pt_rgsw,
        &sk_out_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut ct_rgsw_exec: GGSWCiphertextExec<Vec<u8>, B> =
        GGSWCiphertextExec::alloc(module, basek, k_ggsw, rows, digits, rank_out);

    ct_rgsw_exec.prepare(module, &ct_rgsw, scratch.borrow());

    // gglwe_(m) (x) RGSW_(X^k) = gglwe_(m * X^k)
    ct_gglwe_out.external_product(module, &ct_gglwe_in, &ct_rgsw_exec, scratch.borrow());

    (0..rank_in).for_each(|i| {
        module.vec_znx_rotate_inplace(r as i64, &mut sk_in.data, i); // * X^{r}
    });

    let var_gct_err_lhs: f64 = sigma * sigma;
    let var_gct_err_rhs: f64 = 0f64;

    let var_msg: f64 = 1f64 / module.n() as f64; // X^{k}
    let var_a0_err: f64 = sigma * sigma;
    let var_a1_err: f64 = 1f64 / 12f64;

    let max_noise: f64 = noise_ggsw_product(
        module.n() as f64,
        basek * digits,
        var_xs,
        var_msg,
        var_a0_err,
        var_a1_err,
        var_gct_err_lhs,
        var_gct_err_rhs,
        rank_out as f64,
        k_in,
        k_ggsw,
    );

    ct_gglwe_out
        .key
        .assert_noise(module, &sk_out_exec, &sk_in.data, max_noise);
}

pub(crate) fn test_external_product_inplace<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k_ct: usize,
    k_ggsw: usize,
    digits: usize,
    rank_in: usize,
    rank_out: usize,
    sigma: f64,
) where
    Module<B>: GLWESwitchingKeyEncryptSkFamily<B> + GLWEExternalProductFamily<B> + GGSWLayoutFamily<B> + GLWEDecryptFamily<B>,
{
    let rows: usize = k_ct.div_ceil(basek * digits);

    let digits_in: usize = 1;

    let mut ct_gglwe: GLWESwitchingKey<Vec<u8>> =
        GLWESwitchingKey::alloc(module, basek, k_ct, rows, digits_in, rank_in, rank_out);
    let mut ct_rgsw: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(module, basek, k_ggsw, rows, digits, rank_out);

    let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k_ct, rank_in, rank_out)
            | GLWESwitchingKey::external_product_inplace_scratch_space(module, basek, k_ct, k_ggsw, digits, rank_out)
            | GGSWCiphertext::encrypt_sk_scratch_space(module, basek, k_ggsw, rank_out),
    );

    let r: usize = 1;

    pt_rgsw.to_mut().raw_mut()[r] = 1; // X^{r}

    let var_xs: f64 = 0.5;

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank_in);
    sk_in.fill_ternary_prob(var_xs, &mut source_xs);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank_out);
    sk_out.fill_ternary_prob(var_xs, &mut source_xs);
    let sk_out_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk_out);

    // gglwe_{s1}(s0) = s0 -> s1
    ct_gglwe.encrypt_sk(
        module,
        &sk_in,
        &sk_out,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_rgsw.encrypt_sk(
        module,
        &pt_rgsw,
        &sk_out_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut ct_rgsw_exec: GGSWCiphertextExec<Vec<u8>, B> =
        GGSWCiphertextExec::alloc(module, basek, k_ggsw, rows, digits, rank_out);

    ct_rgsw_exec.prepare(module, &ct_rgsw, scratch.borrow());

    // gglwe_(m) (x) RGSW_(X^k) = gglwe_(m * X^k)
    ct_gglwe.external_product_inplace(module, &ct_rgsw_exec, scratch.borrow());

    (0..rank_in).for_each(|i| {
        module.vec_znx_rotate_inplace(r as i64, &mut sk_in.data, i); // * X^{r}
    });

    let var_gct_err_lhs: f64 = sigma * sigma;
    let var_gct_err_rhs: f64 = 0f64;

    let var_msg: f64 = 1f64 / module.n() as f64; // X^{k}
    let var_a0_err: f64 = sigma * sigma;
    let var_a1_err: f64 = 1f64 / 12f64;

    let max_noise: f64 = noise_ggsw_product(
        module.n() as f64,
        basek * digits,
        var_xs,
        var_msg,
        var_a0_err,
        var_a1_err,
        var_gct_err_lhs,
        var_gct_err_rhs,
        rank_out as f64,
        k_ct,
        k_ggsw,
    );

    ct_gglwe
        .key
        .assert_noise(module, &sk_out_exec, &sk_in.data, max_noise);
}
