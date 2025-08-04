use backend::{
    hal::{
        api::{
            ModuleNew, ScalarZnxAlloc, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniform, VecZnxRotateInplace, ZnxViewMut,
        },
        layouts::{Backend, Module, ScalarZnx, ScratchOwned},
        oep::{ScratchTakeVecZnxBigImpl, ScratchTakeVecZnxDftImpl},
    },
    implementation::cpu_avx::FFT64,
};
use sampling::source::Source;

use crate::{
    GGSWCiphertext, GGSWCiphertextExec, GGSWLayoutFamily, GLWECiphertext, GLWEDecryptFamily, GLWEEncryptSkFamily,
    GLWEExternalProductFamily, GLWEPlaintext, GLWESecret, GLWESecretExec, GLWESecretFamily, Infos, noise::noise_ggsw_product,
};

#[test]
fn apply() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_in: usize = 45;
    let digits: usize = k_in.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ggsw: usize = k_in + basek * di;
            let k_out: usize = k_ggsw; // Better capture noise
            println!("test external_product digits: {} rank: {}", di, rank);
            test_external_product(&module, basek, k_out, k_in, k_ggsw, di, rank, 3.2);
        });
    });
}

#[test]
fn apply_inplace() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_ct: usize = 60;
    let digits: usize = k_ct.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ggsw: usize = k_ct + basek * di;
            println!("test external_product digits: {} rank: {}", di, rank);
            test_external_product_inplace(&module, basek, k_ct, k_ggsw, di, rank, 3.2);
        });
    });
}

fn test_external_product<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k_out: usize,
    k_in: usize,
    k_ggsw: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) where
    Module<B>:
        GLWEEncryptSkFamily<B> + GLWEDecryptFamily<B> + GLWESecretFamily<B> + GLWEExternalProductFamily<B> + GGSWLayoutFamily<B>,
    B: ScratchTakeVecZnxDftImpl<B> + ScratchTakeVecZnxBigImpl<B>,
{
    let rows: usize = k_in.div_ceil(basek * digits);

    let mut ct_ggsw: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(module, basek, k_ggsw, rows, digits, rank);
    let mut ct_glwe_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, basek, k_in, rank);
    let mut ct_glwe_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, basek, k_out, rank);
    let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k_in);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    module.vec_znx_fill_uniform(basek, &mut pt_want.data, 0, k_in, &mut source_xa);

    pt_want.data.at_mut(0, 0)[1] = 1;

    let k: usize = 1;

    pt_rgsw.raw_mut()[k] = 1; // X^{k}

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GGSWCiphertext::encrypt_sk_scratch_space(module, basek, ct_ggsw.k(), rank)
            | GLWECiphertext::encrypt_sk_scratch_space(module, basek, ct_glwe_in.k())
            | GLWECiphertext::external_product_scratch_space(
                module,
                basek,
                ct_glwe_out.k(),
                ct_glwe_in.k(),
                ct_ggsw.k(),
                digits,
                rank,
            ),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk);

    ct_ggsw.encrypt_sk(
        module,
        &pt_rgsw,
        &sk_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_glwe_in.encrypt_sk(
        module,
        &pt_want,
        &sk_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let ct_ggsw_exec: GGSWCiphertextExec<Vec<u8>, B> = GGSWCiphertextExec::from(module, &ct_ggsw, scratch.borrow());

    ct_glwe_out.external_product(module, &ct_glwe_in, &ct_ggsw_exec, scratch.borrow());

    module.vec_znx_rotate_inplace(k as i64, &mut pt_want.data, 0);

    let var_gct_err_lhs: f64 = sigma * sigma;
    let var_gct_err_rhs: f64 = 0f64;

    let var_msg: f64 = 1f64 / module.n() as f64; // X^{k}
    let var_a0_err: f64 = sigma * sigma;
    let var_a1_err: f64 = 1f64 / 12f64;

    let max_noise: f64 = noise_ggsw_product(
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
    );

    ct_glwe_out.assert_noise(module, &sk_exec, &pt_want, max_noise + 0.5);
}

fn test_external_product_inplace<B: Backend>(
    module: &Module<B>,
    basek: usize,
    k_ct: usize,
    k_ggsw: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) where
    Module<B>:
        GLWEEncryptSkFamily<B> + GLWEDecryptFamily<B> + GLWESecretFamily<B> + GLWEExternalProductFamily<B> + GGSWLayoutFamily<B>,
    B: ScratchTakeVecZnxDftImpl<B> + ScratchTakeVecZnxBigImpl<B>,
{
    let rows: usize = k_ct.div_ceil(basek * digits);

    let mut ct_ggsw: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(module, basek, k_ggsw, rows, digits, rank);
    let mut ct_glwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, basek, k_ct, rank);
    let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k_ct);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    // Random input plaintext
    module.vec_znx_fill_uniform(basek, &mut pt_want.data, 0, k_ct, &mut source_xa);

    pt_want.data.at_mut(0, 0)[1] = 1;

    let k: usize = 1;

    pt_rgsw.raw_mut()[k] = 1; // X^{k}

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GGSWCiphertext::encrypt_sk_scratch_space(module, basek, ct_ggsw.k(), rank)
            | GLWECiphertext::encrypt_sk_scratch_space(module, basek, ct_glwe.k())
            | GLWECiphertext::external_product_inplace_scratch_space(module, basek, ct_glwe.k(), ct_ggsw.k(), digits, rank),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk);

    ct_ggsw.encrypt_sk(
        module,
        &pt_rgsw,
        &sk_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_glwe.encrypt_sk(
        module,
        &pt_want,
        &sk_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let ct_ggsw_exec: GGSWCiphertextExec<Vec<u8>, B> = GGSWCiphertextExec::from(module, &ct_ggsw, scratch.borrow());

    ct_glwe.external_product_inplace(module, &ct_ggsw_exec, scratch.borrow());

    module.vec_znx_rotate_inplace(k as i64, &mut pt_want.data, 0);

    let var_gct_err_lhs: f64 = sigma * sigma;
    let var_gct_err_rhs: f64 = 0f64;

    let var_msg: f64 = 1f64 / module.n() as f64; // X^{k}
    let var_a0_err: f64 = sigma * sigma;
    let var_a1_err: f64 = 1f64 / 12f64;

    let max_noise: f64 = noise_ggsw_product(
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
    );

    ct_glwe.assert_noise(module, &sk_exec, &pt_want, max_noise + 0.5);
}
