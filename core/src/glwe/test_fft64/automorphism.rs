use backend::{
    Backend, FFT64, MatZnxAlloc, Module, ModuleNew, ScratchOwned, ScratchOwnedAlloc, ScratchOwnedBorrow, ScratchTakeSvpPPolImpl,
    ScratchTakeVecZnxBigImpl, ScratchTakeVecZnxDftImpl, VecZnxAutomorphismInplace, VecZnxFillUniform,
};

use sampling::source::Source;

use crate::{
    AutomorphismExecFamily, AutomorphismKey, AutomorphismKeyEncryptSkFamily, AutomorphismKeyExec, GGLWEExecLayoutFamily,
    GLWECiphertext, GLWEDecryptFamily, GLWEPlaintext, GLWESecret, GLWESecretExec, Infos, noise::log2_std_noise_gglwe_product,
};

#[test]
fn apply_inplace() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_ct: usize = 60;
    let digits: usize = k_ct.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = k_ct + basek * di;
            println!("test automorphism_inplace digits: {} rank: {}", di, rank);
            test_automorphism_inplace(&module, basek, -5, k_ct, k_ksk, di, rank, 3.2);
        });
    });
}

#[test]
fn apply() {
    let log_n: usize = 8;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let basek: usize = 12;
    let k_in: usize = 60;
    let digits: usize = k_in.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = k_in + basek * di;
            let k_out: usize = k_ksk; // Better capture noise.
            println!("test automorphism digits: {} rank: {}", di, rank);
            test_automorphism(&module, basek, -5, k_out, k_in, k_ksk, di, rank, 3.2);
        })
    });
}

fn test_automorphism<B: Backend>(
    module: &Module<B>,
    basek: usize,
    p: i64,
    k_out: usize,
    k_in: usize,
    k_ksk: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) where
    Module<B>: AutomorphismKeyEncryptSkFamily<B>
        + GLWEDecryptFamily<B>
        + AutomorphismExecFamily<B>
        + GGLWEExecLayoutFamily<B>
        + MatZnxAlloc,
    B: ScratchTakeVecZnxDftImpl<B> + ScratchTakeVecZnxBigImpl<B> + ScratchTakeSvpPPolImpl<B>,
{
    let rows: usize = k_in.div_ceil(basek * digits);

    let mut autokey: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc(module, basek, k_ksk, rows, digits, rank);
    let mut ct_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, basek, k_in, rank);
    let mut ct_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, basek, k_out, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k_in);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    module.vec_znx_fill_uniform(basek, &mut pt_want.data, 0, k_in, &mut source_xa);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        AutomorphismKey::encrypt_sk_scratch_space(module, basek, autokey.k(), rank)
            | GLWECiphertext::decrypt_scratch_space(module, basek, ct_out.k())
            | GLWECiphertext::encrypt_sk_scratch_space(module, basek, ct_in.k())
            | GLWECiphertext::automorphism_scratch_space(
                module,
                basek,
                ct_out.k(),
                ct_in.k(),
                autokey.k(),
                digits,
                rank,
            ),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk);

    autokey.encrypt_sk(
        module,
        p,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_in.encrypt_sk(
        module,
        &pt_want,
        &sk_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut autokey_exec: AutomorphismKeyExec<Vec<u8>, B> = AutomorphismKeyExec::alloc(module, basek, k_ksk, rows, digits, rank);
    autokey_exec.prepare(module, &autokey, scratch.borrow());

    ct_out.automorphism(module, &ct_in, &autokey_exec, scratch.borrow());

    let max_noise: f64 = log2_std_noise_gglwe_product(
        module.n() as f64,
        basek * digits,
        0.5,
        0.5,
        0f64,
        sigma * sigma,
        0f64,
        rank as f64,
        k_in,
        k_ksk,
    );

    module.vec_znx_automorphism_inplace(p, &mut pt_want.data, 0);

    ct_out.assert_noise(module, &sk_exec, &pt_want, max_noise + 1.0);
}

fn test_automorphism_inplace<B: Backend>(
    module: &Module<B>,
    basek: usize,
    p: i64,
    k_ct: usize,
    k_ksk: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) where
    Module<B>: AutomorphismKeyEncryptSkFamily<B>
        + GLWEDecryptFamily<B>
        + AutomorphismExecFamily<B>
        + GGLWEExecLayoutFamily<B>
        + MatZnxAlloc,
    B: ScratchTakeVecZnxDftImpl<B> + ScratchTakeVecZnxBigImpl<B> + ScratchTakeSvpPPolImpl<B>,
{
    let rows: usize = k_ct.div_ceil(basek * digits);

    let mut autokey: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc(module, basek, k_ksk, rows, digits, rank);
    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, basek, k_ct, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k_ct);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    module.vec_znx_fill_uniform(basek, &mut pt_want.data, 0, k_ct, &mut source_xa);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        AutomorphismKey::encrypt_sk_scratch_space(module, basek, autokey.k(), rank)
            | GLWECiphertext::decrypt_scratch_space(module, basek, ct.k())
            | GLWECiphertext::encrypt_sk_scratch_space(module, basek, ct.k())
            | GLWECiphertext::automorphism_inplace_scratch_space(module, basek, ct.k(), autokey.k(), digits, rank),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_exec: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk);

    autokey.encrypt_sk(
        module,
        p,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct.encrypt_sk(
        module,
        &pt_want,
        &sk_exec,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut autokey_exec: AutomorphismKeyExec<Vec<u8>, B> = AutomorphismKeyExec::alloc(module, basek, k_ksk, rows, digits, rank);
    autokey_exec.prepare(module, &autokey, scratch.borrow());

    ct.automorphism_inplace(module, &autokey_exec, scratch.borrow());

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

    module.vec_znx_automorphism_inplace(p, &mut pt_want.data, 0);

    ct.assert_noise(module, &sk_exec, &pt_want, max_noise + 1.0);
}
