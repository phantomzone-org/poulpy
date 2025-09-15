use poulpy_backend::cpu_spqlios::FFT64Spqlios;
use poulpy_core::{
    GLWEOperations, SIGMA,
    layouts::{
        GLWECiphertext, GLWEPlaintext, GLWESecret, Infos,
        prepared::{GLWESecretPrepared, PrepareAlloc},
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniform},
    layouts::{Module, ScratchOwned},
    source::Source,
};

fn main() {
    // Ring degree
    let log_n: usize = 10;

    let n: usize = 1 << log_n;

    // Base-2-k (implicit digit decomposition)
    let basek: usize = 14;

    // Ciphertext Torus precision (equivalent to ciphertext modulus)
    let k_ct: usize = 27;

    // Plaintext Torus precision (equivament to plaintext modulus)
    let k_pt: usize = basek;

    // GLWE rank
    let rank: usize = 1;

    // Instantiate Module (DFT Tables)
    let module: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(n as u64);

    // Allocates ciphertext & plaintexts
    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_ct, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_pt);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_pt);

    // CPRNG
    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);

    // Scratch space
    let mut scratch: ScratchOwned<FFT64Spqlios> = ScratchOwned::alloc(
        GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct.k())
            | GLWECiphertext::decrypt_scratch_space(&module, basek, ct.k()),
    );

    // Generate secret-key
    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    // Backend-prepared secret
    let sk_prepared: GLWESecretPrepared<Vec<u8>, FFT64Spqlios> = sk.prepare_alloc(&module, scratch.borrow());

    // Uniform plaintext
    module.vec_znx_fill_uniform(basek, &mut pt_want.data, 0, &mut source_xa);

    // Encryption
    ct.encrypt_sk(
        &module,
        &pt_want,
        &sk_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    // Decryption
    ct.decrypt(&module, &mut pt_have, &sk_prepared, scratch.borrow());

    // Diff between pt - Dec(Enc(pt))
    pt_want.sub_inplace_ab(&module, &pt_have);

    // Ideal vs. actual noise
    let noise_have: f64 = pt_want.data.std(basek, 0) * (ct.k() as f64).exp2();
    let noise_want: f64 = SIGMA;

    // Check
    assert!(noise_have <= noise_want + 0.2);
}
