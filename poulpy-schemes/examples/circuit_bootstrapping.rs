use poulpy_core::{
    GLWEOperations,
    layouts::{
        GGSWCiphertext, GLWECiphertext, GLWEPlaintext, GLWESecret, Infos, LWECiphertext, LWEPlaintext, LWESecret,
        prepared::{GGSWCiphertextPrepared, GLWESecretPrepared, PrepareAlloc},
    },
};
use std::time::Instant;

use poulpy_backend::{
    hal::{
        api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalizeInplace, ZnxView, ZnxViewMut},
        layouts::{Module, ScalarZnx, ScratchOwned},
        source::Source,
    },
    implementation::cpu_spqlios::FFT64,
};

use poulpy_schemes::tfhe::{
    blind_rotation::CGGI,
    circuit_bootstrapping::{
        CircuitBootstrappingKey, CircuitBootstrappingKeyEncryptSk, CircuitBootstrappingKeyPrepared, CirtuitBootstrappingExecute,
    },
};

fn main() {
    // GLWE ring degree
    let n_glwe: usize = 1024;

    // Module provides access to the backend arithmetic
    let module: Module<FFT64> = Module::<FFT64>::new(n_glwe as u64);

    // Base 2 loga
    let basek: usize = 13;

    // Lookup table extension factor
    let extension_factor: usize = 1;

    // GLWE rank
    let rank: usize = 1;

    // Noise (discrete) standard deviation
    let sigma: f64 = 3.2;

    // LWE degree
    let n_lwe: usize = 574;

    // LWE plaintext modulus
    let k_lwe_pt: usize = 1;

    // LWE ciphertext modulus
    let k_lwe_ct: usize = 13;

    // LWE block binary key block size
    let block_size: usize = 7;

    // GGSW output number of rows
    let rows_ggsw_res: usize = 2;

    // GGSW output modulus
    let k_ggsw_res: usize = (rows_ggsw_res + 1) * basek;

    // Blind rotation key GGSW number of rows
    let rows_brk: usize = rows_ggsw_res + 1;

    // Blind rotation key GGSW modulus
    let k_brk: usize = (rows_brk + 1) * basek;

    // GGLWE automorphism keys number of rows
    let rows_trace: usize = rows_ggsw_res + 1;

    // GGLWE automorphism keys modulus
    let k_trace: usize = (rows_trace + 1) * basek;

    // GGLWE tensor key number of rows
    let rows_tsk: usize = rows_ggsw_res + 1;

    // GGLWE tensor key modulus
    let k_tsk: usize = (rows_tsk + 1) * basek;

    // Scratch space (4MB)
    let mut scratch: ScratchOwned<FFT64> = ScratchOwned::alloc(1 << 22);

    // Secret key sampling source
    let mut source_xs: Source = Source::new([1u8; 32]);

    // Public randomness sampling source
    let mut source_xa: Source = Source::new([1u8; 32]);

    // Noise sampling source
    let mut source_xe: Source = Source::new([1u8; 32]);

    // LWE secret
    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
    sk_lwe.fill_binary_block(block_size, &mut source_xs);
    sk_lwe.fill_zero();

    // GLWE secret
    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(n_glwe, rank);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);
    // sk_glwe.fill_zero();

    // GLWE secret prepared (opaque backend dependant write only struct)
    let sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, FFT64> = sk_glwe.prepare_alloc(&module, scratch.borrow());

    // Plaintext value to circuit bootstrap
    let data: i64 = 1 % (1 << k_lwe_pt);

    // LWE plaintext
    let mut pt_lwe: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(basek, k_lwe_pt);

    // LWE plaintext(data * 2^{- (k_lwe_pt - 1)})
    pt_lwe.encode_i64(data, k_lwe_pt + 1); // +1 for padding bit
    module.vec_znx_normalize_inplace(basek, pt_lwe.data_mut(), 0, scratch.borrow());

    println!("pt_lwe: {}", pt_lwe);

    // LWE ciphertext
    let mut ct_lwe: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(n_lwe, basek, k_lwe_ct);

    // Encrypt LWE Plaintext
    ct_lwe.encrypt_sk(
        &module,
        &pt_lwe,
        &sk_lwe,
        &mut source_xa,
        &mut source_xe,
        sigma,
    );

    let now: Instant = Instant::now();

    // Circuit bootstrapping evaluation key
    let cbt_key: CircuitBootstrappingKey<Vec<u8>, CGGI> = CircuitBootstrappingKey::encrypt_sk(
        &module,
        basek,
        &sk_lwe,
        &sk_glwe,
        k_brk,
        rows_brk,
        k_trace,
        rows_trace,
        k_tsk,
        rows_tsk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );
    println!("CBT-KGEN: {} ms", now.elapsed().as_millis());

    // Output GGSW
    let mut res: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n_glwe, basek, k_ggsw_res, rows_ggsw_res, 1, rank);

    // Circuit bootstrapping key prepared (opaque backend dependant write only struct)
    let cbt_prepared: CircuitBootstrappingKeyPrepared<Vec<u8>, CGGI, FFT64> = cbt_key.prepare_alloc(&module, scratch.borrow());

    // Apply circuit bootstrapping: LWE(data * 2^{- (k_lwe_pt + 2)}) -> GGSW(data)
    let now: Instant = Instant::now();
    cbt_prepared.execute_to_constant(
        &module,
        &mut res,
        &ct_lwe,
        k_lwe_pt,
        extension_factor,
        scratch.borrow(),
    );
    println!("CBT: {} ms", now.elapsed().as_millis());

    // Allocate "ideal" GGSW(data) plaintext
    let mut pt_ggsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n_glwe, 1);
    pt_ggsw.at_mut(0, 0)[0] = data;

    // Prints noise of GGSW(data)
    res.print_noise(&module, &sk_glwe_prepared, &pt_ggsw);

    // Tests RLWE(1) * GGSW(data)

    // GLWE ciphertext modulus
    let mut ct_glwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n_glwe, basek, k_ggsw_res - basek, rank);

    // Some GLWE plaintext with signed data
    let k_glwe_pt: usize = 3;
    let mut pt_glwe: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n_glwe, basek, basek);
    let mut data_vec: Vec<i64> = vec![0i64; n_glwe];
    data_vec
        .iter_mut()
        .enumerate()
        .for_each(|(x, y)| *y = (x % (1 << (k_glwe_pt - 1))) as i64 - (1 << (k_glwe_pt - 2)));

    pt_glwe.encode_vec_i64(&data_vec, k_lwe_pt + 2);
    pt_glwe.normalize_inplace(&module, scratch.borrow());

    println!("{}", pt_glwe);

    // Encrypt
    ct_glwe.encrypt_sk(
        &module,
        &pt_glwe,
        &sk_glwe_prepared,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    // Prepare GGSW output of circuit bootstrapping (opaque backend dependant write only struct)
    let res_prepared: GGSWCiphertextPrepared<Vec<u8>, FFT64> = res.prepare_alloc(&module, scratch.borrow());

    // Apply GLWE x GGSW
    ct_glwe.external_product_inplace(&module, &res_prepared, scratch.borrow());

    // Decrypt
    let mut pt_res: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n_glwe, basek, ct_glwe.k());
    ct_glwe.decrypt(&module, &mut pt_res, &sk_glwe_prepared, scratch.borrow());

    println!("pt_res: {:?}", &pt_res.data.at(0, 0)[..64]);
}
