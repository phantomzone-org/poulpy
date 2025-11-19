use poulpy_core::{
    GLWENormalize,
    layouts::{
        GGLWEToGGSWKeyLayout, GGSW, GGSWInfos, GGSWLayout, GLWE, GLWEAutomorphismKeyLayout, GLWEInfos, GLWELayout, GLWEPlaintext,
        GLWESecret, LWE, LWELayout, LWEPlaintext, LWESecret,
        prepared::{GGSWPrepared, GLWESecretPrepared},
    },
};
use std::time::Instant;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use poulpy_backend::FFT64Avx as BackendImpl;

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
use poulpy_backend::FFT64Ref as BackendImpl;

use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalizeInplace},
    layouts::{Module, ScalarZnx, ScratchOwned, ZnxView, ZnxViewMut},
    source::Source,
};

use poulpy_schemes::tfhe::{
    blind_rotation::{BlindRotationKeyLayout, CGGI},
    circuit_bootstrapping::{CircuitBootstrappingKey, CircuitBootstrappingKeyLayout, CircuitBootstrappingKeyPrepared},
};

fn main() {
    // GLWE ring degree
    let n_glwe: usize = 1024;

    // Module provides access to the backend arithmetic
    let module: Module<BackendImpl> = Module::<BackendImpl>::new(n_glwe as u64);

    // Base 2 loga
    let base2k: usize = 13;

    // Lookup table extension factor
    let extension_factor: usize = 1;

    // GLWE rank
    let rank: usize = 1;

    // LWE degree
    let n_lwe: usize = 574;

    // LWE plaintext modulus
    let k_lwe_pt: usize = 1;

    // LWE ciphertext modulus
    let k_lwe_ct: usize = 13;

    // LWE block binary key block size
    let block_size: usize = 7;

    // GGSW output number of dnum
    let rows_ggsw_res: usize = 2;

    // GGSW output modulus
    let k_ggsw_res: usize = (rows_ggsw_res + 1) * base2k;

    // Blind rotation key GGSW number of dnum
    let rows_brk: usize = rows_ggsw_res + 1;

    // Blind rotation key GGSW modulus
    let k_brk: usize = (rows_brk + 1) * base2k;

    // GGLWE automorphism keys number of dnum
    let rows_trace: usize = rows_ggsw_res + 1;

    // GGLWE automorphism keys modulus
    let k_trace: usize = (rows_trace + 1) * base2k;

    // GGLWE tensor key number of dnum
    let rows_tsk: usize = rows_ggsw_res + 1;

    // GGLWE tensor key modulus
    let k_tsk: usize = (rows_tsk + 1) * base2k;

    let cbt_infos: CircuitBootstrappingKeyLayout = CircuitBootstrappingKeyLayout {
        layout_brk: BlindRotationKeyLayout {
            n_glwe: n_glwe.into(),
            n_lwe: n_lwe.into(),
            base2k: base2k.into(),
            k: k_brk.into(),
            dnum: rows_brk.into(),
            rank: rank.into(),
        },
        layout_atk: GLWEAutomorphismKeyLayout {
            n: n_glwe.into(),
            base2k: base2k.into(),
            k: k_trace.into(),
            dnum: rows_trace.into(),
            dsize: 1_u32.into(),
            rank: rank.into(),
        },
        layout_tsk: GGLWEToGGSWKeyLayout {
            n: n_glwe.into(),
            base2k: base2k.into(),
            k: k_tsk.into(),
            dnum: rows_tsk.into(),
            dsize: 1_u32.into(),
            rank: rank.into(),
        },
    };

    let ggsw_infos: GGSWLayout = GGSWLayout {
        n: n_glwe.into(),
        base2k: base2k.into(),
        k: k_ggsw_res.into(),
        dnum: rows_ggsw_res.into(),
        dsize: 1_u32.into(),
        rank: rank.into(),
    };

    let lwe_infos = LWELayout {
        n: n_lwe.into(),
        k: k_lwe_ct.into(),
        base2k: base2k.into(),
    };

    // Scratch space (4MB)
    let mut scratch: ScratchOwned<BackendImpl> = ScratchOwned::alloc(1 << 22);

    // Secret key sampling source
    let mut source_xs: Source = Source::new([1u8; 32]);

    // Public randomness sampling source
    let mut source_xa: Source = Source::new([1u8; 32]);

    // Noise sampling source
    let mut source_xe: Source = Source::new([1u8; 32]);

    // LWE secret
    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe.into());
    sk_lwe.fill_binary_block(block_size, &mut source_xs);
    sk_lwe.fill_zero();

    // GLWE secret
    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(n_glwe.into(), rank.into());
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);
    // sk_glwe.fill_zero();

    // GLWE secret prepared (opaque backend dependant write only struct)
    let mut sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, BackendImpl> = GLWESecretPrepared::alloc(&module, rank.into());
    sk_glwe_prepared.prepare(&module, &sk_glwe);

    // Plaintext value to circuit bootstrap
    let data: i64 = 1 % (1 << k_lwe_pt);

    // LWE plaintext
    let mut pt_lwe: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(base2k.into(), k_lwe_pt.into());

    // LWE plaintext(data * 2^{- (k_lwe_pt - 1)})
    pt_lwe.encode_i64(data, (k_lwe_pt + 1).into()); // +1 for padding bit

    // Normalize plaintext to nicely print coefficients
    module.vec_znx_normalize_inplace(base2k, pt_lwe.data_mut(), 0, scratch.borrow());
    println!("pt_lwe: {pt_lwe}");

    // LWE ciphertext
    let mut ct_lwe: LWE<Vec<u8>> = LWE::alloc_from_infos(&lwe_infos);

    // Encrypt LWE Plaintext
    ct_lwe.encrypt_sk(
        &module,
        &pt_lwe,
        &sk_lwe,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let now: Instant = Instant::now();

    // Circuit bootstrapping evaluation key
    let mut cbt_key: CircuitBootstrappingKey<Vec<u8>, CGGI> = CircuitBootstrappingKey::alloc_from_infos(&cbt_infos);

    cbt_key.encrypt_sk(
        &module,
        &sk_lwe,
        &sk_glwe,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    println!("CBT-KGEN: {} ms", now.elapsed().as_millis());

    // Output GGSW
    let mut res: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_infos);

    // Circuit bootstrapping key prepared (opaque backend dependant write only struct)
    let mut cbt_prepared: CircuitBootstrappingKeyPrepared<Vec<u8>, CGGI, BackendImpl> =
        CircuitBootstrappingKeyPrepared::alloc_from_infos(&module, &cbt_infos);
    cbt_prepared.prepare(&module, &cbt_key, scratch.borrow());

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
    for row in 0..res.dnum().as_usize() {
        for col in 0..res.rank().as_usize() + 1 {
            println!(
                "row:{row} col:{col} -> {}",
                res.noise(
                    &module,
                    row,
                    col,
                    &pt_ggsw,
                    &sk_glwe_prepared,
                    scratch.borrow()
                )
                .std()
                .log2()
            )
        }
    }

    // Tests RLWE(1) * GGSW(data)

    let glwe_infos: GLWELayout = GLWELayout {
        n: n_glwe.into(),
        base2k: base2k.into(),
        k: (k_ggsw_res - base2k).into(),
        rank: rank.into(),
    };

    // GLWE ciphertext modulus
    let mut ct_glwe: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_infos);

    // Some GLWE plaintext with signed data
    let k_glwe_pt: usize = 3;
    let mut pt_glwe: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_infos);
    let mut data_vec: Vec<i64> = vec![0i64; n_glwe];
    data_vec
        .iter_mut()
        .enumerate()
        .for_each(|(x, y)| *y = (x % (1 << (k_glwe_pt - 1))) as i64 - (1 << (k_glwe_pt - 2)));

    pt_glwe.encode_vec_i64(&data_vec, (k_lwe_pt + 2).into());
    module.glwe_normalize_inplace(&mut pt_glwe, scratch.borrow());

    println!("{}", pt_glwe);

    // Encrypt
    ct_glwe.encrypt_sk(
        &module,
        &pt_glwe,
        &sk_glwe_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    // Prepare GGSW output of circuit bootstrapping (opaque backend dependant write only struct)
    let mut res_prepared: GGSWPrepared<Vec<u8>, BackendImpl> = GGSWPrepared::alloc_from_infos(&module, &res);
    res_prepared.prepare(&module, &res, scratch.borrow());

    // Apply GLWE x GGSW
    ct_glwe.external_product_inplace(&module, &res_prepared, scratch.borrow());

    // Decrypt
    let mut pt_res: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_infos);
    ct_glwe.decrypt(&module, &mut pt_res, &sk_glwe_prepared, scratch.borrow());

    println!("pt_res: {:?}", &pt_res.data.at(0, 0)[..64]);
}
