use crate::{
    automorphism::AutomorphismKey,
    glwe_ciphertext::GLWECiphertext,
    glwe_ops::GLWEOps,
    glwe_packing::StreamPacker,
    glwe_plaintext::GLWEPlaintext,
    keys::{SecretKey, SecretKeyFourier},
};
use std::collections::HashMap;

use backend::{Encoding, FFT64, Module, ScratchOwned, Stats};
use sampling::source::Source;
use std::time::Instant;

#[test]
fn packing() {
    let log_n: usize = 5;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let basek: usize = 18;
    let k_ct: usize = 36;
    let k_auto_key: usize = k_ct + basek;
    let k_pt: usize = 18;
    let rank: usize = 3;
    let rows: usize = (k_ct + basek - 1) / basek;
    let sigma: f64 = 3.2;
    let ct_size: usize = rows;
    let auto_key_size: usize = (k_auto_key + basek - 1) / basek;

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWECiphertext::encrypt_sk_scratch_space(&module, ct_size)
            | GLWECiphertext::decrypt_scratch_space(&module, ct_size)
            | AutomorphismKey::generate_from_sk_scratch_space(&module, rank, auto_key_size)
            | StreamPacker::scratch_space(&module, ct_size, auto_key_size, rank),
    );

    let mut sk: SecretKey<Vec<u8>> = SecretKey::alloc(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::alloc(&module, rank);
    sk_dft.dft(&module, &sk);

    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);
    let mut data: Vec<i64> = vec![0i64; module.n()];
    data.iter_mut().enumerate().for_each(|(i, x)| {
        *x = i as i64;
    });
    pt.data.encode_vec_i64(0, basek, k_pt, &data, 32);

    let gal_els: Vec<i64> = StreamPacker::galois_elements(&module);

    let mut auto_keys: HashMap<i64, AutomorphismKey<Vec<u8>, FFT64>> = HashMap::new();
    gal_els.iter().for_each(|gal_el| {
        let mut key: AutomorphismKey<Vec<u8>, FFT64> = AutomorphismKey::alloc(&module, basek, k_auto_key, rows, rank);
        key.generate_from_sk(
            &module,
            *gal_el,
            &sk,
            &mut source_xa,
            &mut source_xe,
            sigma,
            scratch.borrow(),
        );
        auto_keys.insert(*gal_el, key);
    });

    let log_batch: usize = 0;

    let mut packer: StreamPacker = StreamPacker::new(&module, log_batch, basek, k_ct, rank);

    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_ct, rank);

    ct.encrypt_sk(
        &module,
        &pt,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut res: Vec<GLWECiphertext<Vec<u8>>> = Vec::new();

    let start = Instant::now();
    (0..module.n() >> log_batch).for_each(|i| {
        println!("pt {}", pt.data);
        ct.encrypt_sk(
            &module,
            &pt,
            &sk_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            scratch.borrow(),
        );

        pt.rotate_inplace(&module, -(1 << log_batch)); // X^-batch * pt

        if reverse_bits_msb(i, log_n as u32) % 5 == 0 {
            packer.add(&module, &mut res, Some(&ct), &auto_keys, scratch.borrow());
        } else {
            packer.add(
                &module,
                &mut res,
                None::<&GLWECiphertext<Vec<u8>>>,
                &auto_keys,
                scratch.borrow(),
            )
        }
    });
    let duration = start.elapsed();
    println!("Elapsed time: {} ms", duration.as_millis());

    packer.flush(&module, &mut res, &auto_keys, scratch.borrow());
    packer.reset();

    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);

    println!("{}", res.len());

    res.iter().enumerate().for_each(|(i, res_i)| {
        let mut data: Vec<i64> = vec![0i64; module.n()];
        data.iter_mut().enumerate().for_each(|(i, x)| {
            if i % 5 == 0 {
                *x = reverse_bits_msb(i, log_n as u32) as i64;
            }
        });
        pt_want.data.encode_vec_i64(0, basek, k_pt, &data, 32);

        res_i.decrypt(&module, &mut pt, &sk_dft, scratch.borrow());

        println!("{}", pt.data);

        if i & 1 == 0 {
            pt.sub_inplace_ab(&module, &pt_want);
        } else {
            pt.add_inplace(&module, &pt_want);
        }

        let noise_have = pt.data.std(0, basek).log2();
        println!("noise_have: {}", noise_have);
        assert!(
            noise_have < -((k_ct - basek) as f64),
            "noise: {}",
            noise_have
        );
    });
}

#[inline(always)]
fn reverse_bits_msb(x: usize, n: u32) -> usize {
    x.reverse_bits() >> (usize::BITS - n)
}
