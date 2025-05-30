use crate::{AutomorphismKey, GLWECiphertext, GLWEOps, GLWEPlaintext, GLWESecret, StreamPacker};
use std::collections::HashMap;

use backend::{Encoding, FFT64, Module, ScratchOwned, Stats};
use sampling::source::Source;

#[test]
fn packing() {
    let log_n: usize = 5;
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let basek: usize = 18;
    let ct_k: usize = 36;
    let atk_k: usize = ct_k + basek;
    let pt_k: usize = 18;
    let rank: usize = 3;
    let rows: usize = (ct_k + basek - 1) / basek;
    let sigma: f64 = 3.2;

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct_k)
            | GLWECiphertext::decrypt_scratch_space(&module, basek, ct_k)
            | AutomorphismKey::generate_from_sk_scratch_space(&module, basek, atk_k, rank)
            | StreamPacker::scratch_space(&module, basek, ct_k, atk_k, rank),
    );

    let mut sk: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(&module, 0.5, &mut source_xs);

    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, ct_k);
    let mut data: Vec<i64> = vec![0i64; module.n()];
    data.iter_mut().enumerate().for_each(|(i, x)| {
        *x = i as i64;
    });
    pt.data.encode_vec_i64(0, basek, pt_k, &data, 32);

    let gal_els: Vec<i64> = StreamPacker::galois_elements(&module);

    let mut auto_keys: HashMap<i64, AutomorphismKey<Vec<u8>, FFT64>> = HashMap::new();
    gal_els.iter().for_each(|gal_el| {
        let mut key: AutomorphismKey<Vec<u8>, FFT64> = AutomorphismKey::alloc(&module, basek, atk_k, rows, rank);
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

    let mut packer: StreamPacker = StreamPacker::new(&module, log_batch, basek, ct_k, rank);

    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, ct_k, rank);

    ct.encrypt_sk(
        &module,
        &pt,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut res: Vec<GLWECiphertext<Vec<u8>>> = Vec::new();

    (0..module.n() >> log_batch).for_each(|i| {
        ct.encrypt_sk(
            &module,
            &pt,
            &sk,
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

    packer.flush(&module, &mut res, &auto_keys, scratch.borrow());
    packer.reset();

    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, ct_k);

    res.iter().enumerate().for_each(|(i, res_i)| {
        let mut data: Vec<i64> = vec![0i64; module.n()];
        data.iter_mut().enumerate().for_each(|(i, x)| {
            if i % 5 == 0 {
                *x = reverse_bits_msb(i, log_n as u32) as i64;
            }
        });
        pt_want.data.encode_vec_i64(0, basek, pt_k, &data, 32);

        res_i.decrypt(&module, &mut pt, &sk, scratch.borrow());

        if i & 1 == 0 {
            pt.sub_inplace_ab(&module, &pt_want);
        } else {
            pt.add_inplace(&module, &pt_want);
        }

        let noise_have = pt.data.std(0, basek).log2();
        // println!("noise_have: {}", noise_have);
        assert!(
            noise_have < -((ct_k - basek) as f64),
            "noise: {}",
            noise_have
        );
    });
}

#[inline(always)]
fn reverse_bits_msb(x: usize, n: u32) -> usize {
    x.reverse_bits() >> (usize::BITS - n)
}
