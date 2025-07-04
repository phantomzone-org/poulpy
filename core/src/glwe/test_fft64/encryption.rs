use backend::{Decoding, Encoding, FFT64, Module, ScratchOwned, Stats, VecZnxOps, ZnxZero};
use itertools::izip;
use sampling::source::Source;

use crate::{FourierGLWECiphertext, FourierGLWESecret, GLWECiphertext, GLWEPlaintext, GLWEPublicKey, GLWESecret, Infos};

#[test]
fn encrypt_sk() {
    let log_n: usize = 8;
    (1..4).for_each(|rank| {
        [2, 10, 30].iter().for_each(|k_pt| {
            println!("test encrypt_sk rank: {}, k_pt: {k_pt}", rank);
            test_encrypt_sk(log_n, 8, 54, *k_pt, 3.2, rank);
        });
    });
}

#[test]
fn encrypt_zero_sk() {
    let log_n: usize = 8;
    (1..4).for_each(|rank| {
        println!("test encrypt_zero_sk rank: {}", rank);
        test_encrypt_zero_sk(log_n, 8, 64, 3.2, rank);
    });
}

#[test]
fn encrypt_pk() {
    let log_n: usize = 8;
    (1..4).for_each(|rank| {
        println!("test encrypt_pk rank: {}", rank);
        test_encrypt_pk(log_n, 8, 64, 64, 3.2, rank)
    });
}

fn test_encrypt_sk(log_n: usize, basek: usize, k_ct: usize, k_pt: usize, sigma: f64, rank: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_ct, rank);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_pt);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct.k())
            | GLWECiphertext::decrypt_scratch_space(&module, basek, ct.k()),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk);

    let mut data_want: Vec<i64> = vec![0i64; module.n()];
    if k_pt < 64 {
        let pt_max = 1 << k_pt;
        data_want.iter_mut().for_each(|x| {
            let v = source_xa.next_u64n(pt_max, pt_max - 1);
            *x = if v >= pt_max / 2 {
                -((pt_max - v) as i64)
            } else {
                v as i64
            };
        });
    } else {
        data_want.iter_mut().for_each(|x| *x = source_xa.next_i64());
    }
    pt.data
        .encode_vec_i64(0, basek, k_pt, &data_want, std::cmp::min(k_pt, 64));

    ct.encrypt_sk(
        &module,
        &pt,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    pt.data.zero();

    ct.decrypt(&module, &mut pt, &sk_dft, scratch.borrow());

    let mut data_have: Vec<i64> = vec![0i64; module.n()];

    pt.data
        .decode_vec_i64(0, basek, pt.size() * basek, &mut data_have);

    // TODO: properly assert the decryption noise through std(dec(ct) - pt)
    let scale: f64 = (1 << (pt.size() * basek - k_pt)) as f64;
    izip!(data_want.iter(), data_have.iter()).for_each(|(a, b)| {
        let b_scaled = (*b as f64) / scale;
        assert!(
            (*a as f64 - b_scaled).abs() < 0.1,
            "a={} b={}",
            *a as f64,
            b_scaled
        )
    });
}

fn test_encrypt_zero_sk(log_n: usize, basek: usize, k_ct: usize, sigma: f64, rank: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk);

    let mut ct_dft: FourierGLWECiphertext<Vec<u8>, FFT64> = FourierGLWECiphertext::alloc(&module, basek, k_ct, rank);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        FourierGLWECiphertext::decrypt_scratch_space(&module, basek, k_ct)
            | FourierGLWECiphertext::encrypt_sk_scratch_space(&module, basek, k_ct, rank),
    );

    ct_dft.encrypt_zero_sk(
        &module,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );
    ct_dft.decrypt(&module, &mut pt, &sk_dft, scratch.borrow());

    assert!((sigma - pt.data.std(0, basek) * (k_ct as f64).exp2()) <= 0.2);
}

fn test_encrypt_pk(log_n: usize, basek: usize, k_ct: usize, k_pk: usize, sigma: f64, rank: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_ct, rank);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xu: Source = Source::new([0u8; 32]);

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk);

    let mut pk: GLWEPublicKey<Vec<u8>, FFT64> = GLWEPublicKey::alloc(&module, basek, k_pk, rank);
    pk.generate_from_sk(&module, &sk_dft, &mut source_xa, &mut source_xe, sigma);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct.k())
            | GLWECiphertext::decrypt_scratch_space(&module, basek, ct.k())
            | GLWECiphertext::encrypt_pk_scratch_space(&module, basek, pk.k()),
    );

    let mut data_want: Vec<i64> = vec![0i64; module.n()];

    data_want
        .iter_mut()
        .for_each(|x| *x = source_xa.next_i64() & 0);

    pt_want.data.encode_vec_i64(0, basek, k_ct, &data_want, 10);

    ct.encrypt_pk(
        &module,
        &pt_want,
        &pk,
        &mut source_xu,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);

    ct.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());

    module.vec_znx_sub_ab_inplace(&mut pt_want.data, 0, &pt_have.data, 0);

    let noise_have: f64 = pt_want.data.std(0, basek).log2();
    let noise_want: f64 = ((((rank as f64) + 1.0) * module.n() as f64 * 0.5 * sigma * sigma).sqrt()).log2() - (k_ct as f64);

    assert!(
        (noise_have - noise_want).abs() < 0.2,
        "{} {}",
        noise_have,
        noise_want
    );
}
