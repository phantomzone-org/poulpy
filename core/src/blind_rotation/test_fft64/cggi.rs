use std::time::Instant;

use backend::{Encoding, FFT64, Module, ScratchOwned, Stats, VecZnxOps, ZnxView};
use sampling::source::Source;

use crate::{
    FourierGLWESecret, GLWECiphertext, GLWEPlaintext, GLWESecret, Infos, LWECiphertext, LWESecret,
    blind_rotation::{
        ccgi::{cggi_blind_rotate, cggi_blind_rotate_scratch_space, mod_switch_2n},
        key::BlindRotationKeyCGGI,
        lut::LookUpTable,
    },
    lwe::{LWEPlaintext, ciphertext::LWECiphertextToRef},
};

#[test]
fn blind_rotation() {
    let module: Module<FFT64> = Module::<FFT64>::new(2048);
    let basek: usize = 20;

    let n_lwe: usize = 1071;

    let k_lwe: usize = 22;
    let k_brk: usize = 60;
    let rows_brk: usize = 2;
    let k_lut: usize = 60;
    let rank: usize = 1;
    let block_size: usize = 7;

    let message_modulus: usize = 64;

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);
    let sk_glwe_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk_glwe);

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
    sk_lwe.fill_binary_block(block_size, &mut source_xs);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        BlindRotationKeyCGGI::generate_from_sk_scratch_space(&module, basek, k_brk, rank)
            | cggi_blind_rotate_scratch_space(&module, basek, k_lut, k_brk, rows_brk, rank),
    );

    let start: Instant = Instant::now();
    let mut brk: BlindRotationKeyCGGI<FFT64> = BlindRotationKeyCGGI::allocate(&module, n_lwe, basek, k_brk, rows_brk, rank);

    brk.generate_from_sk(
        &module,
        &sk_glwe_dft,
        &sk_lwe,
        &mut source_xa,
        &mut source_xe,
        3.2,
        scratch.borrow(),
    );

    let duration: std::time::Duration = start.elapsed();
    println!("brk-gen: {} ms", duration.as_millis());

    let mut lwe: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(n_lwe, basek, k_lwe);

    let mut pt_lwe: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(basek, k_lwe);

    let x: i64 = 0;
    let bits: usize = 6;

    pt_lwe.data.encode_coeff_i64(0, basek, bits, 0, x, bits);

    println!("{}", pt_lwe.data);

    lwe.encrypt_sk(&pt_lwe, &sk_lwe, &mut source_xa, &mut source_xe, 3.2);

    lwe.decrypt(&mut pt_lwe, &sk_lwe);

    println!("{}", pt_lwe.data);

    fn lut_fn(x: i64) -> i64 {
        2 * x + 1
    }

    let mut lut: LookUpTable = LookUpTable::alloc(&module, basek, k_lut, 1);
    lut.set(&module, lut_fn, message_modulus);

    let mut res: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_lut, rank);

    let start: Instant = Instant::now();
    (0..1).for_each(|_| {
        cggi_blind_rotate(&module, &mut res, &lwe, &lut, &brk, scratch.borrow());
    });

    let duration: std::time::Duration = start.elapsed();
    println!("blind-rotate: {} ms", duration.as_millis());

    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_lut);

    res.decrypt(&module, &mut pt_have, &sk_glwe_dft, scratch.borrow());

    println!("pt_have: {}", pt_have.data);

    let mut lwe_2n: Vec<i64> = vec![0i64; lwe.n() + 1]; // TODO: from scratch space

    mod_switch_2n(&module, &mut lwe_2n, &lwe.to_ref());

    let pt_want: i64 = (lwe_2n[0]
        + lwe_2n[1..]
            .iter()
            .zip(sk_lwe.data.at(0, 0))
            .map(|(x, y)| x * y)
            .sum::<i64>())
        % (module.n() as i64 * 2);

    module.vec_znx_rotate_inplace(pt_want, &mut lut.data[0], 0);

    println!("pt_want: {}", lut.data[0]);

    module.vec_znx_sub_ab_inplace(&mut lut.data[0], 0, &pt_have.data, 0);

    let noise: f64 = lut.data[0].std(0, basek);

    println!("noise: {}", noise);
}
