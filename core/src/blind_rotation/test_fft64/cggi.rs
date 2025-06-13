use core::time;
use std::time::Instant;

use backend::{Encoding, Module, ScratchOwned, FFT64};
use sampling::source::Source;

use crate::{
    blind_rotation::{ccgi::{cggi_blind_rotate, cggi_blind_rotate_scratch_space}, key::BlindRotationKeyCGGI}, lwe::LWEPlaintext, FourierGLWESecret, GLWECiphertext, GLWEPlaintext, GLWESecret, LWECiphertext, LWESecret
};

#[test]
fn blind_rotation() {
    let module: Module<FFT64> = Module::<FFT64>::new(2048);
    let basek: usize = 17;

    let n_lwe: usize = 1071;

    let k_lwe: usize = 22;
    let k_brk: usize = 54;
    let rows_brk: usize = 1;
    let k_lut: usize = 44;
    let rank: usize = 1;
    let block_size: usize = 7;

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);
    let sk_glwe_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk_glwe);

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
    sk_lwe.fill_binary_block(block_size, &mut source_xs);

    let mut scratch: ScratchOwned = ScratchOwned::new(BlindRotationKeyCGGI::generate_from_sk_scratch_space(
        &module, basek, k_brk, rank,
    ) | cggi_blind_rotate_scratch_space(&module, basek, k_lut, k_brk, rows_brk, rank));

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

    pt_lwe.data.encode_coeff_i64(0, basek, 7, 0, 63, 7);

    println!("{}", pt_lwe.data);

    lwe.encrypt_sk(&pt_lwe, &sk_lwe, &mut source_xa, &mut source_xe, 3.2);

    lwe.decrypt(&mut pt_lwe, &sk_lwe);

    println!("{}", pt_lwe.data);

    let lut: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_lut);

    let mut res: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_lut, rank);

    let start: Instant = Instant::now();
    (0..32).for_each(|i|{
        cggi_blind_rotate(&module, &mut res, &lwe, &lut, &brk, scratch.borrow());
    });
    
    let duration: std::time::Duration = start.elapsed();
    println!("blind-rotate: {} ms", duration.as_millis());

    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_lut);

    res.decrypt(&module , &mut pt, &sk_glwe_dft, scratch.borrow());

    println!("{}", pt.data);
}
