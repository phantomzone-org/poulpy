use backend::{Encoding, FFT64, Module, ScratchOwned, ZnxView};
use sampling::source::Source;

use crate::{
    FourierGLWESecret, GLWECiphertext, GLWEPlaintext, GLWESecret, Infos, LWECiphertext, LWESecret,
    blind_rotation::{
        cggi::{cggi_blind_rotate, cggi_blind_rotate_scratch_space, mod_switch_2n},
        key::BlindRotationKeyCGGI,
        lut::LookUpTable,
    },
    lwe::{LWEPlaintext, ciphertext::LWECiphertextToRef},
};

#[test]
fn standard() {
    blind_rotatio_test(224, 1, 1);
}

#[test]
fn block_binary() {
    blind_rotatio_test(224, 7, 1);
}

#[test]
fn block_binary_extended() {
    blind_rotatio_test(224, 7, 2);
}

fn blind_rotatio_test(n_lwe: usize, block_size: usize, extension_factor: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(512);
    let basek: usize = 19;

    let k_lwe: usize = 24;
    let k_brk: usize = 3 * basek;
    let rows_brk: usize = 2; // Ensures first limb is noise-free.
    let k_lut: usize = 1 * basek;
    let k_res: usize = 2 * basek;
    let rank: usize = 1;

    let message_modulus: usize = 1 << 4;

    let mut source_xs: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([2u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);
    let sk_glwe_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk_glwe);

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
    sk_lwe.fill_binary_block(block_size, &mut source_xs);

    let mut scratch: ScratchOwned = ScratchOwned::new(BlindRotationKeyCGGI::generate_from_sk_scratch_space(
        &module, basek, k_brk, rank,
    ));

    let mut scratch_br: ScratchOwned = ScratchOwned::new(cggi_blind_rotate_scratch_space(
        &module,
        block_size,
        extension_factor,
        basek,
        k_res,
        k_brk,
        rows_brk,
        rank,
    ));

    let mut brk: BlindRotationKeyCGGI<Vec<u8>, FFT64> =
        BlindRotationKeyCGGI::allocate(&module, n_lwe, basek, k_brk, rows_brk, rank);

    brk.generate_from_sk(
        &module,
        &sk_glwe_dft,
        &sk_lwe,
        &mut source_xa,
        &mut source_xe,
        3.2,
        scratch.borrow(),
    );

    let mut lwe: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(n_lwe, basek, k_lwe);

    let mut pt_lwe: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(basek, k_lwe);

    let x: i64 = 2;
    let bits: usize = 8;

    pt_lwe.data.encode_coeff_i64(0, basek, bits, 0, x, bits);

    lwe.encrypt_sk(&pt_lwe, &sk_lwe, &mut source_xa, &mut source_xe, 3.2);

    let mut f: Vec<i64> = vec![0i64; message_modulus];
    f.iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = 2 * (i as i64) + 1);

    let mut lut: LookUpTable = LookUpTable::alloc(&module, basek, k_lut, extension_factor);
    lut.set(&module, &f, message_modulus);

    let mut res: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_res, rank);

    cggi_blind_rotate(&module, &mut res, &lwe, &lut, &brk, scratch_br.borrow());

    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_res);

    res.decrypt(&module, &mut pt_have, &sk_glwe_dft, scratch.borrow());

    let mut lwe_2n: Vec<i64> = vec![0i64; lwe.n() + 1]; // TODO: from scratch space

    mod_switch_2n(
        2 * lut.domain_size(),
        &mut lwe_2n,
        &lwe.to_ref(),
        lut.rotation_direction(),
    );

    let pt_want: i64 = (lwe_2n[0]
        + lwe_2n[1..]
            .iter()
            .zip(sk_lwe.data.at(0, 0))
            .map(|(x, y)| x * y)
            .sum::<i64>())
        & (2 * lut.domain_size() - 1) as i64;

    lut.rotate(pt_want);

    // First limb should be exactly equal (test are parameterized such that the noise does not reach
    // the first limb)
    assert_eq!(pt_have.data.at(0, 0), lut.data[0].at(0, 0));
}
