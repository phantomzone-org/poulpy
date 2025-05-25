use backend::{
    FFT64, Module, ScalarZnx, ScalarZnxDft, ScalarZnxDftAlloc, ScalarZnxDftOps, ScratchOwned, Stats, VecZnxDftOps, VecZnxOps,
};
use sampling::source::Source;

use crate::{
    elem::{GetRow, Infos},
    glwe_ciphertext_fourier::GLWECiphertextFourier,
    glwe_plaintext::GLWEPlaintext,
    keys::{SecretKey, SecretKeyFourier},
    tensor_key::TensorKey,
};

#[test]
fn encrypt_sk() {
    (1..4).for_each(|rank| {
        println!("test encrypt_sk rank: {}", rank);
        test_encrypt_sk(12, 16, 54, 3.2, rank);
    });
}

fn test_encrypt_sk(log_n: usize, basek: usize, k: usize, sigma: f64, rank: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let rows: usize = (k + basek - 1) / basek;

    let mut tensor_key: TensorKey<Vec<u8>, FFT64> = TensorKey::alloc(&module, basek, k, rows, rank);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(TensorKey::generate_from_sk_scratch_space(
        &module,
        rank,
        tensor_key.size(),
    ));

    let mut sk: SecretKey<Vec<u8>> = SecretKey::alloc(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::alloc(&module, rank);
    sk_dft.dft(generate_from_sksk);

    tensor_key.encrypt_sk(
        &module,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut ct_glwe_fourier: GLWECiphertextFourier<Vec<u8>, FFT64> = GLWECiphertextFourier::alloc(&module, basek, k, rank);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k);

    (0..rank).for_each(|i| {
        (0..rank).for_each(|j| {
            let mut sk_ij_dft: ScalarZnxDft<Vec<u8>, FFT64> = module.new_scalar_znx_dft(1);
            module.svp_apply(&mut sk_ij_dft, 0, &sk_dft.data, i, &sk_dft.data, j);
            let sk_ij: ScalarZnx<Vec<u8>> = module
                .vec_znx_idft_consume(sk_ij_dft.as_vec_znx_dft())
                .to_vec_znx_small()
                .to_scalar_znx();

            (0..tensor_key.rank_in()).for_each(|col_i| {
                (0..tensor_key.rows()).for_each(|row_i| {
                    tensor_key
                        .at(i, j)
                        .get_row(&module, row_i, col_i, &mut ct_glwe_fourier);
                    ct_glwe_fourier.decrypt(&module, &mut pt, &sk_dft, scratch.borrow());
                    module.vec_znx_sub_scalar_inplace(&mut pt, 0, row_i, &sk_ij, col_i);
                    let std_pt: f64 = pt.data.std(0, basek) * (k as f64).exp2();
                    assert!((sigma - std_pt).abs() <= 0.2, "{} {}", sigma, std_pt);
                });
            });
        })
    })
}
