use backend::{FFT64, Module, ScalarZnxDftOps, ScratchOwned, Stats, VecZnxOps};
use sampling::source::Source;

use crate::{FourierGLWECiphertext, FourierGLWESecret, GLWEPlaintext, GLWESecret, GLWETensorKey, GetRow, Infos};

#[test]
fn encrypt_sk() {
    let log_n: usize = 8;
    (1..4).for_each(|rank| {
        println!("test encrypt_sk rank: {}", rank);
        test_encrypt_sk(log_n, 16, 54, 3.2, rank);
    });
}

fn test_encrypt_sk(log_n: usize, basek: usize, k: usize, sigma: f64, rank: usize) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let rows: usize = k / basek;

    let mut tensor_key: GLWETensorKey<Vec<u8>, FFT64> = GLWETensorKey::alloc(&module, basek, k, rows, 1, rank);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(GLWETensorKey::encrypt_sk_scratch_space(
        &module,
        basek,
        tensor_key.k(),
        rank,
    ));

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk);

    tensor_key.encrypt_sk(
        &module,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut ct_glwe_fourier: FourierGLWECiphertext<Vec<u8>, FFT64> = FourierGLWECiphertext::alloc(&module, basek, k, rank);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k);

    let mut sk_ij_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::alloc(&module, 1);
    let mut sk_ij: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, 1);

    (0..rank).for_each(|i| {
        (0..rank).for_each(|j| {
            module.svp_apply(&mut sk_ij_dft.data, 0, &sk_dft.data, i, &sk_dft.data, j);
            module.scalar_znx_idft(&mut sk_ij.data, 0, &sk_ij_dft.data, 0, scratch.borrow());
            (0..tensor_key.rank_in()).for_each(|col_i| {
                (0..tensor_key.rows()).for_each(|row_i| {
                    tensor_key
                        .at(i, j)
                        .get_row(&module, row_i, col_i, &mut ct_glwe_fourier);
                    ct_glwe_fourier.decrypt(&module, &mut pt, &sk_dft, scratch.borrow());
                    module.vec_znx_sub_scalar_inplace(&mut pt.data, 0, row_i, &sk_ij.data, col_i);
                    let std_pt: f64 = pt.data.std(0, basek) * (k as f64).exp2();
                    assert!((sigma - std_pt).abs() <= 0.5, "{} {}", sigma, std_pt);
                });
            });
        })
    })
}
