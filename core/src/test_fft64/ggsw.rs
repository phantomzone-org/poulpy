use backend::{
    FFT64, Module, ScalarZnx, ScalarZnxAlloc, ScalarZnxDftOps, ScalarZnxOps, ScratchOwned, Stats, VecZnxBig, VecZnxBigAlloc,
    VecZnxBigOps, VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxOps, VecZnxToMut, ZnxViewMut, ZnxZero,
};
use sampling::source::Source;

use crate::{
    FourierGLWECiphertext, GGSWCiphertext, GLWEPlaintext, GLWESecret, GLWESwitchingKey, GetRow, Infos, TensorKey,
    automorphism::AutomorphismKey,
    test_fft64::{noise_ggsw_keyswitch, noise_ggsw_product},
};

#[test]
fn encrypt_sk() {
    let log_n: usize = 8;
    let basek: usize = 12;
    let k_ct: usize = 54;
    let digits: usize = k_ct / basek;
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            println!("test encrypt_sk digits: {} rank: {}", di, rank);
            test_encrypt_sk(log_n, basek, k_ct, di, rank, 3.2);
        });
    });
}

#[test]
fn keyswitch() {
    let log_n: usize = 8;
    let basek: usize = 12;
    let k_in: usize = 54;
    let digits: usize = k_in.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = k_in + basek * di;
            let k_tsk: usize = k_ksk;
            println!("test keyswitch digits: {} rank: {}", di, rank);
            let k_out: usize = k_ksk; // Better capture noise.
            test_keyswitch(log_n, basek, k_out, k_in, k_ksk, k_tsk, di, rank, 3.2);
        });
    });
}

#[test]
fn keyswitch_inplace() {
    let log_n: usize = 8;
    let basek: usize = 12;
    let k_ct: usize = 54;
    let digits: usize = k_ct.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = k_ct + basek * di;
            let k_tsk: usize = k_ksk;
            println!("test keyswitch_inplace digits: {} rank: {}", di, rank);
            test_keyswitch_inplace(log_n, basek, k_ct, k_ksk, k_tsk, di, rank, 3.2);
        });
    });
}

#[test]
fn automorphism() {
    let log_n: usize = 8;
    let basek: usize = 12;
    let k_in: usize = 54;
    let digits: usize = k_in.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = k_in + basek * di;
            let k_tsk: usize = k_ksk;
            println!("test automorphism rank: {}", rank);
            let k_out: usize = k_ksk; // Better capture noise.
            test_automorphism(-5, log_n, basek, k_out, k_in, k_ksk, k_tsk, di, rank, 3.2);
        });
    });
}

#[test]
fn automorphism_inplace() {
    let log_n: usize = 8;
    let basek: usize = 12;
    let k_ct: usize = 54;
    let digits: usize = k_ct.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ksk: usize = k_ct + basek * di;
            let k_tsk: usize = k_ksk;
            println!("test automorphism_inplace rank: {}", rank);
            test_automorphism_inplace(-5, log_n, basek, k_ct, k_ksk, k_tsk, di, rank, 3.2);
        });
    });
}

#[test]
fn external_product() {
    let log_n: usize = 8;
    let basek: usize = 12;
    let k_in: usize = 60;
    let digits: usize = k_in.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits + 1).for_each(|di| {
            let k_ggsw: usize = k_in + basek * di;
            println!("test external_product digits: {} ranks: {}", di, rank);
            let k_out: usize = k_in; // Better capture noise.
            test_external_product(log_n, basek, k_in, k_out, k_ggsw, di, rank, 3.2);
        });
    });
}

#[test]
fn external_product_inplace() {
    let log_n: usize = 5;
    let basek: usize = 12;
    let k_ct: usize = 60;
    let digits: usize = k_ct.div_ceil(basek);
    (1..4).for_each(|rank| {
        (1..digits).for_each(|di| {
            let k_ggsw: usize = k_ct + basek * di;
            println!("test external_product digits: {} rank: {}", di, rank);
            test_external_product_inplace(log_n, basek, k_ct, k_ggsw, di, rank, 3.2);
        });
    });
}

fn test_encrypt_sk(log_n: usize, basek: usize, k: usize, digits: usize, rank: usize, sigma: f64) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let rows: usize = (k - digits * basek) / (digits * basek);

    let mut ct: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k, rows, digits, rank);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k);
    let mut pt_scalar: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    pt_scalar.fill_ternary_hw(0, module.n(), &mut source_xs);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(&module, basek, k, rank)
            | FourierGLWECiphertext::decrypt_scratch_space(&module, basek, k),
    );

    let mut sk: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(&module, 0.5, &mut source_xs);

    ct.encrypt_sk(
        &module,
        &pt_scalar,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut ct_glwe_fourier: FourierGLWECiphertext<Vec<u8>, FFT64> = FourierGLWECiphertext::alloc(&module, basek, k, rank);
    let mut pt_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(1, ct.size());
    let mut pt_big: VecZnxBig<Vec<u8>, FFT64> = module.new_vec_znx_big(1, ct.size());

    (0..ct.rank() + 1).for_each(|col_j| {
        (0..ct.rows()).for_each(|row_i| {
            module.vec_znx_add_scalar_inplace(
                &mut pt_want.data,
                0,
                (digits - 1) + row_i * digits,
                &pt_scalar,
                0,
            );

            // mul with sk[col_j-1]
            if col_j > 0 {
                module.vec_znx_dft(1, 0, &mut pt_dft, 0, &pt_want.data, 0);
                module.svp_apply_inplace(&mut pt_dft, 0, &sk.data_fourier, col_j - 1);
                module.vec_znx_idft_tmp_a(&mut pt_big, 0, &mut pt_dft, 0);
                module.vec_znx_big_normalize(basek, &mut pt_want.data, 0, &pt_big, 0, scratch.borrow());
            }

            ct.get_row(&module, row_i, col_j, &mut ct_glwe_fourier);

            ct_glwe_fourier.decrypt(&module, &mut pt_have, &sk, scratch.borrow());

            module.vec_znx_sub_ab_inplace(&mut pt_have.data, 0, &pt_want.data, 0);

            let std_pt: f64 = pt_have.data.std(0, basek) * (k as f64).exp2();
            assert!((sigma - std_pt).abs() <= 0.5, "{} {}", sigma, std_pt);

            pt_want.data.zero();
        });
    });
}

fn test_keyswitch(
    log_n: usize,
    basek: usize,
    k_out: usize,
    k_in: usize,
    k_ksk: usize,
    k_tsk: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = k_in.div_ceil(digits * basek);

    let digits_in: usize = 1;

    let mut ct_in: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_in, rows, digits_in, rank);
    let mut ct_out: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_out, rows, digits_in, rank);
    let mut tsk: TensorKey<Vec<u8>, FFT64> = TensorKey::alloc(&module, basek, k_ksk, rows, digits, rank);
    let mut ksk: GLWESwitchingKey<Vec<u8>, FFT64> = GLWESwitchingKey::alloc(&module, basek, k_ksk, rows, digits, rank, rank);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_out);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_out);
    let mut pt_scalar: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(&module, basek, k_in, rank)
            | FourierGLWECiphertext::decrypt_scratch_space(&module, basek, k_out)
            | GLWESwitchingKey::encrypt_sk_scratch_space(&module, basek, k_ksk, rank)
            | TensorKey::generate_from_sk_scratch_space(&module, basek, k_tsk, rank)
            | GGSWCiphertext::keyswitch_scratch_space(
                &module, basek, k_out, k_in, k_ksk, digits, k_tsk, digits, rank,
            ),
    );

    let var_xs: f64 = 0.5;

    let mut sk_in: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk_in.fill_ternary_prob(&module, var_xs, &mut source_xs);

    let mut sk_out: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk_out.fill_ternary_prob(&module, var_xs, &mut source_xs);

    ksk.generate_from_sk(
        &module,
        &sk_in,
        &sk_out,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );
    tsk.generate_from_sk(
        &module,
        &sk_out,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    pt_scalar.fill_ternary_hw(0, module.n(), &mut source_xs);

    ct_in.encrypt_sk(
        &module,
        &pt_scalar,
        &sk_in,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_out.keyswitch(&module, &ct_in, &ksk, &tsk, scratch.borrow());

    let mut ct_glwe_fourier: FourierGLWECiphertext<Vec<u8>, FFT64> = FourierGLWECiphertext::alloc(&module, basek, k_out, rank);
    let mut pt_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(1, ct_out.size());
    let mut pt_big: VecZnxBig<Vec<u8>, FFT64> = module.new_vec_znx_big(1, ct_out.size());

    (0..ct_out.rank() + 1).for_each(|col_j| {
        (0..ct_out.rows()).for_each(|row_i| {
            module.vec_znx_add_scalar_inplace(&mut pt_want.data, 0, row_i, &pt_scalar, 0);

            // mul with sk[col_j-1]
            if col_j > 0 {
                module.vec_znx_dft(1, 0, &mut pt_dft, 0, &pt_want.data, 0);
                module.svp_apply_inplace(&mut pt_dft, 0, &sk_out.data_fourier, col_j - 1);
                module.vec_znx_idft_tmp_a(&mut pt_big, 0, &mut pt_dft, 0);
                module.vec_znx_big_normalize(basek, &mut pt_want.data, 0, &pt_big, 0, scratch.borrow());
            }

            ct_out.get_row(&module, row_i, col_j, &mut ct_glwe_fourier);

            ct_glwe_fourier.decrypt(&module, &mut pt_have, &sk_out, scratch.borrow());

            module.vec_znx_sub_ab_inplace(&mut pt_have.data, 0, &pt_want.data, 0);

            let noise_have: f64 = pt_have.data.std(0, basek).log2();
            let noise_want: f64 = noise_ggsw_keyswitch(
                module.n() as f64,
                basek * digits,
                col_j,
                var_xs,
                0f64,
                sigma * sigma,
                0f64,
                rank as f64,
                k_in,
                k_ksk,
                k_tsk,
            );

            println!("{} {}", noise_have, noise_want);

            assert!(
                noise_have < noise_want + 0.5,
                "{} {}",
                noise_have,
                noise_want
            );

            pt_want.data.zero();
        });
    });
}

fn test_keyswitch_inplace(
    log_n: usize,
    basek: usize,
    k_ct: usize,
    k_ksk: usize,
    k_tsk: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = k_ct.div_ceil(digits * basek);

    let digits_in: usize = 1;

    let mut ct: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_ct, rows, digits_in, rank);
    let mut tsk: TensorKey<Vec<u8>, FFT64> = TensorKey::alloc(&module, basek, k_tsk, rows, digits, rank);
    let mut ksk: GLWESwitchingKey<Vec<u8>, FFT64> = GLWESwitchingKey::alloc(&module, basek, k_ksk, rows, digits, rank, rank);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);
    let mut pt_scalar: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(&module, basek, k_ct, rank)
            | FourierGLWECiphertext::decrypt_scratch_space(&module, basek, k_ct)
            | GLWESwitchingKey::encrypt_sk_scratch_space(&module, basek, k_ksk, rank)
            | TensorKey::generate_from_sk_scratch_space(&module, basek, k_tsk, rank)
            | GGSWCiphertext::keyswitch_inplace_scratch_space(&module, basek, k_ct, k_ksk, digits, k_tsk, digits, rank),
    );

    let var_xs: f64 = 0.5;

    let mut sk_in: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk_in.fill_ternary_prob(&module, var_xs, &mut source_xs);

    let mut sk_out: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk_out.fill_ternary_prob(&module, var_xs, &mut source_xs);

    ksk.generate_from_sk(
        &module,
        &sk_in,
        &sk_out,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );
    tsk.generate_from_sk(
        &module,
        &sk_out,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    pt_scalar.fill_ternary_hw(0, module.n(), &mut source_xs);

    ct.encrypt_sk(
        &module,
        &pt_scalar,
        &sk_in,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct.keyswitch_inplace(&module, &ksk, &tsk, scratch.borrow());

    let mut ct_glwe_fourier: FourierGLWECiphertext<Vec<u8>, FFT64> = FourierGLWECiphertext::alloc(&module, basek, k_ct, rank);
    let mut pt_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(1, ct.size());
    let mut pt_big: VecZnxBig<Vec<u8>, FFT64> = module.new_vec_znx_big(1, ct.size());

    (0..ct.rank() + 1).for_each(|col_j| {
        (0..ct.rows()).for_each(|row_i| {
            module.vec_znx_add_scalar_inplace(
                &mut pt_want.data,
                0,
                (digits_in - 1) + row_i * digits_in,
                &pt_scalar,
                0,
            );

            // mul with sk[col_j-1]
            if col_j > 0 {
                module.vec_znx_dft(1, 0, &mut pt_dft, 0, &pt_want.data, 0);
                module.svp_apply_inplace(&mut pt_dft, 0, &sk_out.data_fourier, col_j - 1);
                module.vec_znx_idft_tmp_a(&mut pt_big, 0, &mut pt_dft, 0);
                module.vec_znx_big_normalize(basek, &mut pt_want.data, 0, &pt_big, 0, scratch.borrow());
            }

            ct.get_row(&module, row_i, col_j, &mut ct_glwe_fourier);

            ct_glwe_fourier.decrypt(&module, &mut pt_have, &sk_out, scratch.borrow());

            module.vec_znx_sub_ab_inplace(&mut pt_have.data, 0, &pt_want.data, 0);

            let noise_have: f64 = pt_have.data.std(0, basek).log2();
            let noise_want: f64 = noise_ggsw_keyswitch(
                module.n() as f64,
                basek * digits,
                col_j,
                var_xs,
                0f64,
                sigma * sigma,
                0f64,
                rank as f64,
                k_ct,
                k_ksk,
                k_tsk,
            );

            println!("{} {}", noise_have, noise_want);

            assert!(
                noise_have < noise_want + 0.5,
                "{} {}",
                noise_have,
                noise_want
            );

            pt_want.data.zero();
        });
    });
}

fn test_automorphism(
    p: i64,
    log_n: usize,
    basek: usize,
    k_out: usize,
    k_in: usize,
    k_ksk: usize,
    k_tsk: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = k_in.div_ceil(basek * digits);
    let rows_in: usize = k_in.div_euclid(basek * digits);

    let digits_in: usize = 1;

    let mut ct_in: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_in, rows_in, digits_in, rank);
    let mut ct_out: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_out, rows_in, digits_in, rank);
    let mut tensor_key: TensorKey<Vec<u8>, FFT64> = TensorKey::alloc(&module, basek, k_tsk, rows, digits, rank);
    let mut auto_key: AutomorphismKey<Vec<u8>, FFT64> = AutomorphismKey::alloc(&module, basek, k_ksk, rows, digits, rank);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_out);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_out);
    let mut pt_scalar: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(&module, basek, k_in, rank)
            | FourierGLWECiphertext::decrypt_scratch_space(&module, basek, k_out)
            | AutomorphismKey::generate_from_sk_scratch_space(&module, basek, k_ksk, rank)
            | TensorKey::generate_from_sk_scratch_space(&module, basek, k_tsk, rank)
            | GGSWCiphertext::automorphism_scratch_space(
                &module, basek, k_out, k_in, k_ksk, digits, k_tsk, digits, rank,
            ),
    );

    let var_xs: f64 = 0.5;

    let mut sk: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(&module, var_xs, &mut source_xs);

    auto_key.generate_from_sk(
        &module,
        p,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );
    tensor_key.generate_from_sk(
        &module,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    pt_scalar.fill_ternary_hw(0, module.n(), &mut source_xs);

    ct_in.encrypt_sk(
        &module,
        &pt_scalar,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_out.automorphism(&module, &ct_in, &auto_key, &tensor_key, scratch.borrow());

    module.scalar_znx_automorphism_inplace(p, &mut pt_scalar, 0);

    let mut ct_glwe_fourier: FourierGLWECiphertext<Vec<u8>, FFT64> = FourierGLWECiphertext::alloc(&module, basek, k_out, rank);
    let mut pt_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(1, ct_out.size());
    let mut pt_big: VecZnxBig<Vec<u8>, FFT64> = module.new_vec_znx_big(1, ct_out.size());

    (0..ct_out.rank() + 1).for_each(|col_j| {
        (0..ct_out.rows()).for_each(|row_i| {
            module.vec_znx_add_scalar_inplace(&mut pt_want.data, 0, row_i, &pt_scalar, 0);

            // mul with sk[col_j-1]
            if col_j > 0 {
                module.vec_znx_dft(1, 0, &mut pt_dft, 0, &pt_want.data, 0);
                module.svp_apply_inplace(&mut pt_dft, 0, &sk.data_fourier, col_j - 1);
                module.vec_znx_idft_tmp_a(&mut pt_big, 0, &mut pt_dft, 0);
                module.vec_znx_big_normalize(basek, &mut pt_want.data, 0, &pt_big, 0, scratch.borrow());
            }

            ct_out.get_row(&module, row_i, col_j, &mut ct_glwe_fourier);

            ct_glwe_fourier.decrypt(&module, &mut pt_have, &sk, scratch.borrow());

            module.vec_znx_sub_ab_inplace(&mut pt_have.data, 0, &pt_want.data, 0);

            let noise_have: f64 = pt_have.data.std(0, basek).log2();
            let noise_want: f64 = noise_ggsw_keyswitch(
                module.n() as f64,
                basek * digits,
                col_j,
                var_xs,
                0f64,
                sigma * sigma,
                0f64,
                rank as f64,
                k_in,
                k_ksk,
                k_tsk,
            );

            assert!(
                noise_have < noise_want + 0.5,
                "{} {}",
                noise_have,
                noise_want
            );

            pt_want.data.zero();
        });
    });
}

fn test_automorphism_inplace(
    p: i64,
    log_n: usize,
    basek: usize,
    k_ct: usize,
    k_ksk: usize,
    k_tsk: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = k_ct.div_ceil(digits * basek);
    let rows_in: usize = k_ct.div_euclid(basek * digits);
    let digits_in: usize = 1;

    let mut ct: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_ct, rows_in, digits_in, rank);
    let mut tensor_key: TensorKey<Vec<u8>, FFT64> = TensorKey::alloc(&module, basek, k_tsk, rows, digits, rank);
    let mut auto_key: AutomorphismKey<Vec<u8>, FFT64> = AutomorphismKey::alloc(&module, basek, k_ksk, rows, digits, rank);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);
    let mut pt_scalar: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned = ScratchOwned::new(
        GGSWCiphertext::encrypt_sk_scratch_space(&module, basek, k_ct, rank)
            | FourierGLWECiphertext::decrypt_scratch_space(&module, basek, k_ct)
            | AutomorphismKey::generate_from_sk_scratch_space(&module, basek, k_ksk, rank)
            | TensorKey::generate_from_sk_scratch_space(&module, basek, k_tsk, rank)
            | GGSWCiphertext::automorphism_inplace_scratch_space(&module, basek, k_ct, k_ksk, digits, k_tsk, digits, rank),
    );

    let var_xs: f64 = 0.5;

    let mut sk: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(&module, var_xs, &mut source_xs);

    auto_key.generate_from_sk(
        &module,
        p,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );
    tensor_key.generate_from_sk(
        &module,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    pt_scalar.fill_ternary_hw(0, module.n(), &mut source_xs);

    ct.encrypt_sk(
        &module,
        &pt_scalar,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct.automorphism_inplace(&module, &auto_key, &tensor_key, scratch.borrow());

    module.scalar_znx_automorphism_inplace(p, &mut pt_scalar, 0);

    let mut ct_glwe_fourier: FourierGLWECiphertext<Vec<u8>, FFT64> = FourierGLWECiphertext::alloc(&module, basek, k_ct, rank);
    let mut pt_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(1, ct.size());
    let mut pt_big: VecZnxBig<Vec<u8>, FFT64> = module.new_vec_znx_big(1, ct.size());

    (0..ct.rank() + 1).for_each(|col_j| {
        (0..ct.rows()).for_each(|row_i| {
            module.vec_znx_add_scalar_inplace(&mut pt_want.data, 0, row_i, &pt_scalar, 0);

            // mul with sk[col_j-1]
            if col_j > 0 {
                module.vec_znx_dft(1, 0, &mut pt_dft, 0, &pt_want.data, 0);
                module.svp_apply_inplace(&mut pt_dft, 0, &sk.data_fourier, col_j - 1);
                module.vec_znx_idft_tmp_a(&mut pt_big, 0, &mut pt_dft, 0);
                module.vec_znx_big_normalize(basek, &mut pt_want.data, 0, &pt_big, 0, scratch.borrow());
            }

            ct.get_row(&module, row_i, col_j, &mut ct_glwe_fourier);

            ct_glwe_fourier.decrypt(&module, &mut pt_have, &sk, scratch.borrow());

            module.vec_znx_sub_ab_inplace(&mut pt_have.data, 0, &pt_want.data, 0);

            let noise_have: f64 = pt_have.data.std(0, basek).log2();
            let noise_want: f64 = noise_ggsw_keyswitch(
                module.n() as f64,
                basek * digits,
                col_j,
                var_xs,
                0f64,
                sigma * sigma,
                0f64,
                rank as f64,
                k_ct,
                k_ksk,
                k_tsk,
            );

            assert!(
                noise_have <= noise_want + 0.5,
                "{} {}",
                noise_have,
                noise_want
            );

            pt_want.data.zero();
        });
    });
}

fn test_external_product(
    log_n: usize,
    basek: usize,
    k_in: usize,
    k_out: usize,
    k_ggsw: usize,
    digits: usize,
    rank: usize,
    sigma: f64,
) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);

    let rows: usize = k_in.div_ceil(basek * digits);
    let rows_in: usize = k_in.div_euclid(basek * digits);
    let digits_in: usize = 1;

    let mut ct_ggsw_lhs_in: GGSWCiphertext<Vec<u8>, FFT64> =
        GGSWCiphertext::alloc(&module, basek, k_in, rows_in, digits_in, rank);
    let mut ct_ggsw_lhs_out: GGSWCiphertext<Vec<u8>, FFT64> =
        GGSWCiphertext::alloc(&module, basek, k_out, rows_in, digits_in, rank);
    let mut ct_ggsw_rhs: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_ggsw, rows, digits, rank);
    let mut pt_ggsw_lhs: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
    let mut pt_ggsw_rhs: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    pt_ggsw_lhs.fill_ternary_prob(0, 0.5, &mut source_xs);

    let k: usize = 1;

    pt_ggsw_rhs.to_mut().raw_mut()[k] = 1; //X^{k}

    let mut scratch: ScratchOwned = ScratchOwned::new(
        FourierGLWECiphertext::decrypt_scratch_space(&module, basek, k_out)
            | GGSWCiphertext::encrypt_sk_scratch_space(&module, basek, k_ggsw, rank)
            | GGSWCiphertext::external_product_scratch_space(&module, basek, k_out, k_in, k_ggsw, digits, rank),
    );

    let mut sk: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(&module, 0.5, &mut source_xs);

    ct_ggsw_rhs.encrypt_sk(
        &module,
        &pt_ggsw_rhs,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_ggsw_lhs_in.encrypt_sk(
        &module,
        &pt_ggsw_lhs,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_ggsw_lhs_out.external_product(&module, &ct_ggsw_lhs_in, &ct_ggsw_rhs, scratch.borrow());

    let mut ct_glwe_fourier: FourierGLWECiphertext<Vec<u8>, FFT64> = FourierGLWECiphertext::alloc(&module, basek, k_out, rank);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_out);
    let mut pt_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(1, ct_ggsw_lhs_out.size());
    let mut pt_big: VecZnxBig<Vec<u8>, FFT64> = module.new_vec_znx_big(1, ct_ggsw_lhs_out.size());
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_out);

    module.vec_znx_rotate_inplace(k as i64, &mut pt_ggsw_lhs, 0);

    (0..ct_ggsw_lhs_out.rank() + 1).for_each(|col_j| {
        (0..ct_ggsw_lhs_out.rows()).for_each(|row_i| {
            module.vec_znx_add_scalar_inplace(
                &mut pt_want.data,
                0,
                (digits_in - 1) + row_i * digits_in,
                &pt_ggsw_lhs,
                0,
            );

            if col_j > 0 {
                module.vec_znx_dft(1, 0, &mut pt_dft, 0, &pt_want.data, 0);
                module.svp_apply_inplace(&mut pt_dft, 0, &sk.data_fourier, col_j - 1);
                module.vec_znx_idft_tmp_a(&mut pt_big, 0, &mut pt_dft, 0);
                module.vec_znx_big_normalize(basek, &mut pt_want.data, 0, &pt_big, 0, scratch.borrow());
            }

            ct_ggsw_lhs_out.get_row(&module, row_i, col_j, &mut ct_glwe_fourier);
            ct_glwe_fourier.decrypt(&module, &mut pt, &sk, scratch.borrow());

            module.vec_znx_sub_ab_inplace(&mut pt.data, 0, &pt_want.data, 0);

            let noise_have: f64 = pt.data.std(0, basek).log2();

            let var_gct_err_lhs: f64 = sigma * sigma;
            let var_gct_err_rhs: f64 = 0f64;

            let var_msg: f64 = 1f64 / module.n() as f64; // X^{k}
            let var_a0_err: f64 = sigma * sigma;
            let var_a1_err: f64 = 1f64 / 12f64;

            let noise_want: f64 = noise_ggsw_product(
                module.n() as f64,
                basek * digits,
                0.5,
                var_msg,
                var_a0_err,
                var_a1_err,
                var_gct_err_lhs,
                var_gct_err_rhs,
                rank as f64,
                k_in,
                k_ggsw,
            );

            assert!(
                noise_have <= noise_want + 0.5,
                "have: {} want: {}",
                noise_have,
                noise_want
            );

            println!("{} {}", noise_have, noise_want);

            pt_want.data.zero();
        });
    });
}

fn test_external_product_inplace(log_n: usize, basek: usize, k_ct: usize, k_ggsw: usize, digits: usize, rank: usize, sigma: f64) {
    let module: Module<FFT64> = Module::<FFT64>::new(1 << log_n);
    let rows: usize = k_ct.div_ceil(digits * basek);
    let rows_in: usize = k_ct.div_euclid(basek * digits);
    let digits_in: usize = 1;

    let mut ct_ggsw_lhs: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_ct, rows_in, digits_in, rank);
    let mut ct_ggsw_rhs: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_ggsw, rows, digits, rank);

    let mut pt_ggsw_lhs: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
    let mut pt_ggsw_rhs: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    pt_ggsw_lhs.fill_ternary_prob(0, 0.5, &mut source_xs);

    let k: usize = 1;

    pt_ggsw_rhs.to_mut().raw_mut()[k] = 1; //X^{k}

    let mut scratch: ScratchOwned = ScratchOwned::new(
        FourierGLWECiphertext::decrypt_scratch_space(&module, basek, k_ct)
            | GGSWCiphertext::encrypt_sk_scratch_space(&module, basek, k_ggsw, rank)
            | GGSWCiphertext::external_product_inplace_scratch_space(&module, basek, k_ct, k_ggsw, digits, rank),
    );

    let mut sk: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
    sk.fill_ternary_prob(&module, 0.5, &mut source_xs);

    ct_ggsw_rhs.encrypt_sk(
        &module,
        &pt_ggsw_rhs,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_ggsw_lhs.encrypt_sk(
        &module,
        &pt_ggsw_lhs,
        &sk,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    ct_ggsw_lhs.external_product_inplace(&module, &ct_ggsw_rhs, scratch.borrow());

    let mut ct_glwe_fourier: FourierGLWECiphertext<Vec<u8>, FFT64> = FourierGLWECiphertext::alloc(&module, basek, k_ct, rank);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);
    let mut pt_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(1, ct_ggsw_lhs.size());
    let mut pt_big: VecZnxBig<Vec<u8>, FFT64> = module.new_vec_znx_big(1, ct_ggsw_lhs.size());
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k_ct);

    module.vec_znx_rotate_inplace(k as i64, &mut pt_ggsw_lhs, 0);

    (0..ct_ggsw_lhs.rank() + 1).for_each(|col_j| {
        (0..ct_ggsw_lhs.rows()).for_each(|row_i| {
            module.vec_znx_add_scalar_inplace(
                &mut pt_want.data,
                0,
                (digits_in - 1) + row_i * digits_in,
                &pt_ggsw_lhs,
                0,
            );

            if col_j > 0 {
                module.vec_znx_dft(1, 0, &mut pt_dft, 0, &pt_want.data, 0);
                module.svp_apply_inplace(&mut pt_dft, 0, &sk.data_fourier, col_j - 1);
                module.vec_znx_idft_tmp_a(&mut pt_big, 0, &mut pt_dft, 0);
                module.vec_znx_big_normalize(basek, &mut pt_want.data, 0, &pt_big, 0, scratch.borrow());
            }

            ct_ggsw_lhs.get_row(&module, row_i, col_j, &mut ct_glwe_fourier);
            ct_glwe_fourier.decrypt(&module, &mut pt, &sk, scratch.borrow());

            module.vec_znx_sub_ab_inplace(&mut pt.data, 0, &pt_want.data, 0);

            let noise_have: f64 = pt.data.std(0, basek).log2();

            let var_gct_err_lhs: f64 = sigma * sigma;
            let var_gct_err_rhs: f64 = 0f64;

            let var_msg: f64 = 1f64 / module.n() as f64; // X^{k}
            let var_a0_err: f64 = sigma * sigma;
            let var_a1_err: f64 = 1f64 / 12f64;

            let noise_want: f64 = noise_ggsw_product(
                module.n() as f64,
                basek * digits,
                0.5,
                var_msg,
                var_a0_err,
                var_a1_err,
                var_gct_err_lhs,
                var_gct_err_rhs,
                rank as f64,
                k_ct,
                k_ggsw,
            );

            assert!(
                noise_have <= noise_want + 0.5,
                "have: {} want: {}",
                noise_have,
                noise_want
            );

            pt_want.data.zero();
        });
    });
}
