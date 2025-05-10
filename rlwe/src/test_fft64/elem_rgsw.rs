#[cfg(test)]
mod tests {
    use base2k::{
        FillUniform, Module, ScalarZnx, ScalarZnxAlloc, ScalarZnxDftOps, ScratchOwned, Stats, VecZnxBig, VecZnxBigAlloc, VecZnxBigOps, VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxOps, VecZnxToMut, VecZnxToRef, ZnxView, ZnxViewMut, ZnxZero, FFT64
    };
    use sampling::source::Source;

    use crate::{
        elem::Infos,
        elem_rgsw::RGSWCt,
        elem_rlwe::{RLWECt, RLWECtDft, RLWEPt},
        keys::{SecretKey, SecretKeyDft}, test_fft64::elem_rgsw::noise_rgsw_rlwe_product,
    };

    #[test]
    fn encrypt_rgsw_sk() {
        let module: Module<FFT64> = Module::<FFT64>::new(2048);
        let log_base2k: usize = 8;
        let log_k_ct: usize = 54;
        let rows: usize = 4;

        let sigma: f64 = 3.2;
        let bound: f64 = sigma * 6.0;

        let mut ct: RGSWCt<Vec<u8>, FFT64> = RGSWCt::new(&module, log_base2k, log_k_ct, rows);
        let mut pt_have: RLWEPt<Vec<u8>> = RLWEPt::new(&module, log_base2k, log_k_ct);
        let mut pt_want: RLWEPt<Vec<u8>> = RLWEPt::new(&module, log_base2k, log_k_ct);
        let mut pt_scalar: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        pt_scalar.fill_ternary_hw(0, module.n(), &mut source_xs);

        let mut scratch: ScratchOwned = ScratchOwned::new(
            RGSWCt::encrypt_sk_scratch_space(&module, ct.size()) | RLWECtDft::decrypt_scratch_space(&module, ct.size()),
        );

        let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module);
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_dft: SecretKeyDft<Vec<u8>, FFT64> = SecretKeyDft::new(&module);
        sk_dft.dft(&module, &sk);

        ct.encrypt_sk(
            &module,
            &pt_scalar,
            &sk_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            bound,
            scratch.borrow(),
        );

        let mut ct_rlwe_dft: RLWECtDft<Vec<u8>, FFT64> = RLWECtDft::new(&module, log_base2k, log_k_ct);
        let mut pt_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(1, ct.size());
        let mut pt_big: VecZnxBig<Vec<u8>, FFT64> = module.new_vec_znx_big(1, ct.size());

        (0..ct.cols()).for_each(|col_j| {
            (0..ct.rows()).for_each(|row_i| {
                module.vec_znx_add_scalar_inplace(&mut pt_want, 0, row_i, &pt_scalar, 0);

                if col_j == 1 {
                    module.vec_znx_dft(&mut pt_dft, 0, &pt_want, 0);
                    module.svp_apply_inplace(&mut pt_dft, 0, &sk_dft, 0);
                    module.vec_znx_idft_tmp_a(&mut pt_big, 0, &mut pt_dft, 0);
                    module.vec_znx_big_normalize(log_base2k, &mut pt_want, 0, &pt_big, 0, scratch.borrow());
                }

                ct.get_row(&module, row_i, col_j, &mut ct_rlwe_dft);

                ct_rlwe_dft.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());

                module.vec_znx_sub_ab_inplace(&mut pt_have, 0, &pt_want, 0);

                let std_pt: f64 = pt_have.data.std(0, log_base2k) * (log_k_ct as f64).exp2();
                assert!((sigma - std_pt).abs() <= 0.2, "{} {}", sigma, std_pt);

                pt_want.data.zero();
            });
        });

        module.free();
    }

    #[test]
    fn mul_rlwe() {
        let module: Module<FFT64> = Module::<FFT64>::new(32);
        let log_base2k: usize = 12;
        let log_k_grlwe: usize = 60;
        let log_k_rlwe_in: usize = 45;
        let log_k_rlwe_out: usize = 60;
        let rows: usize = (log_k_rlwe_in + log_base2k - 1) / log_base2k;

        let sigma: f64 = 3.2;
        let bound: f64 = sigma * 6.0;

        let mut ct_rgsw: RGSWCt<Vec<u8>, FFT64> = RGSWCt::new(&module, log_base2k, log_k_grlwe, rows);
        let mut ct_rlwe_in: RLWECt<Vec<u8>> = RLWECt::new(&module, log_base2k, log_k_rlwe_in);
        let mut ct_rlwe_out: RLWECt<Vec<u8>> = RLWECt::new(&module, log_base2k, log_k_rlwe_out);
        let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
        let mut pt_want: RLWEPt<Vec<u8>> = RLWEPt::new(&module, log_base2k, log_k_rlwe_in);
        let mut pt_have: RLWEPt<Vec<u8>> = RLWEPt::new(&module, log_base2k, log_k_rlwe_out);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        // Random input plaintext
        //pt_want
        //    .data
        //    .fill_uniform(log_base2k, 0, pt_want.size(), &mut source_xa);

        pt_want.to_mut().at_mut(0, 0)[0] = 1;

        pt_rgsw.raw_mut()[1] = 1; // X^{1}

        let mut scratch: ScratchOwned = ScratchOwned::new(
            RGSWCt::encrypt_sk_scratch_space(&module, ct_rgsw.size())
                | RLWECt::decrypt_scratch_space(&module, ct_rlwe_out.size())
                | RLWECt::encrypt_sk_scratch_space(&module, ct_rlwe_in.size())
                | RGSWCt::mul_rlwe_scratch_space(
                    &module,
                    ct_rlwe_out.size(),
                    ct_rlwe_in.size(),
                    ct_rgsw.size(),
                ),
        );

        let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module);
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_dft: SecretKeyDft<Vec<u8>, FFT64> = SecretKeyDft::new(&module);
        sk_dft.dft(&module, &sk);

        ct_rgsw.encrypt_sk(
            &module,
            &pt_rgsw,
            &sk_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            bound,
            scratch.borrow(),
        );

        ct_rlwe_in.encrypt_sk(
            &module,
            Some(&pt_want),
            &sk_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            bound,
            scratch.borrow(),
        );

        ct_rgsw.mul_rlwe(&module, &mut ct_rlwe_out, &ct_rlwe_in, scratch.borrow());

        ct_rlwe_out.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());

        module.vec_znx_sub_ab_inplace(&mut pt_have, 0, &pt_want, 0);


        println!("{}", pt_want.data);
        println!("{}", pt_have.data);

        let noise_have: f64 = pt_have.data.std(0, log_base2k).log2();
        let noise_want: f64 = noise_rgsw_rlwe_product(
            module.n() as f64,
            log_base2k,
            0.5,
            0.5,
            0f64,
            0f64,
            sigma * sigma,
            0f64,
            log_k_rlwe_in,
            log_k_grlwe,
        );

        assert!(
            (noise_have - noise_want).abs() <= 0.1,
            "{} {}",
            noise_have,
            noise_want
        );

        module.free();
    }
}

#[allow(dead_code)]
pub(crate) fn noise_rgsw_rlwe_product(
    n: f64,
    log_base2k: usize,
    var_xs: f64,
    var_msg: f64,
    var_a0_err: f64,
    var_a1_err: f64,
    var_gct_err_lhs: f64,
    var_gct_err_rhs: f64,
    a_logq: usize,
    b_logq: usize,
) -> f64 {
    let a_logq: usize = a_logq.min(b_logq);
    let a_cols: usize = (a_logq + log_base2k - 1) / log_base2k;

    let b_scale = 2.0f64.powi(b_logq as i32);
    let a_scale: f64 = 2.0f64.powi((b_logq - a_logq) as i32);

    let base: f64 = (1 << (log_base2k)) as f64;
    let var_base: f64 = base * base / 12f64;

    // lhs = a_cols * n * (var_base * var_gct_err_lhs + var_e_a * var_msg * p^2)
    // rhs = a_cols * n * var_base * var_gct_err_rhs * var_xs
    let mut noise: f64 = 2.0 * (a_cols as f64) * n * var_base * (var_gct_err_lhs + var_xs * var_gct_err_rhs);
    noise += var_msg * var_a0_err * a_scale * a_scale * n;
    noise += var_msg * var_a1_err * a_scale * a_scale * n * var_xs;
    noise = noise.sqrt();
    noise /= b_scale;
    noise.log2().min(-1.0) // max noise is [-2^{-1}, 2^{-1}]
}