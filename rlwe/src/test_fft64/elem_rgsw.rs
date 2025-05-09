#[cfg(test)]
mod tests {
    use base2k::{
        FFT64, Module, ScalarZnx, ScalarZnxAlloc, ScalarZnxDftOps, ScratchOwned, Stats, VecZnxBig, VecZnxBigAlloc, VecZnxBigOps,
        VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxOps, ZnxZero,
    };
    use sampling::source::Source;

    use crate::{
        elem::Infos,
        elem_rgsw::RGSWCt,
        elem_rlwe::{RLWECtDft, RLWEPt},
        keys::{SecretKey, SecretKeyDft},
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
}
