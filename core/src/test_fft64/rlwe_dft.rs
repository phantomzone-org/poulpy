#[cfg(test)]
mod tests {
    use crate::{
        elem::{FromProdBy, FromProdByScratchSpace, Infos, ProdBy, ProdByScratchSpace},
        grlwe::GRLWECt,
        keys::{SecretKey, SecretKeyDft},
        rlwe::{RLWECt, RLWECtDft, RLWEPt},
        test_fft64::grlwe::noise_grlwe_rlwe_product,
    };
    use base2k::{FFT64, FillUniform, Module, ScratchOwned, Stats, VecZnxOps};
    use sampling::source::Source;

    #[test]
    fn from_prod_by_grlwe() {
        let module: Module<FFT64> = Module::<FFT64>::new(2048);
        let log_base2k: usize = 12;
        let log_k_grlwe: usize = 60;
        let log_k_rlwe_in: usize = 45;
        let log_k_rlwe_out: usize = 60;
        let rows: usize = (log_k_rlwe_in + log_base2k - 1) / log_base2k;

        let sigma: f64 = 3.2;
        let bound: f64 = sigma * 6.0;

        let mut ct_grlwe: GRLWECt<Vec<u8>, FFT64> = GRLWECt::new(&module, log_base2k, log_k_grlwe, rows);
        let mut ct_rlwe_in: RLWECt<Vec<u8>> = RLWECt::new(&module, log_base2k, log_k_rlwe_in);
        let mut ct_rlwe_in_dft: RLWECtDft<Vec<u8>, FFT64> = RLWECtDft::new(&module, log_base2k, log_k_rlwe_in);
        let mut ct_rlwe_out: RLWECt<Vec<u8>> = RLWECt::new(&module, log_base2k, log_k_rlwe_out);
        let mut ct_rlwe_out_dft: RLWECtDft<Vec<u8>, FFT64> = RLWECtDft::new(&module, log_base2k, log_k_rlwe_out);
        let mut pt_want: RLWEPt<Vec<u8>> = RLWEPt::new(&module, log_base2k, log_k_rlwe_in);
        let mut pt_have: RLWEPt<Vec<u8>> = RLWEPt::new(&module, log_base2k, log_k_rlwe_out);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        // Random input plaintext
        pt_want
            .data
            .fill_uniform(log_base2k, 0, pt_want.size(), &mut source_xa);

        let mut scratch: ScratchOwned = ScratchOwned::new(
            GRLWECt::encrypt_sk_scratch_space(&module, ct_grlwe.size())
                | RLWECt::decrypt_scratch_space(&module, ct_rlwe_out.size())
                | RLWECt::encrypt_sk_scratch_space(&module, ct_rlwe_in.size())
                | RLWECtDft::from_prod_by_grlwe_scratch_space(
                    &module,
                    ct_rlwe_out.size(),
                    ct_rlwe_in.size(),
                    ct_grlwe.size(),
                ),
        );

        let mut sk0: SecretKey<Vec<u8>> = SecretKey::new(&module);
        sk0.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk0_dft: SecretKeyDft<Vec<u8>, FFT64> = SecretKeyDft::new(&module);
        sk0_dft.dft(&module, &sk0);

        let mut sk1: SecretKey<Vec<u8>> = SecretKey::new(&module);
        sk1.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk1_dft: SecretKeyDft<Vec<u8>, FFT64> = SecretKeyDft::new(&module);
        sk1_dft.dft(&module, &sk1);

        ct_grlwe.encrypt_sk(
            &module,
            &sk0.data,
            &sk1_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            bound,
            scratch.borrow(),
        );

        ct_rlwe_in.encrypt_sk(
            &module,
            Some(&pt_want),
            &sk0_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            bound,
            scratch.borrow(),
        );

        ct_rlwe_in.dft(&module, &mut ct_rlwe_in_dft);
        ct_rlwe_out_dft.from_prod_by_grlwe(&module, &ct_rlwe_in_dft, &ct_grlwe, scratch.borrow());
        ct_rlwe_out_dft.idft(&module, &mut ct_rlwe_out, scratch.borrow());

        ct_rlwe_out.decrypt(&module, &mut pt_have, &sk1_dft, scratch.borrow());

        module.vec_znx_sub_ab_inplace(&mut pt_have, 0, &pt_want, 0);

        let noise_have: f64 = pt_have.data.std(0, log_base2k).log2();
        let noise_want: f64 = noise_grlwe_rlwe_product(
            module.n() as f64,
            log_base2k,
            0.5,
            0.5,
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

    #[test]
    fn prod_by_grlwe() {
        let module: Module<FFT64> = Module::<FFT64>::new(2048);
        let log_base2k: usize = 12;
        let log_k_grlwe: usize = 60;
        let log_k_rlwe: usize = 45;
        let rows: usize = (log_k_rlwe + log_base2k - 1) / log_base2k;

        let sigma: f64 = 3.2;
        let bound: f64 = sigma * 6.0;

        let mut ct_grlwe: GRLWECt<Vec<u8>, FFT64> = GRLWECt::new(&module, log_base2k, log_k_grlwe, rows);
        let mut ct_rlwe: RLWECt<Vec<u8>> = RLWECt::new(&module, log_base2k, log_k_rlwe);
        let mut ct_rlwe_dft: RLWECtDft<Vec<u8>, FFT64> = RLWECtDft::new(&module, log_base2k, log_k_rlwe);
        let mut pt_want: RLWEPt<Vec<u8>> = RLWEPt::new(&module, log_base2k, log_k_rlwe);
        let mut pt_have: RLWEPt<Vec<u8>> = RLWEPt::new(&module, log_base2k, log_k_rlwe);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        // Random input plaintext
        pt_want
            .data
            .fill_uniform(log_base2k, 0, pt_want.size(), &mut source_xa);

        let mut scratch: ScratchOwned = ScratchOwned::new(
            GRLWECt::encrypt_sk_scratch_space(&module, ct_grlwe.size())
                | RLWECt::decrypt_scratch_space(&module, ct_rlwe.size())
                | RLWECt::encrypt_sk_scratch_space(&module, ct_rlwe.size())
                | RLWECtDft::prod_by_grlwe_scratch_space(&module, ct_rlwe_dft.size(), ct_grlwe.size()),
        );

        let mut sk0: SecretKey<Vec<u8>> = SecretKey::new(&module);
        sk0.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk0_dft: SecretKeyDft<Vec<u8>, FFT64> = SecretKeyDft::new(&module);
        sk0_dft.dft(&module, &sk0);

        let mut sk1: SecretKey<Vec<u8>> = SecretKey::new(&module);
        sk1.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk1_dft: SecretKeyDft<Vec<u8>, FFT64> = SecretKeyDft::new(&module);
        sk1_dft.dft(&module, &sk1);

        ct_grlwe.encrypt_sk(
            &module,
            &sk0.data,
            &sk1_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            bound,
            scratch.borrow(),
        );

        ct_rlwe.encrypt_sk(
            &module,
            Some(&pt_want),
            &sk0_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            bound,
            scratch.borrow(),
        );

        ct_rlwe.dft(&module, &mut ct_rlwe_dft);
        ct_rlwe_dft.prod_by_grlwe(&module, &ct_grlwe, scratch.borrow());
        ct_rlwe_dft.idft(&module, &mut ct_rlwe, scratch.borrow());

        ct_rlwe.decrypt(&module, &mut pt_have, &sk1_dft, scratch.borrow());

        module.vec_znx_sub_ab_inplace(&mut pt_have, 0, &pt_want, 0);

        let noise_have: f64 = pt_have.data.std(0, log_base2k).log2();
        let noise_want: f64 = noise_grlwe_rlwe_product(
            module.n() as f64,
            log_base2k,
            0.5,
            0.5,
            0f64,
            sigma * sigma,
            0f64,
            log_k_rlwe,
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
