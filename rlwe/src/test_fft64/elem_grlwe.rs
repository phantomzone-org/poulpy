#[cfg(test)]

mod test {
    use base2k::{FFT64, FillUniform, Module, ScalarZnx, ScalarZnxAlloc, ScratchOwned, Stats, VecZnxOps};
    use sampling::source::Source;

    use crate::{
        elem::Infos,
        elem_grlwe::GRLWECt,
        elem_rlwe::{RLWECt, RLWECtDft, RLWEPt},
        keys::{SecretKey, SecretKeyDft},
        test_fft64::elem_grlwe::noise_grlwe_rlwe_product,
    };

    #[test]
    fn encrypt_sk() {
        let module: Module<FFT64> = Module::<FFT64>::new(2048);
        let log_base2k: usize = 8;
        let log_k_ct: usize = 54;
        let rows: usize = 4;

        let sigma: f64 = 3.2;
        let bound: f64 = sigma * 6.0;

        let mut ct: GRLWECt<Vec<u8>, FFT64> = GRLWECt::new(&module, log_base2k, log_k_ct, rows);
        let mut pt: RLWEPt<Vec<u8>> = RLWEPt::new(&module, log_base2k, log_k_ct);
        let mut pt_scalar: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        pt_scalar.fill_ternary_hw(0, module.n(), &mut source_xs);

        let mut scratch: ScratchOwned = ScratchOwned::new(
            GRLWECt::encrypt_sk_scratch_space(&module, ct.size()) | RLWECtDft::decrypt_scratch_space(&module, ct.size()),
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

        (0..ct.rows()).for_each(|row_i| {
            ct.get_row(&module, row_i, &mut ct_rlwe_dft);
            ct_rlwe_dft.decrypt(&module, &mut pt, &sk_dft, scratch.borrow());
            module.vec_znx_sub_scalar_inplace(&mut pt, 0, row_i, &pt_scalar, 0);
            let std_pt: f64 = pt.data.std(0, log_base2k) * (log_k_ct as f64).exp2();
            assert!((sigma - std_pt).abs() <= 0.2, "{} {}", sigma, std_pt);
        });

        module.free();
    }

    #[test]
    fn mul_rlwe() {
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
        let mut ct_rlwe_out: RLWECt<Vec<u8>> = RLWECt::new(&module, log_base2k, log_k_rlwe_out);
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
                | GRLWECt::mul_rlwe_scratch_space(
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

        ct_grlwe.mul_rlwe(&module, &mut ct_rlwe_out, &ct_rlwe_in, scratch.borrow());

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
    fn mul_rlwe_inplace() {
        let module: Module<FFT64> = Module::<FFT64>::new(2048);
        let log_base2k: usize = 12;
        let log_k_grlwe: usize = 60;
        let log_k_rlwe: usize = 45;
        let rows: usize = (log_k_rlwe + log_base2k - 1) / log_base2k;

        let sigma: f64 = 3.2;
        let bound: f64 = sigma * 6.0;

        let mut ct_grlwe: GRLWECt<Vec<u8>, FFT64> = GRLWECt::new(&module, log_base2k, log_k_grlwe, rows);
        let mut ct_rlwe: RLWECt<Vec<u8>> = RLWECt::new(&module, log_base2k, log_k_rlwe);
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
                | GRLWECt::mul_rlwe_scratch_space(&module, ct_rlwe.size(), ct_rlwe.size(), ct_grlwe.size()),
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

        ct_grlwe.mul_rlwe_inplace(&module, &mut ct_rlwe, scratch.borrow());

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

    #[test]
    fn mul_rlwe_dft() {
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
                | GRLWECt::mul_rlwe_scratch_space(
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
        ct_grlwe.mul_rlwe_dft(
            &module,
            &mut ct_rlwe_out_dft,
            &ct_rlwe_in_dft,
            scratch.borrow(),
        );
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
    fn mul_rlwe_dft_inplace() {
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
                | GRLWECt::mul_rlwe_scratch_space(&module, ct_rlwe.size(), ct_rlwe.size(), ct_grlwe.size()),
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
        ct_grlwe.mul_rlwe_dft_inplace(&module, &mut ct_rlwe_dft, scratch.borrow());
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

    #[test]
    fn mul_grlwe() {
        let module: Module<FFT64> = Module::<FFT64>::new(2048);
        let log_base2k: usize = 12;
        let log_k_grlwe: usize = 60;
        let rows: usize = (log_k_grlwe + log_base2k - 1) / log_base2k;

        let sigma: f64 = 3.2;
        let bound: f64 = sigma * 6.0;

        let mut ct_grlwe_s0s1: GRLWECt<Vec<u8>, FFT64> = GRLWECt::new(&module, log_base2k, log_k_grlwe, rows);
        let mut ct_grlwe_s1s2: GRLWECt<Vec<u8>, FFT64> = GRLWECt::new(&module, log_base2k, log_k_grlwe, rows);
        let mut ct_grlwe_s0s2: GRLWECt<Vec<u8>, FFT64> = GRLWECt::new(&module, log_base2k, log_k_grlwe, rows);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned = ScratchOwned::new(
            GRLWECt::encrypt_sk_scratch_space(&module, ct_grlwe_s0s1.size())
                | RLWECtDft::decrypt_scratch_space(&module, ct_grlwe_s0s2.size())
                | GRLWECt::mul_grlwe_scratch_space(
                    &module,
                    ct_grlwe_s0s2.size(),
                    ct_grlwe_s0s1.size(),
                    ct_grlwe_s1s2.size(),
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

        let mut sk2: SecretKey<Vec<u8>> = SecretKey::new(&module);
        sk2.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk2_dft: SecretKeyDft<Vec<u8>, FFT64> = SecretKeyDft::new(&module);
        sk2_dft.dft(&module, &sk2);

        // GRLWE_{s1}(s0) = s0 -> s1
        ct_grlwe_s0s1.encrypt_sk(
            &module,
            &sk0.data,
            &sk1_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            bound,
            scratch.borrow(),
        );

        // GRLWE_{s2}(s1) -> s1 -> s2
        ct_grlwe_s1s2.encrypt_sk(
            &module,
            &sk1.data,
            &sk2_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            bound,
            scratch.borrow(),
        );

        // GRLWE_{s1}(s0) (x) GRLWE_{s2}(s1) = GRLWE_{s2}(s0)
        ct_grlwe_s1s2.mul_grlwe(
            &module,
            &mut ct_grlwe_s0s2,
            &ct_grlwe_s0s1,
            scratch.borrow(),
        );

        let mut ct_rlwe_dft_s0s2: RLWECtDft<Vec<u8>, FFT64> = RLWECtDft::new(&module, log_base2k, log_k_grlwe);
        let mut pt: RLWEPt<Vec<u8>> = RLWEPt::new(&module, log_base2k, log_k_grlwe);

        (0..ct_grlwe_s0s2.rows()).for_each(|row_i| {
            ct_grlwe_s0s2.get_row(&module, row_i, &mut ct_rlwe_dft_s0s2);
            ct_rlwe_dft_s0s2.decrypt(&module, &mut pt, &sk2_dft, scratch.borrow());
            module.vec_znx_sub_scalar_inplace(&mut pt, 0, row_i, &sk0, 0);

            let noise_have: f64 = pt.data.std(0, log_base2k).log2();
            let noise_want: f64 = noise_grlwe_rlwe_product(
                module.n() as f64,
                log_base2k,
                0.5,
                0.5,
                0f64,
                sigma * sigma,
                0f64,
                log_k_grlwe,
                log_k_grlwe,
            );

            assert!(
                (noise_have - noise_want).abs() <= 0.1,
                "{} {}",
                noise_have,
                noise_want
            );
        });

        module.free();
    }

    #[test]
    fn mul_grlwe_inplace() {
        let module: Module<FFT64> = Module::<FFT64>::new(2048);
        let log_base2k: usize = 12;
        let log_k_grlwe: usize = 60;
        let rows: usize = (log_k_grlwe + log_base2k - 1) / log_base2k;

        let sigma: f64 = 3.2;
        let bound: f64 = sigma * 6.0;

        let mut ct_grlwe_s0s1: GRLWECt<Vec<u8>, FFT64> = GRLWECt::new(&module, log_base2k, log_k_grlwe, rows);
        let mut ct_grlwe_s1s2: GRLWECt<Vec<u8>, FFT64> = GRLWECt::new(&module, log_base2k, log_k_grlwe, rows);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned = ScratchOwned::new(
            GRLWECt::encrypt_sk_scratch_space(&module, ct_grlwe_s0s1.size())
                | RLWECtDft::decrypt_scratch_space(&module, ct_grlwe_s0s1.size())
                | GRLWECt::mul_grlwe_scratch_space(
                    &module,
                    ct_grlwe_s0s1.size(),
                    ct_grlwe_s0s1.size(),
                    ct_grlwe_s1s2.size(),
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

        let mut sk2: SecretKey<Vec<u8>> = SecretKey::new(&module);
        sk2.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk2_dft: SecretKeyDft<Vec<u8>, FFT64> = SecretKeyDft::new(&module);
        sk2_dft.dft(&module, &sk2);

        // GRLWE_{s1}(s0) = s0 -> s1
        ct_grlwe_s0s1.encrypt_sk(
            &module,
            &sk0.data,
            &sk1_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            bound,
            scratch.borrow(),
        );

        // GRLWE_{s2}(s1) -> s1 -> s2
        ct_grlwe_s1s2.encrypt_sk(
            &module,
            &sk1.data,
            &sk2_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            bound,
            scratch.borrow(),
        );

        // GRLWE_{s1}(s0) (x) GRLWE_{s2}(s1) = GRLWE_{s2}(s0)
        ct_grlwe_s1s2.mul_grlwe_inplace(&module, &mut ct_grlwe_s0s1, scratch.borrow());

        let ct_grlwe_s0s2: GRLWECt<Vec<u8>, FFT64> = ct_grlwe_s0s1;

        let mut ct_rlwe_dft_s0s2: RLWECtDft<Vec<u8>, FFT64> = RLWECtDft::new(&module, log_base2k, log_k_grlwe);
        let mut pt: RLWEPt<Vec<u8>> = RLWEPt::new(&module, log_base2k, log_k_grlwe);

        (0..ct_grlwe_s0s2.rows()).for_each(|row_i| {
            ct_grlwe_s0s2.get_row(&module, row_i, &mut ct_rlwe_dft_s0s2);
            ct_rlwe_dft_s0s2.decrypt(&module, &mut pt, &sk2_dft, scratch.borrow());
            module.vec_znx_sub_scalar_inplace(&mut pt, 0, row_i, &sk0, 0);

            let noise_have: f64 = pt.data.std(0, log_base2k).log2();
            let noise_want: f64 = noise_grlwe_rlwe_product(
                module.n() as f64,
                log_base2k,
                0.5,
                0.5,
                0f64,
                sigma * sigma,
                0f64,
                log_k_grlwe,
                log_k_grlwe,
            );

            assert!(
                (noise_have - noise_want).abs() <= 0.1,
                "{} {}",
                noise_have,
                noise_want
            );
        });

        module.free();
    }
}

#[allow(dead_code)]
pub(crate) fn noise_grlwe_rlwe_product(
    n: f64,
    log_base2k: usize,
    var_xs: f64,
    var_msg: f64,
    var_a_err: f64,
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
    let mut noise: f64 = (a_cols as f64) * n * var_base * (var_gct_err_lhs + var_xs * var_gct_err_rhs);
    noise += var_msg * var_a_err * a_scale * a_scale * n;
    noise = noise.sqrt();
    noise /= b_scale;
    noise.log2().min(-1.0) // max noise is [-2^{-1}, 2^{-1}]
}
