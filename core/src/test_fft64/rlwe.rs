#[cfg(test)]
mod tests_rlwe {
    use base2k::{
        Decoding, Encoding, FFT64, FillUniform, Module, ScalarZnx, ScalarZnxAlloc, ScratchOwned, Stats, VecZnxOps, VecZnxToMut,
        ZnxViewMut, ZnxZero,
    };
    use itertools::izip;
    use sampling::source::Source;

    use crate::{
        elem::{FromProdBy, FromProdByScratchSpace, Infos, ProdBy, ProdByScratchSpace},
        grlwe::GRLWECt,
        keys::{PublicKey, SecretKey, SecretKeyDft},
        rgsw::RGSWCt,
        rlwe::{RLWECt, RLWECtDft, RLWEPt},
        test_fft64::{grlwe::noise_grlwe_rlwe_product, rgsw::noise_rgsw_rlwe_product},
    };

    #[test]
    fn encrypt_sk() {
        let module: Module<FFT64> = Module::<FFT64>::new(32);
        let log_base2k: usize = 8;
        let log_k_ct: usize = 54;
        let log_k_pt: usize = 30;

        let sigma: f64 = 3.2;
        let bound: f64 = sigma * 6.0;

        let mut ct: RLWECt<Vec<u8>> = RLWECt::new(&module, log_base2k, log_k_ct);
        let mut pt: RLWEPt<Vec<u8>> = RLWEPt::new(&module, log_base2k, log_k_pt);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned = ScratchOwned::new(
            RLWECt::encrypt_sk_scratch_space(&module, ct.size()) | RLWECt::decrypt_scratch_space(&module, ct.size()),
        );

        let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module);
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_dft: SecretKeyDft<Vec<u8>, FFT64> = SecretKeyDft::new(&module);
        sk_dft.dft(&module, &sk);

        let mut data_want: Vec<i64> = vec![0i64; module.n()];

        data_want
            .iter_mut()
            .for_each(|x| *x = source_xa.next_i64() & 0xFF);

        pt.data
            .encode_vec_i64(0, log_base2k, log_k_pt, &data_want, 10);

        ct.encrypt_sk(
            &module,
            Some(&pt),
            &sk_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            bound,
            scratch.borrow(),
        );

        pt.data.zero();

        ct.decrypt(&module, &mut pt, &sk_dft, scratch.borrow());

        let mut data_have: Vec<i64> = vec![0i64; module.n()];

        pt.data
            .decode_vec_i64(0, log_base2k, pt.size() * log_base2k, &mut data_have);

        // TODO: properly assert the decryption noise through std(dec(ct) - pt)
        let scale: f64 = (1 << (pt.size() * log_base2k - log_k_pt)) as f64;
        izip!(data_want.iter(), data_have.iter()).for_each(|(a, b)| {
            let b_scaled = (*b as f64) / scale;
            assert!(
                (*a as f64 - b_scaled).abs() < 0.1,
                "{} {}",
                *a as f64,
                b_scaled
            )
        });

        module.free();
    }

    #[test]
    fn encrypt_zero_sk() {
        let module: Module<FFT64> = Module::<FFT64>::new(1024);
        let log_base2k: usize = 8;
        let log_k_ct: usize = 55;

        let sigma: f64 = 3.2;
        let bound: f64 = sigma * 6.0;

        let mut pt: RLWEPt<Vec<u8>> = RLWEPt::new(&module, log_base2k, log_k_ct);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([1u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module);
        sk.fill_ternary_prob(0.5, &mut source_xs);
        let mut sk_dft: SecretKeyDft<Vec<u8>, FFT64> = SecretKeyDft::new(&module);
        sk_dft.dft(&module, &sk);

        let mut ct_dft: RLWECtDft<Vec<u8>, FFT64> = RLWECtDft::new(&module, log_base2k, log_k_ct);

        let mut scratch: ScratchOwned = ScratchOwned::new(
            RLWECtDft::decrypt_scratch_space(&module, ct_dft.size())
                | RLWECtDft::encrypt_zero_sk_scratch_space(&module, ct_dft.size()),
        );

        ct_dft.encrypt_zero_sk(
            &module,
            &sk_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            bound,
            scratch.borrow(),
        );
        ct_dft.decrypt(&module, &mut pt, &sk_dft, scratch.borrow());

        assert!((sigma - pt.data.std(0, log_base2k) * (log_k_ct as f64).exp2()) <= 0.2);
        module.free();
    }

    #[test]
    fn encrypt_pk() {
        let module: Module<FFT64> = Module::<FFT64>::new(32);
        let log_base2k: usize = 8;
        let log_k_ct: usize = 54;
        let log_k_pk: usize = 64;

        let sigma: f64 = 3.2;
        let bound: f64 = sigma * 6.0;

        let mut ct: RLWECt<Vec<u8>> = RLWECt::new(&module, log_base2k, log_k_ct);
        let mut pt_want: RLWEPt<Vec<u8>> = RLWEPt::new(&module, log_base2k, log_k_ct);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);
        let mut source_xu: Source = Source::new([0u8; 32]);

        let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module);
        sk.fill_ternary_prob(0.5, &mut source_xs);
        let mut sk_dft: SecretKeyDft<Vec<u8>, FFT64> = SecretKeyDft::new(&module);
        sk_dft.dft(&module, &sk);

        let mut pk: PublicKey<Vec<u8>, FFT64> = PublicKey::new(&module, log_base2k, log_k_pk);
        pk.generate(
            &module,
            &sk_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            bound,
        );

        let mut scratch: ScratchOwned = ScratchOwned::new(
            RLWECt::encrypt_sk_scratch_space(&module, ct.size())
                | RLWECt::decrypt_scratch_space(&module, ct.size())
                | RLWECt::encrypt_pk_scratch_space(&module, pk.size()),
        );

        let mut data_want: Vec<i64> = vec![0i64; module.n()];

        data_want
            .iter_mut()
            .for_each(|x| *x = source_xa.next_i64() & 0);

        pt_want
            .data
            .encode_vec_i64(0, log_base2k, log_k_ct, &data_want, 10);

        ct.encrypt_pk(
            &module,
            Some(&pt_want),
            &pk,
            &mut source_xu,
            &mut source_xe,
            sigma,
            bound,
            scratch.borrow(),
        );

        let mut pt_have: RLWEPt<Vec<u8>> = RLWEPt::new(&module, log_base2k, log_k_ct);

        ct.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());

        module.vec_znx_sub_ab_inplace(&mut pt_want, 0, &pt_have, 0);

        assert!(((1.0f64 / 12.0).sqrt() - pt_want.data.std(0, log_base2k) * (log_k_ct as f64).exp2()).abs() < 0.2);

        module.free();
    }

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
                | RLWECt::from_prod_by_grlwe_scratch_space(
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

        ct_rlwe_out.from_prod_by_grlwe(&module, &ct_rlwe_in, &ct_grlwe, scratch.borrow());

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
    fn prod_grlwe() {
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
                | RLWECt::prod_by_grlwe_scratch_space(&module, ct_rlwe.size(), ct_grlwe.size()),
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

        ct_rlwe.prod_by_grlwe(&module, &ct_grlwe, scratch.borrow());

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
    fn from_prod_by_rgsw() {
        let module: Module<FFT64> = Module::<FFT64>::new(2048);
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
        pt_want
            .data
            .fill_uniform(log_base2k, 0, pt_want.size(), &mut source_xa);

        pt_want.to_mut().at_mut(0, 0)[1] = 1;

        let k: usize = 1;

        pt_rgsw.raw_mut()[k] = 1; // X^{k}

        let mut scratch: ScratchOwned = ScratchOwned::new(
            RGSWCt::encrypt_sk_scratch_space(&module, ct_rgsw.size())
                | RLWECt::decrypt_scratch_space(&module, ct_rlwe_out.size())
                | RLWECt::encrypt_sk_scratch_space(&module, ct_rlwe_in.size())
                | RLWECt::from_prod_by_rgsw_scratch_space(
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

        ct_rlwe_out.from_prod_by_rgsw(&module, &ct_rlwe_in, &ct_rgsw, scratch.borrow());

        ct_rlwe_out.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());

        module.vec_znx_rotate_inplace(k as i64, &mut pt_want, 0);

        module.vec_znx_sub_ab_inplace(&mut pt_have, 0, &pt_want, 0);

        let noise_have: f64 = pt_have.data.std(0, log_base2k).log2();

        let var_gct_err_lhs: f64 = sigma * sigma;
        let var_gct_err_rhs: f64 = 0f64;

        let var_msg: f64 = 1f64 / module.n() as f64; // X^{k}
        let var_a0_err: f64 = sigma * sigma;
        let var_a1_err: f64 = 1f64 / 12f64;

        let noise_want: f64 = noise_rgsw_rlwe_product(
            module.n() as f64,
            log_base2k,
            0.5,
            var_msg,
            var_a0_err,
            var_a1_err,
            var_gct_err_lhs,
            var_gct_err_rhs,
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
    fn prod_by_rgsw() {
        let module: Module<FFT64> = Module::<FFT64>::new(2048);
        let log_base2k: usize = 12;
        let log_k_grlwe: usize = 60;
        let log_k_rlwe_in: usize = 45;
        let log_k_rlwe_out: usize = 60;
        let rows: usize = (log_k_rlwe_in + log_base2k - 1) / log_base2k;

        let sigma: f64 = 3.2;
        let bound: f64 = sigma * 6.0;

        let mut ct_rgsw: RGSWCt<Vec<u8>, FFT64> = RGSWCt::new(&module, log_base2k, log_k_grlwe, rows);
        let mut ct_rlwe: RLWECt<Vec<u8>> = RLWECt::new(&module, log_base2k, log_k_rlwe_in);
        let mut pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
        let mut pt_want: RLWEPt<Vec<u8>> = RLWEPt::new(&module, log_base2k, log_k_rlwe_in);
        let mut pt_have: RLWEPt<Vec<u8>> = RLWEPt::new(&module, log_base2k, log_k_rlwe_out);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        // Random input plaintext
        pt_want
            .data
            .fill_uniform(log_base2k, 0, pt_want.size(), &mut source_xa);

        pt_want.to_mut().at_mut(0, 0)[1] = 1;

        let k: usize = 1;

        pt_rgsw.raw_mut()[k] = 1; // X^{k}

        let mut scratch: ScratchOwned = ScratchOwned::new(
            RGSWCt::encrypt_sk_scratch_space(&module, ct_rgsw.size())
                | RLWECt::decrypt_scratch_space(&module, ct_rlwe.size())
                | RLWECt::encrypt_sk_scratch_space(&module, ct_rlwe.size())
                | RLWECt::prod_by_rgsw_scratch_space(&module, ct_rlwe.size(), ct_rgsw.size()),
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

        ct_rlwe.encrypt_sk(
            &module,
            Some(&pt_want),
            &sk_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            bound,
            scratch.borrow(),
        );

        ct_rlwe.prod_by_rgsw(&module, &ct_rgsw, scratch.borrow());

        ct_rlwe.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());

        module.vec_znx_rotate_inplace(k as i64, &mut pt_want, 0);

        module.vec_znx_sub_ab_inplace(&mut pt_have, 0, &pt_want, 0);

        let noise_have: f64 = pt_have.data.std(0, log_base2k).log2();

        let var_gct_err_lhs: f64 = sigma * sigma;
        let var_gct_err_rhs: f64 = 0f64;

        let var_msg: f64 = 1f64 / module.n() as f64; // X^{k}
        let var_a0_err: f64 = sigma * sigma;
        let var_a1_err: f64 = 1f64 / 12f64;

        let noise_want: f64 = noise_rgsw_rlwe_product(
            module.n() as f64,
            log_base2k,
            0.5,
            var_msg,
            var_a0_err,
            var_a1_err,
            var_gct_err_lhs,
            var_gct_err_rhs,
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
