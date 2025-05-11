#[cfg(test)]
mod tests {
    use base2k::{Decoding, Encoding, FFT64, Module, ScratchOwned, Stats, VecZnxOps, ZnxZero};
    use itertools::izip;
    use sampling::source::Source;

    use crate::{
        elem::Infos,
        keys::{PublicKey, SecretKey, SecretKeyDft},
        rlwe::{RLWECt, RLWECtDft, RLWEPt},
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
}
