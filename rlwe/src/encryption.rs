use std::cmp::min;

use base2k::{
    AddNormal, Backend, FFT64, FillUniform, Module, ScalarZnxAlloc, ScalarZnxDft, ScalarZnxDftAlloc, ScalarZnxDftOps,
    ScalarZnxDftToRef, Scratch, VecZnx, VecZnxAlloc, VecZnxBigAlloc, VecZnxBigOps, VecZnxBigScratch, VecZnxDft, VecZnxDftAlloc,
    VecZnxDftOps, VecZnxDftToMut, VecZnxDftToRef, VecZnxOps, VecZnxToMut, VecZnxToRef,
};

use sampling::source::Source;

use crate::{
    elem::{Infos, RLWECt, RLWECtDft, RLWEPt},
    keys::{PublicKey, SecretDistribution, SecretKeyDft},
};

pub fn encrypt_rlwe_sk_scratch_bytes<B: Backend>(module: &Module<B>, size: usize) -> usize {
    (module.vec_znx_big_normalize_tmp_bytes() | module.bytes_of_vec_znx_dft(1, size)) + module.bytes_of_vec_znx_big(1, size)
}

pub fn encrypt_rlwe_sk<C, P, S>(
    module: &Module<FFT64>,
    ct: &mut RLWECt<C>,
    pt: Option<&RLWEPt<P>>,
    sk: &SecretKeyDft<S, FFT64>,
    source_xa: &mut Source,
    source_xe: &mut Source,
    sigma: f64,
    bound: f64,
    scratch: &mut Scratch,
) where
    VecZnx<C>: VecZnxToMut + VecZnxToRef,
    VecZnx<P>: VecZnxToRef,
    ScalarZnxDft<S, FFT64>: ScalarZnxDftToRef<FFT64>,
{
    let log_base2k: usize = ct.log_base2k();
    let log_k: usize = ct.log_k();
    let size: usize = ct.size();

    // c1 = a
    ct.data.fill_uniform(log_base2k, 1, size, source_xa);

    let (mut c0_big, scratch_1) = scratch.tmp_vec_znx_big(module, 1, size);

    {
        let (mut c0_dft, _) = scratch_1.tmp_vec_znx_dft(module, 1, size);
        module.vec_znx_dft(&mut c0_dft, 0, ct, 1);

        // c0_dft = DFT(a) * DFT(s)
        module.svp_apply_inplace(&mut c0_dft, 0, sk, 0);

        // c0_big = IDFT(c0_dft)
        module.vec_znx_idft_tmp_a(&mut c0_big, 0, &mut c0_dft, 0);
    }

    // c0_big = m - c0_big
    if let Some(pt) = pt {
        module.vec_znx_big_sub_small_b_inplace(&mut c0_big, 0, pt, 0);
    }
    // c0_big += e
    c0_big.add_normal(log_base2k, 0, log_k, source_xe, sigma, bound);

    // c0 = norm(c0_big = -as + m + e)
    module.vec_znx_big_normalize(log_base2k, ct, 0, &c0_big, 0, scratch_1);
}

pub fn decrypt_rlwe<P, C, S>(
    module: &Module<FFT64>,
    pt: &mut RLWEPt<P>,
    ct: &RLWECt<C>,
    sk: &SecretKeyDft<S, FFT64>,
    scratch: &mut Scratch,
) where
    VecZnx<P>: VecZnxToMut + VecZnxToRef,
    VecZnx<C>: VecZnxToRef,
    ScalarZnxDft<S, FFT64>: ScalarZnxDftToRef<FFT64>,
{
    let (mut c0_big, scratch_1) = scratch.tmp_vec_znx_big(module, 1, ct.size()); // TODO optimize size when pt << ct

    {
        let (mut c0_dft, _) = scratch_1.tmp_vec_znx_dft(module, 1, ct.size()); // TODO optimize size when pt << ct
        module.vec_znx_dft(&mut c0_dft, 0, ct, 1);

        // c0_dft = DFT(a) * DFT(s)
        module.svp_apply_inplace(&mut c0_dft, 0, sk, 0);

        // c0_big = IDFT(c0_dft)
        module.vec_znx_idft_tmp_a(&mut c0_big, 0, &mut c0_dft, 0);
    }

    // c0_big = (a * s) + (-a * s + m + e) = BIG(m + e)
    module.vec_znx_big_add_small_inplace(&mut c0_big, 0, ct, 0);

    // pt = norm(BIG(m + e))
    module.vec_znx_big_normalize(ct.log_base2k(), pt, 0, &mut c0_big, 0, scratch_1);

    pt.log_base2k = ct.log_base2k();
    pt.log_k = min(pt.log_k(), ct.log_k());
}

pub fn decrypt_rlwe_scratch_bytes<B: Backend>(module: &Module<B>, size: usize) -> usize {
    (module.vec_znx_big_normalize_tmp_bytes() | module.bytes_of_vec_znx_dft(1, size)) + module.bytes_of_vec_znx_big(1, size)
}

impl<C> RLWECt<C> {
    pub fn encrypt_sk<P, S>(
        &mut self,
        module: &Module<FFT64>,
        pt: Option<&RLWEPt<P>>,
        sk: &SecretKeyDft<S, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        bound: f64,
        scratch: &mut Scratch,
    ) where
        VecZnx<C>: VecZnxToMut + VecZnxToRef,
        VecZnx<P>: VecZnxToRef,
        ScalarZnxDft<S, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        encrypt_rlwe_sk(
            module, self, pt, sk, source_xa, source_xe, sigma, bound, scratch,
        )
    }

    pub fn decrypt<P, S>(&self, module: &Module<FFT64>, pt: &mut RLWEPt<P>, sk: &SecretKeyDft<S, FFT64>, scratch: &mut Scratch)
    where
        VecZnx<P>: VecZnxToMut + VecZnxToRef,
        VecZnx<C>: VecZnxToRef,
        ScalarZnxDft<S, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        decrypt_rlwe(module, pt, self, sk, scratch);
    }

    pub fn encrypt_pk<P, S>(
        &mut self,
        module: &Module<FFT64>,
        pt: Option<&RLWEPt<P>>,
        pk: &PublicKey<S, FFT64>,
        source_xu: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        bound: f64,
        scratch: &mut Scratch,
    ) where
        VecZnx<C>: VecZnxToMut + VecZnxToRef,
        VecZnx<P>: VecZnxToRef,
        VecZnxDft<S, FFT64>: VecZnxDftToRef<FFT64>,
    {
        encrypt_rlwe_pk(
            module, self, pt, pk, source_xu, source_xe, sigma, bound, scratch,
        )
    }
}

pub(crate) fn encrypt_zero_rlwe_dft_sk<C, S>(
    module: &Module<FFT64>,
    ct: &mut RLWECtDft<C, FFT64>,
    sk: &SecretKeyDft<S, FFT64>,
    source_xa: &mut Source,
    source_xe: &mut Source,
    sigma: f64,
    bound: f64,
    scratch: &mut Scratch,
) where
    VecZnxDft<C, FFT64>: VecZnxDftToMut<FFT64> + VecZnxDftToRef<FFT64>,
    ScalarZnxDft<S, FFT64>: ScalarZnxDftToRef<FFT64>,
{
    let log_base2k: usize = ct.log_base2k();
    let log_k: usize = ct.log_k();
    let size: usize = ct.size();

    #[cfg(debug_assertions)]
    {
        match sk.dist {
            SecretDistribution::NONE => panic!("invalid sk.dist = SecretDistribution::NONE"),
            _ => {}
        }
        assert_eq!(ct.cols(), 2);
    }

    // ct[1] = DFT(a)
    {
        let (mut tmp_znx, _) = scratch.tmp_vec_znx(module, 1, size);
        tmp_znx.fill_uniform(log_base2k, 0, size, source_xa);
        module.vec_znx_dft(ct, 1, &tmp_znx, 0);
    }

    let (mut c0_big, scratch_1) = scratch.tmp_vec_znx_big(module, 1, size);

    {
        let (mut tmp_dft, _) = scratch_1.tmp_vec_znx_dft(module, 1, size);
        // c0_dft = ct[1] * DFT(s)
        module.svp_apply(&mut tmp_dft, 0, sk, 0, ct, 1);

        // c0_big = IDFT(c0_dft)
        module.vec_znx_idft_tmp_a(&mut c0_big, 0, &mut tmp_dft, 0);
    }

    // c0_big += e
    c0_big.add_normal(log_base2k, 0, log_k, source_xe, sigma, bound);

    // c0 = norm(c0_big = -as - e), NOTE: e is centered at 0.
    let (mut tmp_znx, scratch_2) = scratch_1.tmp_vec_znx(module, 1, size);
    module.vec_znx_big_normalize(log_base2k, &mut tmp_znx, 0, &c0_big, 0, scratch_2);
    module.vec_znx_negate_inplace(&mut tmp_znx, 0);
    // ct[0] = DFT(-as + e)
    module.vec_znx_dft(ct, 0, &tmp_znx, 0);
}

pub(crate) fn encrypt_zero_rlwe_dft_scratch_bytes(module: &Module<FFT64>, size: usize) -> usize {
    (module.bytes_of_vec_znx(1, size) | module.bytes_of_vec_znx_dft(1, size))
        + module.bytes_of_vec_znx_big(1, size)
        + module.bytes_of_vec_znx(1, size)
        + module.vec_znx_big_normalize_tmp_bytes()
}

pub fn decrypt_rlwe_dft<P, C, S>(
    module: &Module<FFT64>,
    pt: &mut RLWEPt<P>,
    ct: &RLWECtDft<C, FFT64>,
    sk: &SecretKeyDft<S, FFT64>,
    scratch: &mut Scratch,
) where
    VecZnx<P>: VecZnxToMut + VecZnxToRef,
    VecZnxDft<C, FFT64>: VecZnxDftToRef<FFT64>,
    ScalarZnxDft<S, FFT64>: ScalarZnxDftToRef<FFT64>,
{
    let (mut c0_big, scratch_1) = scratch.tmp_vec_znx_big(module, 1, ct.size()); // TODO optimize size when pt << ct

    {
        let (mut c0_dft, _) = scratch_1.tmp_vec_znx_dft(module, 1, ct.size()); // TODO optimize size when pt << ct
        // c0_dft = DFT(a) * DFT(s)
        module.svp_apply(&mut c0_dft, 0, sk, 0, ct, 1);
        // c0_big = IDFT(c0_dft)
        module.vec_znx_idft_tmp_a(&mut c0_big, 0, &mut c0_dft, 0);
    }

    {
        let (mut c1_big, scratch_2) = scratch_1.tmp_vec_znx_big(module, 1, ct.size());
        // c0_big = (a * s) + (-a * s + m + e) = BIG(m + e)
        module.vec_znx_idft(&mut c1_big, 0, ct, 0, scratch_2);
        module.vec_znx_big_add_inplace(&mut c0_big, 0, &c1_big, 0);
    }

    // pt = norm(BIG(m + e))
    module.vec_znx_big_normalize(ct.log_base2k(), pt, 0, &mut c0_big, 0, scratch_1);

    pt.log_base2k = ct.log_base2k();
    pt.log_k = min(pt.log_k(), ct.log_k());
}

pub fn decrypt_rlwe_dft_scratch_bytes(module: &Module<FFT64>, size: usize) -> usize {
    (module.vec_znx_big_normalize_tmp_bytes()
        | module.bytes_of_vec_znx_dft(1, size)
        | (module.bytes_of_vec_znx_big(1, size) + module.vec_znx_idft_tmp_bytes()))
        + module.bytes_of_vec_znx_big(1, size)
}

impl<C> RLWECtDft<C, FFT64> {
    pub(crate) fn encrypt_zero_sk<S>(
        &mut self,
        module: &Module<FFT64>,
        sk_dft: &SecretKeyDft<S, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        bound: f64,
        scratch: &mut Scratch,
    ) where
        VecZnxDft<C, FFT64>: VecZnxDftToMut<FFT64> + VecZnxDftToRef<FFT64>,
        ScalarZnxDft<S, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        encrypt_zero_rlwe_dft_sk(
            module, self, sk_dft, source_xa, source_xe, sigma, bound, scratch,
        )
    }

    pub fn decrypt<P, S>(
        &self,
        module: &Module<FFT64>,
        pt: &mut RLWEPt<P>,
        sk_dft: &SecretKeyDft<S, FFT64>,
        scratch: &mut Scratch,
    ) where
        VecZnx<P>: VecZnxToMut + VecZnxToRef,
        VecZnxDft<C, FFT64>: VecZnxDftToRef<FFT64>,
        ScalarZnxDft<S, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        decrypt_rlwe_dft(module, pt, self, sk_dft, scratch);
    }
}

pub fn encrypt_rlwe_pk_scratch_bytes<B: Backend>(module: &Module<B>, pk_size: usize) -> usize {
    ((module.bytes_of_vec_znx_dft(1, pk_size) + module.bytes_of_vec_znx_big(1, pk_size)) | module.bytes_of_scalar_znx(1))
        + module.bytes_of_scalar_znx_dft(1)
        + module.vec_znx_big_normalize_tmp_bytes()
}

pub(crate) fn encrypt_rlwe_pk<C, P, S>(
    module: &Module<FFT64>,
    ct: &mut RLWECt<C>,
    pt: Option<&RLWEPt<P>>,
    pk: &PublicKey<S, FFT64>,
    source_xu: &mut Source,
    source_xe: &mut Source,
    sigma: f64,
    bound: f64,
    scratch: &mut Scratch,
) where
    VecZnx<C>: VecZnxToMut + VecZnxToRef,
    VecZnx<P>: VecZnxToRef,
    VecZnxDft<S, FFT64>: VecZnxDftToRef<FFT64>,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(ct.log_base2k(), pk.log_base2k());
        assert_eq!(ct.n(), module.n());
        assert_eq!(pk.n(), module.n());
        if let Some(pt) = pt {
            assert_eq!(pt.log_base2k(), pk.log_base2k());
            assert_eq!(pt.n(), module.n());
        }
    }

    let log_base2k: usize = pk.log_base2k();
    let size_pk: usize = pk.size();

    // Generates u according to the underlying secret distribution.
    let (mut u_dft, scratch_1) = scratch.tmp_scalar_dft(module, 1);

    {
        let (mut u, _) = scratch_1.tmp_scalar(module, 1);
        match pk.dist {
            SecretDistribution::NONE => panic!(
                "invalid public key: SecretDistribution::NONE, ensure it has been correctly intialized through Self::generate"
            ),
            SecretDistribution::TernaryFixed(hw) => u.fill_ternary_hw(0, hw, source_xu),
            SecretDistribution::TernaryProb(prob) => u.fill_ternary_prob(0, prob, source_xu),
        }

        module.svp_prepare(&mut u_dft, 0, &u, 0);
    }

    let (mut tmp_big, scratch_2) = scratch_1.tmp_vec_znx_big(module, 1, size_pk); // TODO optimize size (e.g. when encrypting at low homomorphic capacity)
    let (mut tmp_dft, scratch_3) = scratch_2.tmp_vec_znx_dft(module, 1, size_pk); // TODO optimize size (e.g. when encrypting at low homomorphic capacity)

    // ct[0] = pk[0] * u + m + e0
    module.svp_apply(&mut tmp_dft, 0, &u_dft, 0, pk, 0);
    module.vec_znx_idft_tmp_a(&mut tmp_big, 0, &mut tmp_dft, 0);
    tmp_big.add_normal(log_base2k, 0, pk.log_k(), source_xe, sigma, bound);

    if let Some(pt) = pt {
        module.vec_znx_big_add_small_inplace(&mut tmp_big, 0, pt, 0);
    }

    module.vec_znx_big_normalize(log_base2k, ct, 0, &tmp_big, 0, scratch_3);

    // ct[1] = pk[1] * u + e1
    module.svp_apply(&mut tmp_dft, 0, &u_dft, 0, pk, 1);
    module.vec_znx_idft_tmp_a(&mut tmp_big, 0, &mut tmp_dft, 0);
    tmp_big.add_normal(log_base2k, 0, pk.log_k(), source_xe, sigma, bound);
    module.vec_znx_big_normalize(log_base2k, ct, 1, &tmp_big, 0, scratch_3);
}

#[cfg(test)]
mod tests {
    use base2k::{Decoding, Encoding, FFT64, Module, ScratchOwned, Stats, VecZnxOps, ZnxZero};
    use itertools::izip;
    use sampling::source::Source;

    use crate::{
        elem::{Infos, RLWECt, RLWECtDft, RLWEPt},
        encryption::{decrypt_rlwe_dft_scratch_bytes, encrypt_zero_rlwe_dft_scratch_bytes},
        keys::{PublicKey, SecretKey, SecretKeyDft},
    };

    use super::{decrypt_rlwe_scratch_bytes, encrypt_rlwe_pk_scratch_bytes, encrypt_rlwe_sk_scratch_bytes};

    #[test]
    fn encrypt_sk_vec_znx_fft64() {
        let module: Module<FFT64> = Module::<FFT64>::new(32);
        let log_base2k: usize = 8;
        let log_k_ct: usize = 54;
        let log_k_pt: usize = 30;

        let sigma: f64 = 3.2;
        let bound: f64 = sigma * 6.0;

        let mut ct: RLWECt<Vec<u8>> = RLWECt::new(&module, log_base2k, log_k_ct, 2);
        let mut pt: RLWEPt<Vec<u8>> = RLWEPt::new(&module, log_base2k, log_k_pt);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned =
            ScratchOwned::new(encrypt_rlwe_sk_scratch_bytes(&module, ct.size()) | decrypt_rlwe_scratch_bytes(&module, ct.size()));

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
    fn encrypt_zero_rlwe_dft_sk_fft64() {
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

        let mut ct_dft: RLWECtDft<Vec<u8>, FFT64> = RLWECtDft::new(&module, log_base2k, log_k_ct, 2);

        let mut scratch: ScratchOwned = ScratchOwned::new(
            encrypt_rlwe_sk_scratch_bytes(&module, ct_dft.size())
                | decrypt_rlwe_dft_scratch_bytes(&module, ct_dft.size())
                | encrypt_zero_rlwe_dft_scratch_bytes(&module, ct_dft.size()),
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
    fn encrypt_pk_vec_znx_fft64() {
        let module: Module<FFT64> = Module::<FFT64>::new(32);
        let log_base2k: usize = 8;
        let log_k_ct: usize = 54;
        let log_k_pk: usize = 64;

        let sigma: f64 = 3.2;
        let bound: f64 = sigma * 6.0;

        let mut ct: RLWECt<Vec<u8>> = RLWECt::new(&module, log_base2k, log_k_ct, 2);
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
            encrypt_rlwe_sk_scratch_bytes(&module, ct.size())
                | decrypt_rlwe_scratch_bytes(&module, ct.size())
                | encrypt_rlwe_pk_scratch_bytes(&module, pk.size()),
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
