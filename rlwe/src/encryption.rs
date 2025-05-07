use base2k::{
    AddNormal, Backend, FFT64, FillUniform, Module, ScalarZnxDftOps, ScalarZnxDftToRef, Scratch, VecZnx, VecZnxAlloc,
    VecZnxBigAlloc, VecZnxBigOps, VecZnxBigScratch, VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxDftToMut, VecZnxToMut,
    VecZnxToRef, ZnxInfos,
};

use sampling::source::Source;

use crate::{
    elem::{Ciphertext, Infos, Plaintext},
    keys::SecretKey,
};

pub trait EncryptSk<B: Backend, C, P> {
    fn encrypt<S>(
        module: &Module<B>,
        res: &mut Ciphertext<C>,
        pt: Option<&Plaintext<P>>,
        sk: &SecretKey<S>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch,
        sigma: f64,
        bound: f64,
    ) where
        S: ScalarZnxDftToRef<B>;

    fn encrypt_scratch_bytes(module: &Module<B>, size: usize) -> usize;
}

impl<C, P> EncryptSk<FFT64, C, P> for Ciphertext<C>
where
    C: VecZnxToMut + ZnxInfos,
    P: VecZnxToRef + ZnxInfos,
{
    fn encrypt<S>(
        module: &Module<FFT64>,
        ct: &mut Ciphertext<C>,
        pt: Option<&Plaintext<P>>,
        sk: &SecretKey<S>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch,
        sigma: f64,
        bound: f64,
    ) where
        S: ScalarZnxDftToRef<FFT64>,
    {
        let log_base2k: usize = ct.log_base2k();
        let log_q: usize = ct.log_q();
        let mut ct_mut: VecZnx<&mut [u8]> = ct.to_mut();
        let size: usize = ct_mut.size();

        // c1 = a
        ct_mut.fill_uniform(log_base2k, 1, size, source_xa);

        let (mut c0_big, scratch_1) = scratch.tmp_vec_znx_big(module, 1, size);

        {
            let (mut c0_dft, _) = scratch_1.tmp_vec_znx_dft(module, 1, size);
            module.vec_znx_dft(&mut c0_dft, 0, &ct_mut, 1);

            // c0_dft = DFT(a) * DFT(s)
            module.svp_apply_inplace(&mut c0_dft, 0, &sk.data().to_ref(), 0);

            // c0_big = IDFT(c0_dft)
            module.vec_znx_idft_tmp_a(&mut c0_big, 0, &mut c0_dft, 0);
        }

        // c0_big = m - c0_big
        if let Some(pt) = pt {
            module.vec_znx_big_sub_small_b_inplace(&mut c0_big, 0, pt, 0);
        }
        // c0_big += e
        c0_big.add_normal(log_base2k, 0, log_q, source_xe, sigma, bound);

        // c0 = norm(c0_big = -as + m + e)
        module.vec_znx_big_normalize(log_base2k, &mut ct_mut, 0, &c0_big, 0, scratch_1);
    }

    fn encrypt_scratch_bytes(module: &Module<FFT64>, size: usize) -> usize {
        (module.vec_znx_big_normalize_tmp_bytes() | module.bytes_of_vec_znx_dft(1, size)) + module.bytes_of_vec_znx_big(1, size)
    }
}

impl<C> Ciphertext<C>
where
    C: VecZnxToMut + ZnxInfos,
{
    pub fn encrypt_sk<P, S>(
        &mut self,
        module: &Module<FFT64>,
        pt: Option<&Plaintext<P>>,
        sk: &SecretKey<S>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch,
        sigma: f64,
        bound: f64,
    ) where
        P: VecZnxToRef + ZnxInfos,
        S: ScalarZnxDftToRef<FFT64>,
    {
        <Self as EncryptSk<FFT64, _, _>>::encrypt(
            module, self, pt, sk, source_xa, source_xe, scratch, sigma, bound,
        );
    }

    pub fn encrypt_sk_scratch_bytes<P>(module: &Module<FFT64>, size: usize) -> usize
    where
        Self: EncryptSk<FFT64, C, P>,
    {
        <Self as EncryptSk<FFT64, C, P>>::encrypt_scratch_bytes(module, size)
    }
}

pub trait EncryptZeroSk<B: Backend, D> {
    fn encrypt_zero<S>(
        module: &Module<B>,
        res: &mut D,
        sk: &SecretKey<S>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch,
        sigma: f64,
        bound: f64,
    ) where
        S: ScalarZnxDftToRef<B>;

    fn encrypt_zero_scratch_bytes(module: &Module<B>, size: usize) -> usize;
}

impl<C> EncryptZeroSk<FFT64, C> for C
where
    C: VecZnxDftToMut<FFT64> + ZnxInfos + Infos,
{
    fn encrypt_zero<S>(
        module: &Module<FFT64>,
        ct: &mut C,
        sk: &SecretKey<S>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch,
        sigma: f64,
        bound: f64,
    ) where
        S: ScalarZnxDftToRef<FFT64>,
    {
        let log_base2k: usize = ct.log_base2k();
        let log_q: usize = ct.log_q();
        let mut ct_mut: VecZnxDft<&mut [u8], FFT64> = ct.to_mut();
        let size: usize = ct_mut.size();

        // ct[1] = DFT(a)
        {
            let (mut tmp_znx, _) = scratch.tmp_vec_znx(module, 1, size);
            tmp_znx.fill_uniform(log_base2k, 1, size, source_xa);
            module.vec_znx_dft(&mut ct_mut, 1, &tmp_znx, 0);
        }

        let (mut c0_big, scratch_1) = scratch.tmp_vec_znx_big(module, 1, size);

        {
            let (mut tmp_dft, _) = scratch_1.tmp_vec_znx_dft(module, 1, size);
            // c0_dft = DFT(a) * DFT(s)
            module.svp_apply(&mut tmp_dft, 0, &sk.data().to_ref(), 0, &ct_mut, 1);
            // c0_big = IDFT(c0_dft)
            module.vec_znx_idft_tmp_a(&mut c0_big, 0, &mut tmp_dft, 0);
        }

        // c0_big += e
        c0_big.add_normal(log_base2k, 0, log_q, source_xe, sigma, bound);

        // c0 = norm(c0_big = -as + e)
        let (mut tmp_znx, scratch_2) = scratch_1.tmp_vec_znx(module, 1, size);
        module.vec_znx_big_normalize(log_base2k, &mut tmp_znx, 0, &c0_big, 0, scratch_2);
        // ct[0] = DFT(-as + e)
        module.vec_znx_dft(&mut ct_mut, 0, &tmp_znx, 0);
    }

    fn encrypt_zero_scratch_bytes(module: &Module<FFT64>, size: usize) -> usize{
        (module.bytes_of_vec_znx(1, size) | module.bytes_of_vec_znx_dft(1, size))
            + module.bytes_of_vec_znx_big(1, size)
            + module.bytes_of_vec_znx(1, size)
            + module.vec_znx_big_normalize_tmp_bytes()
    }
}

#[cfg(test)]
mod tests {
    use base2k::{FFT64, Module, ScratchOwned, VecZnx, Scalar};
    use sampling::source::Source;

    use crate::{elem::{Ciphertext, Infos, Plaintext}, keys::SecretKey};

    #[test]
    fn encrypt_sk_vec_znx_fft64() {
        let module: Module<FFT64> = Module::<FFT64>::new(32);
        let log_base2k: usize = 8;
        let log_q: usize = 54;

        let sigma: f64 = 3.2;
        let bound: f64 = sigma * 6;

        let mut ct: Ciphertext<VecZnx<Vec<u8>>> = Ciphertext::<VecZnx<Vec<u8>>>::new(&module, log_base2k, log_q, 2);
        let mut pt: Plaintext<VecZnx<Vec<u8>>> = Plaintext::<VecZnx<Vec<u8>>>::new(&module, log_base2k, log_q);

        let mut source_xe = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);
        

        let mut scratch: ScratchOwned = ScratchOwned::new(ct.encrypt_encsk_scratch_bytes(&module, ct.size()));

        let mut sk: SecretKey<Scalar<Vec<u8>>> = SecretKey::new(&module);
        let mut sk_prep
        sk.svp_prepare(&module, &mut sk_prep);

        ct.encrypt_sk(
            &module,
            Some(&pt),
            &sk_prep,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
            sigma,
            bound,
        );
    }
}
