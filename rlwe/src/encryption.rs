use base2k::{
    AddNormal, Backend, FFT64, FillUniform, Module, ScalarZnxDftOps, ScalarZnxDftToRef, Scratch, VecZnx, VecZnxAlloc,
    VecZnxBigAlloc, VecZnxBigOps, VecZnxBigScratch, VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxDftToMut, VecZnxToMut,
    VecZnxToRef, ZnxInfos,
};

use sampling::source::Source;

use crate::{elem::Infos, keys::SecretKey};

pub trait EncryptSk<B: Backend, D, P> {
    fn encrypt<S>(
        module: &Module<B>,
        res: &mut D,
        pt: Option<&P>,
        sk: &SecretKey<S>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch,
        sigma: f64,
        bound: f64,
    ) where
        S: ScalarZnxDftToRef<B>;

    fn encrypt_tmp_bytes(module: &Module<B>, size: usize) -> usize {
        (module.vec_znx_big_normalize_tmp_bytes() | module.bytes_of_vec_znx_dft(1, size)) + module.bytes_of_vec_znx_big(1, size)
    }
}

impl<C, P> EncryptSk<FFT64, C, P> for C
where
    C: VecZnxToMut + ZnxInfos + Infos<C>,
    P: VecZnxToRef,
{
    fn encrypt<S>(
        module: &Module<FFT64>,
        ct: &mut C,
        pt: Option<&P>,
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

    fn encrypt_zero_tmp_bytes(module: &Module<B>, size: usize) -> usize {
        (module.bytes_of_vec_znx(1, size) | module.bytes_of_vec_znx_dft(1, size))
            + module.bytes_of_vec_znx_big(1, size)
            + module.bytes_of_vec_znx(1, size)
            + module.vec_znx_big_normalize_tmp_bytes()
    }
}

impl<C> EncryptZeroSk<FFT64, C> for C
where
    C: VecZnxDftToMut<FFT64> + ZnxInfos + Infos<C>,
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
}
