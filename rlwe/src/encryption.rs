use base2k::{
    AddNormal, Backend, FFT64, FillUniform, Module, ScalarZnxDftOps, ScalarZnxDftToRef, Scratch, VecZnx, VecZnxBigAlloc,
    VecZnxBigOps, VecZnxBigScratch, VecZnxDftAlloc, VecZnxDftOps, VecZnxToMut, VecZnxToRef, ZnxInfos,
};

use sampling::source::Source;

use crate::{
    elem::{CipherVecZnx, Plaintext},
    keys::SecretKey,
};

pub trait EncryptSk<B: Backend, D> {
    fn encrypt<P, S>(
        module: &Module<B>,
        res: &mut D,
        pt: Option<&Plaintext<P>>,
        sk: &SecretKey<S>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch,
        sigma: f64,
        bound: f64,
    ) where
        P: VecZnxToRef,
        S: ScalarZnxDftToRef<B>;

    fn encrypt_tmp_bytes(module: &Module<B>, size: usize) -> usize {
        (module.vec_znx_big_normalize_tmp_bytes() | module.bytes_of_vec_znx_dft(1, size)) + module.bytes_of_vec_znx_big(1, size)
    }
}

impl<C> EncryptSk<FFT64, CipherVecZnx<C>> for CipherVecZnx<C>
where
    VecZnx<C>: VecZnxToMut + VecZnxToRef,
{
    fn encrypt<P, S>(
        module: &Module<FFT64>,
        ct: &mut CipherVecZnx<C>,
        pt: Option<&Plaintext<P>>,
        sk: &SecretKey<S>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch,
        sigma: f64,
        bound: f64,
    ) where
        P: VecZnxToRef,
        S: ScalarZnxDftToRef<FFT64>,
    {
        let log_base2k: usize = ct.log_base2k();
        let log_q: usize = ct.log_q();
        let mut ct_mut: VecZnx<&mut [u8]> = ct.data_mut().to_mut();
        let size: usize = ct_mut.size();

        // c1 = a
        ct_mut.fill_uniform(log_base2k, 1, size, source_xa);

        let (mut c0_big, scratch_1) = scratch.tmp_vec_znx_big(module, 1, size);

        {
            let (mut c0_dft, _) = scratch_1.tmp_vec_znx_dft(module, 1, size);

            // c0_dft = DFT(a) * DFT(s)
            module.svp_apply(&mut c0_dft, 0, &sk.data().to_ref(), 0, &ct_mut, 1);

            // c0_big = IDFT(c0_dft)
            module.vec_znx_idft_tmp_a(&mut c0_big, 0, &mut c0_dft, 0);
        }

        // c0_big = m - c0_big
        if let Some(pt) = pt {
            module.vec_znx_big_sub_small_b_inplace(&mut c0_big, 0, &pt.data().to_ref(), 0);
        }
        // c0_big += e
        c0_big.add_normal(log_base2k, 0, log_q, source_xe, sigma, bound);

        // c0 = norm(c0_big = -as + m + e)
        module.vec_znx_big_normalize(log_base2k, &mut ct_mut, 0, &c0_big, 0, scratch_1);
    }
}
