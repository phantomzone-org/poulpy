use backend::{
    Backend, FFT64, MatZnxDft, MatZnxDftOps, MatZnxDftScratch, MatZnxDftToRef, Module, ScalarZnxDft, ScalarZnxDftOps,
    ScalarZnxDftToRef, Scratch, VecZnx, VecZnxAlloc, VecZnxBig, VecZnxBigAlloc, VecZnxBigOps, VecZnxBigScratch, VecZnxDft,
    VecZnxDftAlloc, VecZnxDftOps, VecZnxDftToMut, VecZnxDftToRef, VecZnxToMut, VecZnxToRef, ZnxZero,
};
use sampling::source::Source;

use crate::{
    elem::Infos, ggsw_ciphertext::GGSWCiphertext, glwe_ciphertext::GLWECiphertext, glwe_plaintext::GLWEPlaintext,
    keys::SecretKeyFourier, keyswitch_key::GLWESwitchingKey, utils::derive_size,
};

pub struct GLWECiphertextFourier<C, B: Backend> {
    pub data: VecZnxDft<C, B>,
    pub basek: usize,
    pub k: usize,
}

impl<B: Backend> GLWECiphertextFourier<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rank: usize) -> Self {
        Self {
            data: module.new_vec_znx_dft(rank + 1, derive_size(basek, k)),
            basek: basek,
            k: k,
        }
    }
}

impl<T, B: Backend> Infos for GLWECiphertextFourier<T, B> {
    type Inner = VecZnxDft<T, B>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.basek
    }

    fn k(&self) -> usize {
        self.k
    }
}

impl<T, B: Backend> GLWECiphertextFourier<T, B> {
    pub fn rank(&self) -> usize {
        self.cols() - 1
    }
}

impl<C, B: Backend> VecZnxDftToMut<B> for GLWECiphertextFourier<C, B>
where
    VecZnxDft<C, B>: VecZnxDftToMut<B>,
{
    fn to_mut(&mut self) -> VecZnxDft<&mut [u8], B> {
        self.data.to_mut()
    }
}

impl<C, B: Backend> VecZnxDftToRef<B> for GLWECiphertextFourier<C, B>
where
    VecZnxDft<C, B>: VecZnxDftToRef<B>,
{
    fn to_ref(&self) -> VecZnxDft<&[u8], B> {
        self.data.to_ref()
    }
}

impl GLWECiphertextFourier<Vec<u8>, FFT64> {
    #[allow(dead_code)]
    pub(crate) fn idft_scratch_space(module: &Module<FFT64>, size: usize) -> usize {
        module.bytes_of_vec_znx(1, size) + (module.vec_znx_big_normalize_tmp_bytes() | module.vec_znx_idft_tmp_bytes())
    }

    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, rank: usize, ct_size: usize) -> usize {
        module.bytes_of_vec_znx(rank + 1, ct_size) + GLWECiphertext::encrypt_sk_scratch_space(module, ct_size)
    }

    pub fn decrypt_scratch_space(module: &Module<FFT64>, ct_size: usize) -> usize {
        (module.vec_znx_big_normalize_tmp_bytes()
            | module.bytes_of_vec_znx_dft(1, ct_size)
            | (module.bytes_of_vec_znx_big(1, ct_size) + module.vec_znx_idft_tmp_bytes()))
            + module.bytes_of_vec_znx_big(1, ct_size)
    }

    pub fn keyswitch_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        out_rank: usize,
        in_size: usize,
        in_rank: usize,
        ksk_size: usize,
    ) -> usize {
        module.bytes_of_vec_znx(out_rank + 1, out_size)
            + GLWECiphertext::keyswitch_from_fourier_scratch_space(module, out_size, out_rank, in_size, in_rank, ksk_size)
    }

    pub fn keyswitch_inplace_scratch_space(module: &Module<FFT64>, out_size: usize, out_rank: usize, ksk_size: usize) -> usize {
        Self::keyswitch_scratch_space(module, out_size, out_rank, out_size, out_rank, ksk_size)
    }

    pub fn external_product_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        in_size: usize,
        ggsw_size: usize,
        rank: usize,
    ) -> usize {
        let res_dft: usize = module.bytes_of_vec_znx_dft(rank + 1, out_size);
        let vmp: usize = module.vmp_apply_tmp_bytes(out_size, in_size, in_size, rank + 1, rank + 1, ggsw_size);
        let res_small: usize = module.bytes_of_vec_znx(rank + 1, out_size);
        let normalize: usize = module.vec_znx_big_normalize_tmp_bytes();

        res_dft + (vmp | (res_small + normalize))
    }

    pub fn external_product_inplace_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        ggsw_size: usize,
        rank: usize,
    ) -> usize {
        Self::external_product_scratch_space(module, out_size, out_size, ggsw_size, rank)
    }
}

impl<DataSelf> GLWECiphertextFourier<DataSelf, FFT64>
where
    VecZnxDft<DataSelf, FFT64>: VecZnxDftToMut<FFT64> + VecZnxDftToRef<FFT64>,
{
    pub fn encrypt_zero_sk<DataSk>(
        &mut self,
        module: &Module<FFT64>,
        sk_dft: &SecretKeyFourier<DataSk, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        ScalarZnxDft<DataSk, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        let (vec_znx_tmp, scratch_1) = scratch.tmp_vec_znx(module, self.rank() + 1, self.size());
        let mut ct_idft = GLWECiphertext {
            data: vec_znx_tmp,
            basek: self.basek,
            k: self.k,
        };
        ct_idft.encrypt_zero_sk(module, sk_dft, source_xa, source_xe, sigma, scratch_1);

        ct_idft.dft(module, self);
    }

    pub fn keyswitch<DataLhs, DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWECiphertextFourier<DataLhs, FFT64>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        VecZnxDft<DataLhs, FFT64>: VecZnxDftToRef<FFT64>,
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        let cols_out: usize = rhs.rank_out() + 1;

        // Space fr normalized VMP result outside of DFT domain
        let (res_idft_data, scratch1) = scratch.tmp_vec_znx(module, cols_out, lhs.size());

        let mut res_idft: GLWECiphertext<&mut [u8]> = GLWECiphertext::<&mut [u8]> {
            data: res_idft_data,
            basek: lhs.basek,
            k: lhs.k,
        };

        res_idft.keyswitch_from_fourier(module, lhs, rhs, scratch1);

        (0..cols_out).for_each(|i| {
            module.vec_znx_dft(self, i, &res_idft, i);
        });
    }

    pub fn keyswitch_inplace<DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        unsafe {
            let self_ptr: *mut GLWECiphertextFourier<DataSelf, FFT64> = self as *mut GLWECiphertextFourier<DataSelf, FFT64>;
            self.keyswitch(&module, &*self_ptr, rhs, scratch);
        }
    }

    pub fn external_product<DataLhs, DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWECiphertextFourier<DataLhs, FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        VecZnxDft<DataLhs, FFT64>: VecZnxDftToRef<FFT64>,
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        let basek: usize = self.basek();

        #[cfg(debug_assertions)]
        {
            assert_eq!(rhs.rank(), lhs.rank());
            assert_eq!(rhs.rank(), self.rank());
            assert_eq!(self.basek(), basek);
            assert_eq!(lhs.basek(), basek);
            assert_eq!(rhs.n(), module.n());
            assert_eq!(self.n(), module.n());
            assert_eq!(lhs.n(), module.n());
        }

        let cols: usize = rhs.rank() + 1;

        // Space for VMP result in DFT domain and high precision
        let (mut res_dft, scratch1) = scratch.tmp_vec_znx_dft(module, cols, rhs.size());

        {
            module.vmp_apply(&mut res_dft, lhs, rhs, scratch1);
        }

        // VMP result in high precision
        let res_big: VecZnxBig<&mut [u8], FFT64> = module.vec_znx_idft_consume::<&mut [u8]>(res_dft);

        // Space for VMP result normalized
        let (mut res_small, scratch2) = scratch1.tmp_vec_znx(module, cols, rhs.size());
        (0..cols).for_each(|i| {
            module.vec_znx_big_normalize(basek, &mut res_small, i, &res_big, i, scratch2);
            module.vec_znx_dft(self, i, &res_small, i);
        });
    }

    pub fn external_product_inplace<DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        unsafe {
            let self_ptr: *mut GLWECiphertextFourier<DataSelf, FFT64> = self as *mut GLWECiphertextFourier<DataSelf, FFT64>;
            self.external_product(&module, &*self_ptr, rhs, scratch);
        }
    }
}

impl<DataSelf> GLWECiphertextFourier<DataSelf, FFT64>
where
    VecZnxDft<DataSelf, FFT64>: VecZnxDftToRef<FFT64>,
{
    pub fn decrypt<DataPt, DataSk>(
        &self,
        module: &Module<FFT64>,
        pt: &mut GLWEPlaintext<DataPt>,
        sk_dft: &SecretKeyFourier<DataSk, FFT64>,
        scratch: &mut Scratch,
    ) where
        VecZnx<DataPt>: VecZnxToMut + VecZnxToRef,
        ScalarZnxDft<DataSk, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), sk_dft.rank());
            assert_eq!(self.n(), module.n());
            assert_eq!(pt.n(), module.n());
            assert_eq!(sk_dft.n(), module.n());
        }

        let cols = self.rank() + 1;

        let (mut pt_big, scratch_1) = scratch.tmp_vec_znx_big(module, 1, self.size()); // TODO optimize size when pt << ct
        pt_big.zero();

        {
            (1..cols).for_each(|i| {
                let (mut ci_dft, _) = scratch_1.tmp_vec_znx_dft(module, 1, self.size()); // TODO optimize size when pt << ct
                module.svp_apply(&mut ci_dft, 0, sk_dft, i - 1, self, i);
                let ci_big: VecZnxBig<&mut [u8], FFT64> = module.vec_znx_idft_consume(ci_dft);
                module.vec_znx_big_add_inplace(&mut pt_big, 0, &ci_big, 0);
            });
        }

        {
            let (mut c0_big, scratch_2) = scratch_1.tmp_vec_znx_big(module, 1, self.size());
            // c0_big = (a * s) + (-a * s + m + e) = BIG(m + e)
            module.vec_znx_idft(&mut c0_big, 0, self, 0, scratch_2);
            module.vec_znx_big_add_inplace(&mut pt_big, 0, &c0_big, 0);
        }

        // pt = norm(BIG(m + e))
        module.vec_znx_big_normalize(self.basek(), pt, 0, &mut pt_big, 0, scratch_1);

        pt.basek = self.basek();
        pt.k = pt.k().min(self.k());
    }

    #[allow(dead_code)]
    pub(crate) fn idft<DataRes>(&self, module: &Module<FFT64>, res: &mut GLWECiphertext<DataRes>, scratch: &mut Scratch)
    where
        GLWECiphertext<DataRes>: VecZnxToMut,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), res.rank());
            assert_eq!(self.basek(), res.basek())
        }

        let min_size: usize = self.size().min(res.size());

        let (mut res_big, scratch1) = scratch.tmp_vec_znx_big(module, 1, min_size);

        (0..self.rank() + 1).for_each(|i| {
            module.vec_znx_idft(&mut res_big, 0, self, i, scratch1);
            module.vec_znx_big_normalize(self.basek(), res, i, &res_big, 0, scratch1);
        });
    }
}
