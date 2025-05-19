use base2k::{
    AddNormal, Backend, FFT64, FillUniform, MatZnxDft, MatZnxDftOps, MatZnxDftScratch, MatZnxDftToRef, Module, ScalarZnxAlloc,
    ScalarZnxDft, ScalarZnxDftAlloc, ScalarZnxDftOps, ScalarZnxDftToRef, Scratch, VecZnx, VecZnxAlloc, VecZnxBig, VecZnxBigAlloc,
    VecZnxBigOps, VecZnxBigScratch, VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxDftToMut, VecZnxDftToRef, VecZnxOps,
    VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxZero,
};
use sampling::source::Source;

use crate::{
    SIX_SIGMA,
    automorphism::AutomorphismKey,
    elem::Infos,
    ggsw_ciphertext::GGSWCiphertext,
    glwe_ciphertext_fourier::GLWECiphertextFourier,
    glwe_plaintext::GLWEPlaintext,
    keys::{GLWEPublicKey, SecretDistribution, SecretKeyFourier},
    keyswitch_key::GLWESwitchingKey,
    utils::derive_size,
};

pub struct GLWECiphertext<C> {
    pub data: VecZnx<C>,
    pub basek: usize,
    pub k: usize,
}

impl GLWECiphertext<Vec<u8>> {
    pub fn new<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank: usize) -> Self {
        Self {
            data: module.new_vec_znx(rank + 1, derive_size(basek, k)),
            basek,
            k,
        }
    }
}

impl<T> Infos for GLWECiphertext<T> {
    type Inner = VecZnx<T>;

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

impl<T> GLWECiphertext<T> {
    pub fn rank(&self) -> usize {
        self.cols() - 1
    }
}

impl<C> VecZnxToMut for GLWECiphertext<C>
where
    VecZnx<C>: VecZnxToMut,
{
    fn to_mut(&mut self) -> VecZnx<&mut [u8]> {
        self.data.to_mut()
    }
}

impl<C> VecZnxToRef for GLWECiphertext<C>
where
    VecZnx<C>: VecZnxToRef,
{
    fn to_ref(&self) -> VecZnx<&[u8]> {
        self.data.to_ref()
    }
}

impl<C> GLWECiphertext<C>
where
    VecZnx<C>: VecZnxToRef,
{
    #[allow(dead_code)]
    pub(crate) fn dft<R>(&self, module: &Module<FFT64>, res: &mut GLWECiphertextFourier<R, FFT64>)
    where
        VecZnxDft<R, FFT64>: VecZnxDftToMut<FFT64> + ZnxInfos,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), res.rank());
            assert_eq!(self.basek(), res.basek())
        }

        (0..self.rank() + 1).for_each(|i| {
            module.vec_znx_dft(res, i, self, i);
        })
    }
}

impl GLWECiphertext<Vec<u8>> {
    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, ct_size: usize) -> usize {
        module.vec_znx_big_normalize_tmp_bytes()
            + module.bytes_of_vec_znx_dft(1, ct_size)
            + module.bytes_of_vec_znx_big(1, ct_size)
    }
    pub fn encrypt_pk_scratch_space(module: &Module<FFT64>, pk_size: usize) -> usize {
        ((module.bytes_of_vec_znx_dft(1, pk_size) + module.bytes_of_vec_znx_big(1, pk_size)) | module.bytes_of_scalar_znx(1))
            + module.bytes_of_scalar_znx_dft(1)
            + module.vec_znx_big_normalize_tmp_bytes()
    }

    pub fn decrypt_scratch_space(module: &Module<FFT64>, ct_size: usize) -> usize {
        (module.vec_znx_big_normalize_tmp_bytes() | module.bytes_of_vec_znx_dft(1, ct_size))
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
        let res_dft: usize = module.bytes_of_vec_znx_dft(out_rank + 1, ksk_size);
        let vmp: usize = module.vmp_apply_tmp_bytes(out_size, in_size, in_size, in_rank, out_rank + 1, ksk_size)
            + module.bytes_of_vec_znx_dft(in_rank, in_size);
        let normalize: usize = module.vec_znx_big_normalize_tmp_bytes();

        return res_dft + (vmp | normalize);
    }

    pub fn keyswitch_from_fourier_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        out_rank: usize,
        in_size: usize,
        in_rank: usize,
        ksk_size: usize,
    ) -> usize {
        let res_dft = module.bytes_of_vec_znx_dft(out_rank + 1, ksk_size);

        let vmp: usize = module.vmp_apply_tmp_bytes(out_size, in_size, in_size, in_rank, out_rank + 1, ksk_size)
            + module.bytes_of_vec_znx_dft(in_rank, in_size);

        let a0_big: usize = module.bytes_of_vec_znx_big(1, in_size) + module.vec_znx_idft_tmp_bytes();

        let norm: usize = module.vec_znx_big_normalize_tmp_bytes();

        res_dft + (vmp | a0_big | norm)
    }

    pub fn keyswitch_inplace_scratch_space(module: &Module<FFT64>, out_size: usize, out_rank: usize, ksk_size: usize) -> usize {
        GLWECiphertext::keyswitch_scratch_space(module, out_size, out_rank, out_size, out_rank, ksk_size)
    }

    pub fn automorphism_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        out_rank: usize,
        in_size: usize,
        autokey_size: usize,
    ) -> usize {
        GLWECiphertext::keyswitch_scratch_space(module, out_size, out_rank, in_size, out_rank, autokey_size)
    }

    pub fn automorphism_inplace_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        out_rank: usize,
        autokey_size: usize,
    ) -> usize {
        GLWECiphertext::keyswitch_scratch_space(module, out_size, out_rank, out_size, out_rank, autokey_size)
    }

    pub fn external_product_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        out_rank: usize,
        in_size: usize,
        ggsw_size: usize,
    ) -> usize {
        let res_dft: usize = module.bytes_of_vec_znx_dft(out_rank + 1, ggsw_size);
        let vmp: usize = module.bytes_of_vec_znx_dft(out_rank + 1, in_size)
            + module.vmp_apply_tmp_bytes(
                out_size,
                in_size,
                in_size,      // rows
                out_rank + 1, // cols in
                out_rank + 1, // cols out
                ggsw_size,
            );
        let normalize: usize = module.vec_znx_big_normalize_tmp_bytes();

        res_dft + (vmp | normalize)
    }

    pub fn external_product_inplace_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        out_rank: usize,
        ggsw_size: usize,
    ) -> usize {
        GLWECiphertext::external_product_scratch_space(module, out_size, out_rank, out_size, ggsw_size)
    }
}

impl<DataSelf> GLWECiphertext<DataSelf>
where
    VecZnx<DataSelf>: VecZnxToMut + VecZnxToRef,
{
    pub fn encrypt_sk<DataPt, DataSk>(
        &mut self,
        module: &Module<FFT64>,
        pt: &GLWEPlaintext<DataPt>,
        sk_dft: &SecretKeyFourier<DataSk, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        VecZnx<DataPt>: VecZnxToRef,
        ScalarZnxDft<DataSk, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        self.encrypt_sk_private(
            module,
            Some((pt, 0)),
            sk_dft,
            source_xa,
            source_xe,
            sigma,
            scratch,
        );
    }

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
        self.encrypt_sk_private(module, None, sk_dft, source_xa, source_xe, sigma, scratch);
    }

    pub fn encrypt_pk<DataPt, DataPk>(
        &mut self,
        module: &Module<FFT64>,
        pt: &GLWEPlaintext<DataPt>,
        pk: &GLWEPublicKey<DataPk, FFT64>,
        source_xu: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        VecZnx<DataPt>: VecZnxToRef,
        VecZnxDft<DataPk, FFT64>: VecZnxDftToRef<FFT64>,
    {
        self.encrypt_pk_private(
            module,
            Some((pt, 0)),
            pk,
            source_xu,
            source_xe,
            sigma,
            scratch,
        );
    }

    pub fn encrypt_zero_pk<DataPk>(
        &mut self,
        module: &Module<FFT64>,
        pk: &GLWEPublicKey<DataPk, FFT64>,
        source_xu: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        VecZnxDft<DataPk, FFT64>: VecZnxDftToRef<FFT64>,
    {
        self.encrypt_pk_private(module, None, pk, source_xu, source_xe, sigma, scratch);
    }

    pub fn automorphism<DataLhs, DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        VecZnx<DataLhs>: VecZnxToRef,
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        self.keyswitch(module, lhs, &rhs.key, scratch);
        (0..self.rank() + 1).for_each(|i| {
            module.vec_znx_automorphism_inplace(rhs.p(), self, i);
        })
    }

    pub fn automorphism_inplace<DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        self.keyswitch_inplace(module, &rhs.key, scratch);
        (0..self.rank() + 1).for_each(|i| {
            module.vec_znx_automorphism_inplace(rhs.p(), self, i);
        })
    }

    pub(crate) fn keyswitch_from_fourier<DataLhs, DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWECiphertextFourier<DataLhs, FFT64>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        VecZnxDft<DataLhs, FFT64>: VecZnxDftToRef<FFT64>,
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        let basek: usize = self.basek();

        #[cfg(debug_assertions)]
        {
            assert_eq!(lhs.rank(), rhs.rank_in());
            assert_eq!(self.rank(), rhs.rank_out());
            assert_eq!(self.basek(), basek);
            assert_eq!(lhs.basek(), basek);
            assert_eq!(rhs.n(), module.n());
            assert_eq!(self.n(), module.n());
            assert_eq!(lhs.n(), module.n());
            assert!(
                scratch.available()
                    >= GLWECiphertext::keyswitch_from_fourier_scratch_space(
                        module,
                        self.size(),
                        self.rank(),
                        lhs.size(),
                        lhs.rank(),
                        rhs.size(),
                    )
            );
        }

        let cols_in: usize = rhs.rank_in();
        let cols_out: usize = rhs.rank_out() + 1;

        // Buffer of the result of VMP in DFT
        let (mut res_dft, scratch1) = scratch.tmp_vec_znx_dft(module, cols_out, rhs.size()); // Todo optimise

        {
            // Applies VMP
            let (mut ai_dft, scratch2) = scratch1.tmp_vec_znx_dft(module, cols_in, lhs.size());
            (0..cols_in).for_each(|col_i| {
                module.vec_znx_dft_copy(&mut ai_dft, col_i, lhs, col_i + 1);
            });
            module.vmp_apply(&mut res_dft, &ai_dft, rhs, scratch2);
        }

        // Switches result of VMP outside of DFT
        let mut res_big: VecZnxBig<&mut [u8], FFT64> = module.vec_znx_idft_consume::<&mut [u8]>(res_dft);

        {
            // Switches lhs 0-th outside of DFT domain and adds on
            let (mut a0_big, scratch2) = scratch1.tmp_vec_znx_big(module, 1, lhs.size());
            module.vec_znx_idft(&mut a0_big, 0, lhs, 0, scratch2);
            module.vec_znx_big_add_inplace(&mut res_big, 0, &a0_big, 0);
        }

        (0..cols_out).for_each(|i| {
            module.vec_znx_big_normalize(basek, self, i, &res_big, i, scratch1);
        });
    }

    pub fn keyswitch<DataLhs, DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        VecZnx<DataLhs>: VecZnxToRef,
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        let basek: usize = self.basek();

        #[cfg(debug_assertions)]
        {
            assert_eq!(lhs.rank(), rhs.rank_in());
            assert_eq!(self.rank(), rhs.rank_out());
            assert_eq!(self.basek(), basek);
            assert_eq!(lhs.basek(), basek);
            assert_eq!(rhs.n(), module.n());
            assert_eq!(self.n(), module.n());
            assert_eq!(lhs.n(), module.n());
            assert!(
                scratch.available()
                    >= GLWECiphertext::keyswitch_scratch_space(
                        module,
                        self.size(),
                        self.rank(),
                        lhs.size(),
                        lhs.rank(),
                        rhs.size(),
                    )
            );
        }

        let cols_in: usize = rhs.rank_in();
        let cols_out: usize = rhs.rank_out() + 1;

        let (mut res_dft, scratch1) = scratch.tmp_vec_znx_dft(module, cols_out, rhs.size()); // Todo optimise

        {
            let (mut ai_dft, scratch2) = scratch1.tmp_vec_znx_dft(module, cols_in, lhs.size());
            (0..cols_in).for_each(|col_i| {
                module.vec_znx_dft(&mut ai_dft, col_i, lhs, col_i + 1);
            });
            module.vmp_apply(&mut res_dft, &ai_dft, rhs, scratch2);
        }

        let mut res_big: VecZnxBig<&mut [u8], FFT64> = module.vec_znx_idft_consume(res_dft);

        module.vec_znx_big_add_small_inplace(&mut res_big, 0, lhs, 0);

        (0..cols_out).for_each(|i| {
            module.vec_znx_big_normalize(basek, self, i, &res_big, i, scratch1);
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
            let self_ptr: *mut GLWECiphertext<DataSelf> = self as *mut GLWECiphertext<DataSelf>;
            self.keyswitch(&module, &*self_ptr, rhs, scratch);
        }
    }

    pub fn external_product<DataLhs, DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        VecZnx<DataLhs>: VecZnxToRef,
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

        let (mut res_dft, scratch1) = scratch.tmp_vec_znx_dft(module, cols, rhs.size()); // Todo optimise

        {
            let (mut a_dft, scratch2) = scratch1.tmp_vec_znx_dft(module, cols, lhs.size());
            (0..cols).for_each(|col_i| {
                module.vec_znx_dft(&mut a_dft, col_i, lhs, col_i);
            });
            module.vmp_apply(&mut res_dft, &a_dft, rhs, scratch2);
        }

        let res_big: VecZnxBig<&mut [u8], FFT64> = module.vec_znx_idft_consume(res_dft);

        (0..cols).for_each(|i| {
            module.vec_znx_big_normalize(basek, self, i, &res_big, i, scratch1);
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
            let self_ptr: *mut GLWECiphertext<DataSelf> = self as *mut GLWECiphertext<DataSelf>;
            self.external_product(&module, &*self_ptr, rhs, scratch);
        }
    }

    pub(crate) fn encrypt_sk_private<DataPt, DataSk>(
        &mut self,
        module: &Module<FFT64>,
        pt: Option<(&GLWEPlaintext<DataPt>, usize)>,
        sk_dft: &SecretKeyFourier<DataSk, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        VecZnx<DataPt>: VecZnxToRef,
        ScalarZnxDft<DataSk, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), sk_dft.rank());
            assert_eq!(sk_dft.n(), module.n());
            assert_eq!(self.n(), module.n());
            if let Some((pt, col)) = pt {
                assert_eq!(pt.n(), module.n());
                assert!(col < self.rank() + 1);
            }
        }

        let log_base2k: usize = self.basek();
        let log_k: usize = self.k();
        let size: usize = self.size();
        let cols: usize = self.rank() + 1;

        let (mut c0_big, scratch_1) = scratch.tmp_vec_znx(module, 1, size);
        c0_big.zero();

        {
            // c[i] = uniform
            // c[0] -= c[i] * s[i],
            (1..cols).for_each(|i| {
                let (mut ci_dft, scratch_2) = scratch_1.tmp_vec_znx_dft(module, 1, size);

                // c[i] = uniform
                self.data.fill_uniform(log_base2k, i, size, source_xa);

                // c[i] = norm(IDFT(DFT(c[i]) * DFT(s[i])))
                module.vec_znx_dft(&mut ci_dft, 0, self, i);
                module.svp_apply_inplace(&mut ci_dft, 0, sk_dft, i - 1);
                let ci_big: VecZnxBig<&mut [u8], FFT64> = module.vec_znx_idft_consume(ci_dft);

                // use c[0] as buffer, which is overwritten later by the normalization step
                module.vec_znx_big_normalize(log_base2k, self, 0, &ci_big, 0, scratch_2);

                // c0_tmp = -c[i] * s[i] (use c[0] as buffer)
                module.vec_znx_sub_ab_inplace(&mut c0_big, 0, self, 0);

                // c[i] += m if col = i
                if let Some((pt, col)) = pt {
                    if i == col {
                        module.vec_znx_add_inplace(self, i, pt, 0);
                        module.vec_znx_normalize_inplace(log_base2k, self, i, scratch_2);
                    }
                }
            });
        }

        // c[0] += e
        c0_big.add_normal(log_base2k, 0, log_k, source_xe, sigma, sigma * SIX_SIGMA);

        // c[0] += m if col = 0
        if let Some((pt, col)) = pt {
            if col == 0 {
                module.vec_znx_add_inplace(&mut c0_big, 0, pt, 0);
            }
        }

        // c[0] = norm(c[0])
        module.vec_znx_normalize(log_base2k, self, 0, &c0_big, 0, scratch_1);
    }

    pub(crate) fn encrypt_pk_private<DataPt, DataPk>(
        &mut self,
        module: &Module<FFT64>,
        pt: Option<(&GLWEPlaintext<DataPt>, usize)>,
        pk: &GLWEPublicKey<DataPk, FFT64>,
        source_xu: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        VecZnx<DataPt>: VecZnxToRef,
        VecZnxDft<DataPk, FFT64>: VecZnxDftToRef<FFT64>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.basek(), pk.basek());
            assert_eq!(self.n(), module.n());
            assert_eq!(pk.n(), module.n());
            assert_eq!(self.rank(), pk.rank());
            if let Some((pt, _)) = pt {
                assert_eq!(pt.basek(), pk.basek());
                assert_eq!(pt.n(), module.n());
            }
        }

        let log_base2k: usize = pk.basek();
        let size_pk: usize = pk.size();
        let cols: usize = self.rank() + 1;

        // Generates u according to the underlying secret distribution.
        let (mut u_dft, scratch_1) = scratch.tmp_scalar_znx_dft(module, 1);

        {
            let (mut u, _) = scratch_1.tmp_scalar_znx(module, 1);
            match pk.dist {
                SecretDistribution::NONE => panic!(
                    "invalid public key: SecretDistribution::NONE, ensure it has been correctly intialized through \
                     Self::generate"
                ),
                SecretDistribution::TernaryFixed(hw) => u.fill_ternary_hw(0, hw, source_xu),
                SecretDistribution::TernaryProb(prob) => u.fill_ternary_prob(0, prob, source_xu),
                SecretDistribution::ZERO => {}
            }

            module.svp_prepare(&mut u_dft, 0, &u, 0);
        }

        // ct[i] = pk[i] * u + ei (+ m if col = i)
        (0..cols).for_each(|i| {
            let (mut ci_dft, scratch_2) = scratch_1.tmp_vec_znx_dft(module, 1, size_pk);
            // ci_dft = DFT(u) * DFT(pk[i])
            module.svp_apply(&mut ci_dft, 0, &u_dft, 0, pk, i);

            // ci_big = u * p[i]
            let mut ci_big = module.vec_znx_idft_consume(ci_dft);

            // ci_big = u * pk[i] + e
            ci_big.add_normal(log_base2k, 0, pk.k(), source_xe, sigma, sigma * SIX_SIGMA);

            // ci_big = u * pk[i] + e + m (if col = i)
            if let Some((pt, col)) = pt {
                if col == i {
                    module.vec_znx_big_add_small_inplace(&mut ci_big, 0, pt, 0);
                }
            }

            // ct[i] = norm(ci_big)
            module.vec_znx_big_normalize(log_base2k, self, i, &ci_big, 0, scratch_2);
        });
    }
}

impl<DataSelf> GLWECiphertext<DataSelf>
where
    VecZnx<DataSelf>: VecZnxToRef,
{
    pub fn decrypt<DataPt, DataSk>(
        &self,
        module: &Module<FFT64>,
        pt: &mut GLWEPlaintext<DataPt>,
        sk_dft: &SecretKeyFourier<DataSk, FFT64>,
        scratch: &mut Scratch,
    ) where
        VecZnx<DataPt>: VecZnxToMut,
        ScalarZnxDft<DataSk, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), sk_dft.rank());
            assert_eq!(self.n(), module.n());
            assert_eq!(pt.n(), module.n());
            assert_eq!(sk_dft.n(), module.n());
        }

        let cols: usize = self.rank() + 1;

        let (mut c0_big, scratch_1) = scratch.tmp_vec_znx_big(module, 1, self.size()); // TODO optimize size when pt << ct
        c0_big.zero();

        {
            (1..cols).for_each(|i| {
                // ci_dft = DFT(a[i]) * DFT(s[i])
                let (mut ci_dft, _) = scratch_1.tmp_vec_znx_dft(module, 1, self.size()); // TODO optimize size when pt << ct
                module.vec_znx_dft(&mut ci_dft, 0, self, i);
                module.svp_apply_inplace(&mut ci_dft, 0, sk_dft, i - 1);
                let ci_big = module.vec_znx_idft_consume(ci_dft);

                // c0_big += a[i] * s[i]
                module.vec_znx_big_add_inplace(&mut c0_big, 0, &ci_big, 0);
            });
        }

        // c0_big = (a * s) + (-a * s + m + e) = BIG(m + e)
        module.vec_znx_big_add_small_inplace(&mut c0_big, 0, self, 0);

        // pt = norm(BIG(m + e))
        module.vec_znx_big_normalize(self.basek(), pt, 0, &mut c0_big, 0, scratch_1);

        pt.basek = self.basek();
        pt.k = pt.k().min(self.k());
    }
}
