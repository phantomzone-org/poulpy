use base2k::{
    AddNormal, Backend, FFT64, FillUniform, MatZnxDft, MatZnxDftToRef, Module, ScalarZnxAlloc, ScalarZnxDft, ScalarZnxDftAlloc,
    ScalarZnxDftOps, ScalarZnxDftToRef, Scratch, VecZnx, VecZnxAlloc, VecZnxBig, VecZnxBigAlloc, VecZnxBigOps, VecZnxBigScratch,
    VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxDftToMut, VecZnxDftToRef, VecZnxOps, VecZnxToMut, VecZnxToRef, ZnxInfos,
    ZnxZero,
};
use sampling::source::Source;

use crate::{
    SIX_SIGMA,
    elem::Infos,
    gglwe_ciphertext::GGLWECiphertext,
    ggsw_ciphertext::GGSWCiphertext,
    glwe_ciphertext_fourier::GLWECiphertextFourier,
    glwe_plaintext::GLWEPlaintext,
    keys::{GLWEPublicKey, SecretDistribution, SecretKeyFourier},
    keyswitch_key::GLWESwitchingKey,
    utils::derive_size,
    vec_glwe_product::{VecGLWEProduct, VecGLWEProductScratchSpace},
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
    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, _rank: usize, ct_size: usize) -> usize {
        module.vec_znx_big_normalize_tmp_bytes()
            + module.bytes_of_vec_znx_dft(1, ct_size)
            + module.bytes_of_vec_znx_big(1, ct_size)
    }
    pub fn encrypt_pk_scratch_space(module: &Module<FFT64>, _rank: usize, pk_size: usize) -> usize {
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
        res_size: usize,
        lhs: usize,
        rhs: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize {
        <GGLWECiphertext<Vec<u8>, FFT64> as VecGLWEProductScratchSpace>::prod_with_glwe_scratch_space(
            module, res_size, lhs, rhs, rank_in, rank_out,
        )
    }

    pub fn keyswitch_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, rhs: usize, rank: usize) -> usize {
        <GGLWECiphertext<Vec<u8>, FFT64> as VecGLWEProductScratchSpace>::prod_with_glwe_inplace_scratch_space(
            module, res_size, rhs, rank,
        )
    }

    pub fn external_product_scratch_space(module: &Module<FFT64>, res_size: usize, lhs: usize, rhs: usize, rank: usize) -> usize {
        <GGSWCiphertext<Vec<u8>, FFT64> as VecGLWEProductScratchSpace>::prod_with_glwe_scratch_space(
            module, res_size, lhs, rhs, rank, rank,
        )
    }

    pub fn external_product_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, rhs: usize, rank: usize) -> usize {
        <GGSWCiphertext<Vec<u8>, FFT64> as VecGLWEProductScratchSpace>::prod_with_glwe_inplace_scratch_space(
            module, res_size, rhs, rank,
        )
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
        rhs.0.prod_with_glwe(module, self, lhs, scratch);
    }

    pub fn keyswitch_inplace<DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        rhs.0.prod_with_glwe_inplace(module, self, scratch);
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
        rhs.prod_with_glwe(module, self, lhs, scratch);
    }

    pub fn external_product_inplace<DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        rhs.prod_with_glwe_inplace(module, self, scratch);
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
