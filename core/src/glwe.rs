use base2k::{
    AddNormal, Backend, FFT64, FillUniform, MatZnxDft, MatZnxDftToRef, Module, ScalarZnxAlloc, ScalarZnxDft, ScalarZnxDftAlloc,
    ScalarZnxDftOps, ScalarZnxDftToRef, Scratch, VecZnx, VecZnxAlloc, VecZnxBigAlloc, VecZnxBigOps, VecZnxBigScratch, VecZnxDft,
    VecZnxDftAlloc, VecZnxDftOps, VecZnxDftToMut, VecZnxDftToRef, VecZnxOps, VecZnxToMut, VecZnxToRef, ZnxInfos,
};
use sampling::source::Source;

use crate::{
    elem::Infos,
    encryption::{EncryptSk, EncryptSkScratchSpace, EncryptZeroSkScratchSpace},
    external_product::{
        ExternalProduct, ExternalProductInplace, ExternalProductInplaceScratchSpace, ExternalProductScratchSpace,
    },
    ggsw::GGSWCiphertext,
    keys::{PublicKey, SecretDistribution, SecretKeyFourier},
    keyswitch::{KeySwitch, KeySwitchInplace, KeySwitchInplaceScratchSpace, KeySwitchScratchSpace},
    keyswitch_key::GLWEKeySwitchKey,
    utils::derive_size,
    vec_glwe_product::{VecGLWEProduct, VecGLWEProductScratchSpace},
};

pub struct GLWECiphertext<C> {
    pub data: VecZnx<C>,
    pub log_base2k: usize,
    pub log_k: usize,
}

impl GLWECiphertext<Vec<u8>> {
    pub fn new<B: Backend>(module: &Module<B>, log_base2k: usize, log_k: usize) -> Self {
        Self {
            data: module.new_vec_znx(2, derive_size(log_base2k, log_k)),
            log_base2k: log_base2k,
            log_k: log_k,
        }
    }
}

impl<T> Infos for GLWECiphertext<T> {
    type Inner = VecZnx<T>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.log_base2k
    }

    fn k(&self) -> usize {
        self.log_k
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
            assert_eq!(self.rank(), 2);
            assert_eq!(res.rank(), 2);
            assert_eq!(self.basek(), res.basek())
        }

        module.vec_znx_dft(res, 0, self, 0);
        module.vec_znx_dft(res, 1, self, 1);
    }
}

impl KeySwitchScratchSpace for GLWECiphertext<Vec<u8>> {
    fn keyswitch_scratch_space(module: &Module<FFT64>, res_size: usize, lhs: usize, rhs: usize) -> usize {
        <GLWEKeySwitchKey<Vec<u8>, FFT64> as VecGLWEProductScratchSpace>::prod_with_glwe_scratch_space(module, res_size, lhs, rhs)
    }
}

impl<DataSelf, DataLhs, DataRhs> KeySwitch<DataLhs, DataRhs> for GLWECiphertext<DataSelf>
where
    VecZnx<DataSelf>: VecZnxToMut + VecZnxToRef,
    VecZnx<DataLhs>: VecZnxToRef,
    MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
{
    type Lhs = GLWECiphertext<DataLhs>;
    type Rhs = GLWEKeySwitchKey<DataRhs, FFT64>;

    fn keyswitch(&mut self, module: &Module<FFT64>, lhs: &Self::Lhs, rhs: &Self::Rhs, scratch: &mut Scratch) {
        rhs.prod_with_glwe(module, self, lhs, scratch);
    }
}

impl KeySwitchInplaceScratchSpace for GLWECiphertext<Vec<u8>> {
    fn keyswitch_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, rhs: usize) -> usize {
        <GLWEKeySwitchKey<Vec<u8>, FFT64> as VecGLWEProductScratchSpace>::prod_with_glwe_inplace_scratch_space(
            module, res_size, rhs,
        )
    }
}

impl<DataSelf, DataRhs> KeySwitchInplace<DataRhs> for GLWECiphertext<DataSelf>
where
    VecZnx<DataSelf>: VecZnxToMut + VecZnxToRef,
    MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
{
    type Rhs = GLWEKeySwitchKey<DataRhs, FFT64>;

    fn keyswitch_inplace(&mut self, module: &Module<FFT64>, rhs: &Self::Rhs, scratch: &mut Scratch) {
        rhs.prod_with_glwe_inplace(module, self, scratch);
    }
}

impl ExternalProductScratchSpace for GLWECiphertext<Vec<u8>> {
    fn external_product_scratch_space(module: &Module<FFT64>, res_size: usize, lhs: usize, rhs: usize) -> usize {
        <GGSWCiphertext<Vec<u8>, FFT64> as VecGLWEProductScratchSpace>::prod_with_glwe_scratch_space(module, res_size, lhs, rhs)
    }
}

impl<DataSelf, DataLhs, DataRhs> ExternalProduct<DataLhs, DataRhs> for GLWECiphertext<DataSelf>
where
    VecZnx<DataSelf>: VecZnxToMut + VecZnxToRef,
    VecZnx<DataLhs>: VecZnxToRef,
    MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
{
    type Lhs = GLWECiphertext<DataLhs>;
    type Rhs = GGSWCiphertext<DataRhs, FFT64>;

    fn external_product(&mut self, module: &Module<FFT64>, lhs: &Self::Lhs, rhs: &Self::Rhs, scratch: &mut Scratch) {
        rhs.prod_with_glwe(module, self, lhs, scratch);
    }
}

impl ExternalProductInplaceScratchSpace for GLWECiphertext<Vec<u8>> {
    fn external_product_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, rhs: usize) -> usize {
        <GGSWCiphertext<Vec<u8>, FFT64> as VecGLWEProductScratchSpace>::prod_with_glwe_inplace_scratch_space(
            module, res_size, rhs,
        )
    }
}

impl<DataSelf, DataRhs> ExternalProductInplace<DataRhs> for GLWECiphertext<DataSelf>
where
    VecZnx<DataSelf>: VecZnxToMut + VecZnxToRef,
    MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64> + ZnxInfos,
{
    type Rhs = GGSWCiphertext<DataRhs, FFT64>;

    fn external_product_inplace(&mut self, module: &Module<FFT64>, rhs: &Self::Rhs, scratch: &mut Scratch) {
        rhs.prod_with_glwe_inplace(module, self, scratch);
    }
}

impl GLWECiphertext<Vec<u8>> {
    pub fn encrypt_pk_scratch_space<B: Backend>(module: &Module<B>, pk_size: usize) -> usize {
        ((module.bytes_of_vec_znx_dft(1, pk_size) + module.bytes_of_vec_znx_big(1, pk_size)) | module.bytes_of_scalar_znx(1))
            + module.bytes_of_scalar_znx_dft(1)
            + module.vec_znx_big_normalize_tmp_bytes()
    }

    pub fn decrypt_scratch_space<B: Backend>(module: &Module<B>, size: usize) -> usize {
        (module.vec_znx_big_normalize_tmp_bytes() | module.bytes_of_vec_znx_dft(1, size)) + module.bytes_of_vec_znx_big(1, size)
    }
}

impl EncryptSkScratchSpace for GLWECiphertext<Vec<u8>> {
    fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, size: usize) -> usize {
        (module.vec_znx_big_normalize_tmp_bytes() | module.bytes_of_vec_znx_dft(1, size)) + module.bytes_of_vec_znx_big(1, size)
    }
}

impl<DataCt, DataPt, DataSk> EncryptSk<DataCt, DataPt, DataSk, FFT64> for GLWECiphertext<DataCt>
where
    VecZnx<DataCt>: VecZnxToMut + VecZnxToRef,
    VecZnx<DataPt>: VecZnxToRef,
    ScalarZnxDft<DataSk, FFT64>: ScalarZnxDftToRef<FFT64>,
{
    type Ciphertext = GLWECiphertext<DataCt>;
    type Plaintext = GLWEPlaintext<DataPt>;
    type SecretKey = SecretKeyFourier<DataSk, FFT64>;

    fn encrypt_sk(
        &self,
        module: &Module<FFT64>,
        ct: &mut Self::Ciphertext,
        pt: &Self::Plaintext,
        sk: &Self::SecretKey,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        bound: f64,
        scratch: &mut Scratch,
    ) {
        encrypt_glwe_sk(
            module,
            ct,
            Some((pt, 0)),
            sk,
            source_xa,
            source_xe,
            sigma,
            bound,
            scratch,
        );
    }
}

pub(crate) fn encrypt_glwe_sk<DataCt, DataPt, DataSk>(
    module: &Module<FFT64>,
    ct: &mut GLWECiphertext<DataCt>,
    pt: Option<(&GLWEPlaintext<DataPt>, usize)>,
    sk_dft: &SecretKeyFourier<DataSk, FFT64>,
    source_xa: &mut Source,
    source_xe: &mut Source,
    sigma: f64,
    bound: f64,
    scratch: &mut Scratch,
) where
    VecZnx<DataCt>: VecZnxToMut + VecZnxToRef,
    VecZnx<DataPt>: VecZnxToRef,
    ScalarZnxDft<DataSk, FFT64>: ScalarZnxDftToRef<FFT64>,
{
    let log_base2k: usize = ct.basek();
    let log_k: usize = ct.k();
    let size: usize = ct.size();

    // c1 = a
    ct.data.fill_uniform(log_base2k, 1, size, source_xa);

    let (mut c0_big, scratch_1) = scratch.tmp_vec_znx_big(module, 1, size);

    {
        let (mut c0_dft, _) = scratch_1.tmp_vec_znx_dft(module, 1, size);
        module.vec_znx_dft(&mut c0_dft, 0, ct, 1);

        // c0_dft = DFT(a) * DFT(s)
        module.svp_apply_inplace(&mut c0_dft, 0, sk_dft, 0);

        // c0_big = IDFT(c0_dft)
        module.vec_znx_idft_tmp_a(&mut c0_big, 0, &mut c0_dft, 0);
    }

    // c0_big = m - c0_big
    if let Some((pt, col)) = pt {
        match col {
            0 => module.vec_znx_big_sub_small_b_inplace(&mut c0_big, 0, pt, 0),
            1 => {
                module.vec_znx_big_negate_inplace(&mut c0_big, 0);
                module.vec_znx_add_inplace(ct, 1, pt, 0);
                module.vec_znx_normalize_inplace(log_base2k, ct, 1, scratch_1);
            }
            _ => panic!("invalid target column: {}", col),
        }
    } else {
        module.vec_znx_big_negate_inplace(&mut c0_big, 0);
    }
    // c0_big += e
    c0_big.add_normal(log_base2k, 0, log_k, source_xe, sigma, bound);

    // c0 = norm(c0_big = -as + m + e)
    module.vec_znx_big_normalize(log_base2k, ct, 0, &c0_big, 0, scratch_1);
}

pub fn decrypt_glwe<P, C, S>(
    module: &Module<FFT64>,
    pt: &mut GLWEPlaintext<P>,
    ct: &GLWECiphertext<C>,
    sk_dft: &SecretKeyFourier<S, FFT64>,
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
        module.svp_apply_inplace(&mut c0_dft, 0, sk_dft, 0);

        // c0_big = IDFT(c0_dft)
        module.vec_znx_idft_tmp_a(&mut c0_big, 0, &mut c0_dft, 0);
    }

    // c0_big = (a * s) + (-a * s + m + e) = BIG(m + e)
    module.vec_znx_big_add_small_inplace(&mut c0_big, 0, ct, 0);

    // pt = norm(BIG(m + e))
    module.vec_znx_big_normalize(ct.basek(), pt, 0, &mut c0_big, 0, scratch_1);

    pt.log_base2k = ct.basek();
    pt.log_k = pt.k().min(ct.k());
}

impl<MUT> GLWECiphertext<MUT>
where
    VecZnx<MUT>: VecZnxToMut + VecZnxToRef,
{
    pub fn encrypt_sk<R0, R1>(
        &mut self,
        module: &Module<FFT64>,
        pt: &GLWEPlaintext<R0>,
        sk_dft: &SecretKeyFourier<R1, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        bound: f64,
        scratch: &mut Scratch,
    ) where
        VecZnx<R0>: VecZnxToRef,
        ScalarZnxDft<R1, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        encrypt_glwe_sk(
            module,
            self,
            Some((pt, 0)),
            sk_dft,
            source_xa,
            source_xe,
            sigma,
            bound,
            scratch,
        )
    }

    pub fn encrypt_zero_sk<R>(
        &mut self,
        module: &Module<FFT64>,
        sk_dft: &SecretKeyFourier<R, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        bound: f64,
        scratch: &mut Scratch,
    ) where
        ScalarZnxDft<R, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        encrypt_glwe_sk::<MUT, _, R>(
            module, self, None, sk_dft, source_xa, source_xe, sigma, bound, scratch,
        )
    }

    pub fn encrypt_pk<R0, R1>(
        &mut self,
        module: &Module<FFT64>,
        pt: &GLWEPlaintext<R0>,
        pk: &PublicKey<R1, FFT64>,
        source_xu: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        bound: f64,
        scratch: &mut Scratch,
    ) where
        VecZnx<R0>: VecZnxToRef,
        VecZnxDft<R1, FFT64>: VecZnxDftToRef<FFT64>,
    {
        encrypt_glwe_pk(
            module,
            self,
            Some(pt),
            pk,
            source_xu,
            source_xe,
            sigma,
            bound,
            scratch,
        )
    }

    pub fn encrypt_zero_pk<R>(
        &mut self,
        module: &Module<FFT64>,
        pk: &PublicKey<R, FFT64>,
        source_xu: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        bound: f64,
        scratch: &mut Scratch,
    ) where
        VecZnxDft<R, FFT64>: VecZnxDftToRef<FFT64>,
    {
        encrypt_glwe_pk::<MUT, _, R>(
            module, self, None, pk, source_xu, source_xe, sigma, bound, scratch,
        )
    }
}

impl<REF> GLWECiphertext<REF>
where
    VecZnx<REF>: VecZnxToRef,
{
    pub fn decrypt<MUT, REF1>(
        &self,
        module: &Module<FFT64>,
        pt: &mut GLWEPlaintext<MUT>,
        sk_dft: &SecretKeyFourier<REF1, FFT64>,
        scratch: &mut Scratch,
    ) where
        VecZnx<MUT>: VecZnxToMut + VecZnxToRef,
        ScalarZnxDft<REF1, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        decrypt_glwe(module, pt, self, sk_dft, scratch);
    }
}

pub(crate) fn encrypt_glwe_pk<C, P, S>(
    module: &Module<FFT64>,
    ct: &mut GLWECiphertext<C>,
    pt: Option<&GLWEPlaintext<P>>,
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
        assert_eq!(ct.basek(), pk.basek());
        assert_eq!(ct.n(), module.n());
        assert_eq!(pk.n(), module.n());
        if let Some(pt) = pt {
            assert_eq!(pt.basek(), pk.basek());
            assert_eq!(pt.n(), module.n());
        }
    }

    let log_base2k: usize = pk.basek();
    let size_pk: usize = pk.size();

    // Generates u according to the underlying secret distribution.
    let (mut u_dft, scratch_1) = scratch.tmp_scalar_znx_dft(module, 1);

    {
        let (mut u, _) = scratch_1.tmp_scalar_znx(module, 1);
        match pk.dist {
            SecretDistribution::NONE => panic!(
                "invalid public key: SecretDistribution::NONE, ensure it has been correctly intialized through Self::generate"
            ),
            SecretDistribution::TernaryFixed(hw) => u.fill_ternary_hw(0, hw, source_xu),
            SecretDistribution::TernaryProb(prob) => u.fill_ternary_prob(0, prob, source_xu),
            SecretDistribution::ZERO => {}
        }

        module.svp_prepare(&mut u_dft, 0, &u, 0);
    }

    let (mut tmp_big, scratch_2) = scratch_1.tmp_vec_znx_big(module, 1, size_pk); // TODO optimize size (e.g. when encrypting at low homomorphic capacity)
    let (mut tmp_dft, scratch_3) = scratch_2.tmp_vec_znx_dft(module, 1, size_pk); // TODO optimize size (e.g. when encrypting at low homomorphic capacity)

    // ct[0] = pk[0] * u + m + e0
    module.svp_apply(&mut tmp_dft, 0, &u_dft, 0, pk, 0);
    module.vec_znx_idft_tmp_a(&mut tmp_big, 0, &mut tmp_dft, 0);
    tmp_big.add_normal(log_base2k, 0, pk.k(), source_xe, sigma, bound);

    if let Some(pt) = pt {
        module.vec_znx_big_add_small_inplace(&mut tmp_big, 0, pt, 0);
    }

    module.vec_znx_big_normalize(log_base2k, ct, 0, &tmp_big, 0, scratch_3);

    // ct[1] = pk[1] * u + e1
    module.svp_apply(&mut tmp_dft, 0, &u_dft, 0, pk, 1);
    module.vec_znx_idft_tmp_a(&mut tmp_big, 0, &mut tmp_dft, 0);
    tmp_big.add_normal(log_base2k, 0, pk.k(), source_xe, sigma, bound);
    module.vec_znx_big_normalize(log_base2k, ct, 1, &tmp_big, 0, scratch_3);
}

pub struct GLWEPlaintext<C> {
    pub data: VecZnx<C>,
    pub log_base2k: usize,
    pub log_k: usize,
}

impl<T> Infos for GLWEPlaintext<T> {
    type Inner = VecZnx<T>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.log_base2k
    }

    fn k(&self) -> usize {
        self.log_k
    }
}

impl<C> VecZnxToMut for GLWEPlaintext<C>
where
    VecZnx<C>: VecZnxToMut,
{
    fn to_mut(&mut self) -> VecZnx<&mut [u8]> {
        self.data.to_mut()
    }
}

impl<C> VecZnxToRef for GLWEPlaintext<C>
where
    VecZnx<C>: VecZnxToRef,
{
    fn to_ref(&self) -> VecZnx<&[u8]> {
        self.data.to_ref()
    }
}

impl GLWEPlaintext<Vec<u8>> {
    pub fn new<B: Backend>(module: &Module<B>, log_base2k: usize, log_k: usize) -> Self {
        Self {
            data: module.new_vec_znx(1, derive_size(log_base2k, log_k)),
            log_base2k: log_base2k,
            log_k: log_k,
        }
    }
}

pub struct GLWECiphertextFourier<C, B: Backend> {
    pub data: VecZnxDft<C, B>,
    pub log_base2k: usize,
    pub log_k: usize,
}

impl<B: Backend> GLWECiphertextFourier<Vec<u8>, B> {
    pub fn new(module: &Module<B>, log_base2k: usize, log_k: usize) -> Self {
        Self {
            data: module.new_vec_znx_dft(2, derive_size(log_base2k, log_k)),
            log_base2k: log_base2k,
            log_k: log_k,
        }
    }
}

impl<T, B: Backend> Infos for GLWECiphertextFourier<T, B> {
    type Inner = VecZnxDft<T, B>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.log_base2k
    }

    fn k(&self) -> usize {
        self.log_k
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

impl<C> GLWECiphertextFourier<C, FFT64>
where
    GLWECiphertextFourier<C, FFT64>: VecZnxDftToRef<FFT64>,
{
    #[allow(dead_code)]
    pub(crate) fn idft_scratch_space(module: &Module<FFT64>, size: usize) -> usize {
        module.bytes_of_vec_znx(2, size) + (module.vec_znx_big_normalize_tmp_bytes() | module.vec_znx_idft_tmp_bytes())
    }

    pub(crate) fn idft<R>(&self, module: &Module<FFT64>, res: &mut GLWECiphertext<R>, scratch: &mut Scratch)
    where
        GLWECiphertext<R>: VecZnxToMut,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), 2);
            assert_eq!(res.rank(), 2);
            assert_eq!(self.basek(), res.basek())
        }

        let min_size: usize = self.size().min(res.size());

        let (mut res_big, scratch1) = scratch.tmp_vec_znx_big(module, 2, min_size);

        module.vec_znx_idft(&mut res_big, 0, self, 0, scratch1);
        module.vec_znx_idft(&mut res_big, 1, self, 1, scratch1);
        module.vec_znx_big_normalize(self.basek(), res, 0, &res_big, 0, scratch1);
        module.vec_znx_big_normalize(self.basek(), res, 1, &res_big, 1, scratch1);
    }
}

pub(crate) fn encrypt_zero_glwe_dft_sk<C, S>(
    module: &Module<FFT64>,
    ct: &mut GLWECiphertextFourier<C, FFT64>,
    sk: &SecretKeyFourier<S, FFT64>,
    source_xa: &mut Source,
    source_xe: &mut Source,
    sigma: f64,
    bound: f64,
    scratch: &mut Scratch,
) where
    VecZnxDft<C, FFT64>: VecZnxDftToMut<FFT64> + VecZnxDftToRef<FFT64>,
    ScalarZnxDft<S, FFT64>: ScalarZnxDftToRef<FFT64>,
{
    let log_base2k: usize = ct.basek();
    let log_k: usize = ct.k();
    let size: usize = ct.size();

    #[cfg(debug_assertions)]
    {
        match sk.dist {
            SecretDistribution::NONE => panic!("invalid sk.dist = SecretDistribution::NONE"),
            _ => {}
        }
        assert_eq!(ct.rank(), 2);
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

impl GLWECiphertextFourier<Vec<u8>, FFT64> {
    pub fn encrypt_zero_sk_scratch_space(module: &Module<FFT64>, size: usize) -> usize {
        (module.bytes_of_vec_znx(1, size) | module.bytes_of_vec_znx_dft(1, size))
            + module.bytes_of_vec_znx_big(1, size)
            + module.bytes_of_vec_znx(1, size)
            + module.vec_znx_big_normalize_tmp_bytes()
    }

    pub fn decrypt_scratch_space(module: &Module<FFT64>, size: usize) -> usize {
        (module.vec_znx_big_normalize_tmp_bytes()
            | module.bytes_of_vec_znx_dft(1, size)
            | (module.bytes_of_vec_znx_big(1, size) + module.vec_znx_idft_tmp_bytes()))
            + module.bytes_of_vec_znx_big(1, size)
    }
}

pub fn decrypt_rlwe_dft<P, C, S>(
    module: &Module<FFT64>,
    pt: &mut GLWEPlaintext<P>,
    ct: &GLWECiphertextFourier<C, FFT64>,
    sk: &SecretKeyFourier<S, FFT64>,
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
    module.vec_znx_big_normalize(ct.basek(), pt, 0, &mut c0_big, 0, scratch_1);

    pt.log_base2k = ct.basek();
    pt.log_k = pt.k().min(ct.k());
}

impl<C> GLWECiphertextFourier<C, FFT64>
where
    VecZnxDft<C, FFT64>: VecZnxDftToMut<FFT64> + VecZnxDftToRef<FFT64>,
{
    pub(crate) fn encrypt_zero_sk<S>(
        &mut self,
        module: &Module<FFT64>,
        sk_dft: &SecretKeyFourier<S, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        bound: f64,
        scratch: &mut Scratch,
    ) where
        ScalarZnxDft<S, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        encrypt_zero_glwe_dft_sk(
            module, self, sk_dft, source_xa, source_xe, sigma, bound, scratch,
        )
    }

    pub fn decrypt<P, S>(
        &self,
        module: &Module<FFT64>,
        pt: &mut GLWEPlaintext<P>,
        sk_dft: &SecretKeyFourier<S, FFT64>,
        scratch: &mut Scratch,
    ) where
        VecZnx<P>: VecZnxToMut + VecZnxToRef,
        ScalarZnxDft<S, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        decrypt_rlwe_dft(module, pt, self, sk_dft, scratch);
    }
}

impl KeySwitchScratchSpace for GLWECiphertextFourier<Vec<u8>, FFT64> {
    fn keyswitch_scratch_space(module: &Module<FFT64>, res_size: usize, lhs: usize, rhs: usize) -> usize {
        <GLWEKeySwitchKey<Vec<u8>, FFT64> as VecGLWEProductScratchSpace>::prod_with_glwe_scratch_space(module, res_size, lhs, rhs)
    }
}

impl<DataSelf, DataLhs, DataRhs> KeySwitch<DataLhs, DataRhs> for GLWECiphertextFourier<DataSelf, FFT64>
where
    VecZnxDft<DataSelf, FFT64>: VecZnxDftToMut<FFT64> + VecZnxDftToRef<FFT64>,
    VecZnxDft<DataLhs, FFT64>: VecZnxDftToRef<FFT64>,
    MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
{
    type Lhs = GLWECiphertextFourier<DataLhs, FFT64>;
    type Rhs = GLWEKeySwitchKey<DataRhs, FFT64>;

    fn keyswitch(&mut self, module: &Module<FFT64>, lhs: &Self::Lhs, rhs: &Self::Rhs, scratch: &mut Scratch) {
        rhs.prod_with_glwe_fourier(module, self, lhs, scratch);
    }
}

impl KeySwitchInplaceScratchSpace for GLWECiphertextFourier<Vec<u8>, FFT64> {
    fn keyswitch_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, rhs: usize) -> usize {
        <GLWEKeySwitchKey<Vec<u8>, FFT64> as VecGLWEProductScratchSpace>::prod_with_glwe_inplace_scratch_space(
            module, res_size, rhs,
        )
    }
}

impl<DataSelf, DataRhs> KeySwitchInplace<DataRhs> for GLWECiphertextFourier<DataSelf, FFT64>
where
    VecZnxDft<DataSelf, FFT64>: VecZnxDftToMut<FFT64> + VecZnxDftToRef<FFT64>,
    MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
{
    type Rhs = GLWEKeySwitchKey<DataRhs, FFT64>;

    fn keyswitch_inplace(&mut self, module: &Module<FFT64>, rhs: &Self::Rhs, scratch: &mut Scratch) {
        rhs.prod_with_glwe_fourier_inplace(module, self, scratch);
    }
}

impl ExternalProductScratchSpace for GLWECiphertextFourier<Vec<u8>, FFT64> {
    fn external_product_scratch_space(module: &Module<FFT64>, res_size: usize, lhs: usize, rhs: usize) -> usize {
        <GGSWCiphertext<Vec<u8>, FFT64> as VecGLWEProductScratchSpace>::prod_with_glwe_scratch_space(module, res_size, lhs, rhs)
    }
}

impl<DataSelf, DataLhs, DataRhs> ExternalProduct<DataLhs, DataRhs> for GLWECiphertextFourier<DataSelf, FFT64>
where
    VecZnxDft<DataSelf, FFT64>: VecZnxDftToMut<FFT64> + VecZnxDftToRef<FFT64>,
    VecZnxDft<DataLhs, FFT64>: VecZnxDftToRef<FFT64>,
    MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
{
    type Lhs = GLWECiphertextFourier<DataLhs, FFT64>;
    type Rhs = GGSWCiphertext<DataRhs, FFT64>;

    fn external_product(&mut self, module: &Module<FFT64>, lhs: &Self::Lhs, rhs: &Self::Rhs, scratch: &mut Scratch) {
        rhs.prod_with_glwe_fourier(module, self, lhs, scratch);
    }
}

impl ExternalProductInplaceScratchSpace for GLWECiphertextFourier<Vec<u8>, FFT64> {
    fn external_product_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, rhs: usize) -> usize {
        <GGSWCiphertext<Vec<u8>, FFT64> as VecGLWEProductScratchSpace>::prod_with_glwe_inplace_scratch_space(
            module, res_size, rhs,
        )
    }
}

impl<DataSelf, DataRhs> ExternalProductInplace<DataRhs> for GLWECiphertextFourier<DataSelf, FFT64>
where
    VecZnxDft<DataSelf, FFT64>: VecZnxDftToMut<FFT64> + VecZnxDftToRef<FFT64>,
    MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64> + ZnxInfos,
{
    type Rhs = GGSWCiphertext<DataRhs, FFT64>;

    fn external_product_inplace(&mut self, module: &Module<FFT64>, rhs: &Self::Rhs, scratch: &mut Scratch) {
        rhs.prod_with_glwe_fourier_inplace(module, self, scratch);
    }
}
