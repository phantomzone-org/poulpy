use base2k::{
    AddNormal, Backend, FFT64, FillUniform, MatZnxDft, MatZnxDftToRef, Module, ScalarZnxAlloc, ScalarZnxDft, ScalarZnxDftAlloc,
    ScalarZnxDftOps, ScalarZnxDftToRef, Scratch, VecZnx, VecZnxAlloc, VecZnxBigAlloc, VecZnxBigOps, VecZnxBigScratch, VecZnxDft,
    VecZnxDftAlloc, VecZnxDftOps, VecZnxDftToMut, VecZnxDftToRef, VecZnxOps, VecZnxToMut, VecZnxToRef, ZnxInfos,
};
use sampling::source::Source;

use crate::{
    elem::Infos,
    grlwe::GRLWECt,
    keys::{PublicKey, SecretDistribution, SecretKeyDft},
    utils::derive_size,
};

pub struct RLWECt<C> {
    pub data: VecZnx<C>,
    pub log_base2k: usize,
    pub log_k: usize,
}

impl RLWECt<Vec<u8>> {
    pub fn new<B: Backend>(module: &Module<B>, log_base2k: usize, log_k: usize) -> Self {
        Self {
            data: module.new_vec_znx(2, derive_size(log_base2k, log_k)),
            log_base2k: log_base2k,
            log_k: log_k,
        }
    }
}

impl<T> Infos for RLWECt<T> {
    type Inner = VecZnx<T>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn log_base2k(&self) -> usize {
        self.log_base2k
    }

    fn log_k(&self) -> usize {
        self.log_k
    }
}

impl<C> VecZnxToMut for RLWECt<C>
where
    VecZnx<C>: VecZnxToMut,
{
    fn to_mut(&mut self) -> VecZnx<&mut [u8]> {
        self.data.to_mut()
    }
}

impl<C> VecZnxToRef for RLWECt<C>
where
    VecZnx<C>: VecZnxToRef,
{
    fn to_ref(&self) -> VecZnx<&[u8]> {
        self.data.to_ref()
    }
}

impl<C> RLWECt<C>
where
    VecZnx<C>: VecZnxToRef,
{
    #[allow(dead_code)]
    pub(crate) fn dft<R>(&self, module: &Module<FFT64>, res: &mut RLWECtDft<R, FFT64>)
    where
        VecZnxDft<R, FFT64>: VecZnxDftToMut<FFT64> + ZnxInfos,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.cols(), 2);
            assert_eq!(res.cols(), 2);
            assert_eq!(self.log_base2k(), res.log_base2k())
        }

        module.vec_znx_dft(res, 0, self, 0);
        module.vec_znx_dft(res, 1, self, 1);
    }
}

pub struct RLWEPt<C> {
    pub data: VecZnx<C>,
    pub log_base2k: usize,
    pub log_k: usize,
}

impl<T> Infos for RLWEPt<T> {
    type Inner = VecZnx<T>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn log_base2k(&self) -> usize {
        self.log_base2k
    }

    fn log_k(&self) -> usize {
        self.log_k
    }
}

impl<C> VecZnxToMut for RLWEPt<C>
where
    VecZnx<C>: VecZnxToMut,
{
    fn to_mut(&mut self) -> VecZnx<&mut [u8]> {
        self.data.to_mut()
    }
}

impl<C> VecZnxToRef for RLWEPt<C>
where
    VecZnx<C>: VecZnxToRef,
{
    fn to_ref(&self) -> VecZnx<&[u8]> {
        self.data.to_ref()
    }
}

impl RLWEPt<Vec<u8>> {
    pub fn new<B: Backend>(module: &Module<B>, log_base2k: usize, log_k: usize) -> Self {
        Self {
            data: module.new_vec_znx(1, derive_size(log_base2k, log_k)),
            log_base2k: log_base2k,
            log_k: log_k,
        }
    }
}

pub struct RLWECtDft<C, B: Backend> {
    pub data: VecZnxDft<C, B>,
    pub log_base2k: usize,
    pub log_k: usize,
}

impl<B: Backend> RLWECtDft<Vec<u8>, B> {
    pub fn new(module: &Module<B>, log_base2k: usize, log_k: usize) -> Self {
        Self {
            data: module.new_vec_znx_dft(2, derive_size(log_base2k, log_k)),
            log_base2k: log_base2k,
            log_k: log_k,
        }
    }
}

impl<T, B: Backend> Infos for RLWECtDft<T, B> {
    type Inner = VecZnxDft<T, B>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn log_base2k(&self) -> usize {
        self.log_base2k
    }

    fn log_k(&self) -> usize {
        self.log_k
    }
}

impl<C, B: Backend> VecZnxDftToMut<B> for RLWECtDft<C, B>
where
    VecZnxDft<C, B>: VecZnxDftToMut<B>,
{
    fn to_mut(&mut self) -> VecZnxDft<&mut [u8], B> {
        self.data.to_mut()
    }
}

impl<C, B: Backend> VecZnxDftToRef<B> for RLWECtDft<C, B>
where
    VecZnxDft<C, B>: VecZnxDftToRef<B>,
{
    fn to_ref(&self) -> VecZnxDft<&[u8], B> {
        self.data.to_ref()
    }
}

impl<C> RLWECtDft<C, FFT64>
where
    VecZnxDft<C, FFT64>: VecZnxDftToRef<FFT64>,
{
    #[allow(dead_code)]
    pub(crate) fn idft_scratch_space(module: &Module<FFT64>, size: usize) -> usize {
        module.bytes_of_vec_znx(2, size) + (module.vec_znx_big_normalize_tmp_bytes() | module.vec_znx_idft_tmp_bytes())
    }

    pub(crate) fn idft<R>(&self, module: &Module<FFT64>, res: &mut RLWECt<R>, scratch: &mut Scratch)
    where
        VecZnx<R>: VecZnxToMut,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.cols(), 2);
            assert_eq!(res.cols(), 2);
            assert_eq!(self.log_base2k(), res.log_base2k())
        }

        let min_size: usize = self.size().min(res.size());

        let (mut res_big, scratch1) = scratch.tmp_vec_znx_big(module, 2, min_size);

        module.vec_znx_idft(&mut res_big, 0, &self.data, 0, scratch1);
        module.vec_znx_idft(&mut res_big, 1, &self.data, 1, scratch1);
        module.vec_znx_big_normalize(self.log_base2k(), res, 0, &res_big, 0, scratch1);
        module.vec_znx_big_normalize(self.log_base2k(), res, 1, &res_big, 1, scratch1);
    }
}

impl RLWECt<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, size: usize) -> usize {
        (module.vec_znx_big_normalize_tmp_bytes() | module.bytes_of_vec_znx_dft(1, size)) + module.bytes_of_vec_znx_big(1, size)
    }

    pub fn encrypt_pk_scratch_space<B: Backend>(module: &Module<B>, pk_size: usize) -> usize {
        ((module.bytes_of_vec_znx_dft(1, pk_size) + module.bytes_of_vec_znx_big(1, pk_size)) | module.bytes_of_scalar_znx(1))
            + module.bytes_of_scalar_znx_dft(1)
            + module.vec_znx_big_normalize_tmp_bytes()
    }

    pub fn decrypt_scratch_space<B: Backend>(module: &Module<B>, size: usize) -> usize {
        (module.vec_znx_big_normalize_tmp_bytes() | module.bytes_of_vec_znx_dft(1, size)) + module.bytes_of_vec_znx_big(1, size)
    }
}

pub fn encrypt_rlwe_sk<C, P, S>(
    module: &Module<FFT64>,
    ct: &mut RLWECt<C>,
    pt: Option<(&RLWEPt<P>, usize)>,
    sk_dft: &SecretKeyDft<S, FFT64>,
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

pub fn decrypt_rlwe<P, C, S>(
    module: &Module<FFT64>,
    pt: &mut RLWEPt<P>,
    ct: &RLWECt<C>,
    sk_dft: &SecretKeyDft<S, FFT64>,
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
    module.vec_znx_big_normalize(ct.log_base2k(), pt, 0, &mut c0_big, 0, scratch_1);

    pt.log_base2k = ct.log_base2k();
    pt.log_k = pt.log_k().min(ct.log_k());
}

impl<C> RLWECt<C> {
    pub fn encrypt_sk<P, S>(
        &mut self,
        module: &Module<FFT64>,
        pt: Option<&RLWEPt<P>>,
        sk_dft: &SecretKeyDft<S, FFT64>,
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
        if let Some(pt) = pt {
            encrypt_rlwe_sk(
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
        } else {
            encrypt_rlwe_sk::<C, P, S>(
                module, self, None, sk_dft, source_xa, source_xe, sigma, bound, scratch,
            )
        }
    }

    pub fn decrypt<P, S>(
        &self,
        module: &Module<FFT64>,
        pt: &mut RLWEPt<P>,
        sk_dft: &SecretKeyDft<S, FFT64>,
        scratch: &mut Scratch,
    ) where
        VecZnx<P>: VecZnxToMut + VecZnxToRef,
        VecZnx<C>: VecZnxToRef,
        ScalarZnxDft<S, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        decrypt_rlwe(module, pt, self, sk_dft, scratch);
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

impl RLWECtDft<Vec<u8>, FFT64> {
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
    pt.log_k = pt.log_k().min(ct.log_k());
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

    pub fn mul_grlwe_assign<A>(&mut self, module: &Module<FFT64>, a: &GRLWECt<A, FFT64>, scratch: &mut Scratch)
    where
        VecZnxDft<C, FFT64>: VecZnxDftToMut<FFT64> + VecZnxDftToRef<FFT64>,
        MatZnxDft<A, FFT64>: MatZnxDftToRef<FFT64>,
    {
        a.mul_rlwe_dft_inplace(module, self, scratch);
    }
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
