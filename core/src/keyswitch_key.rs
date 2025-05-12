use base2k::{
    Backend, FFT64, MatZnxDft, MatZnxDftAlloc, MatZnxDftOps, MatZnxDftScratch, MatZnxDftToMut, MatZnxDftToRef, Module, ScalarZnx,
    ScalarZnxDft, ScalarZnxDftToRef, ScalarZnxToRef, Scratch, VecZnx, VecZnxAlloc, VecZnxBig, VecZnxBigOps, VecZnxBigScratch,
    VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxDftToMut, VecZnxDftToRef, VecZnxOps, VecZnxToMut, VecZnxToRef, ZnxInfos,
    ZnxZero,
};
use sampling::source::Source;

use crate::{
    elem::{GetRow, Infos, SetRow},
    encryption::EncryptSkScratchSpace,
    external_product::{
        ExternalProduct, ExternalProductInplace, ExternalProductInplaceScratchSpace, ExternalProductScratchSpace,
    },
    ggsw::GGSWCiphertext,
    glwe::{GLWECiphertext, GLWECiphertextFourier, GLWEPlaintext},
    keys::SecretKeyFourier,
    keyswitch::{KeySwitch, KeySwitchInplace, KeySwitchInplaceScratchSpace, KeySwitchScratchSpace},
    utils::derive_size,
    vec_glwe_product::{VecGLWEProduct, VecGLWEProductScratchSpace},
};

pub struct GLWEKeySwitchKey<C, B: Backend> {
    pub data: MatZnxDft<C, B>,
    pub log_base2k: usize,
    pub log_k: usize,
}

impl<B: Backend> GLWEKeySwitchKey<Vec<u8>, B> {
    pub fn new(module: &Module<B>, log_base2k: usize, log_k: usize, rows: usize) -> Self {
        Self {
            data: module.new_mat_znx_dft(rows, 1, 2, derive_size(log_base2k, log_k)),
            log_base2k: log_base2k,
            log_k: log_k,
        }
    }
}

impl<T, B: Backend> Infos for GLWEKeySwitchKey<T, B> {
    type Inner = MatZnxDft<T, B>;

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

impl<C, B: Backend> MatZnxDftToMut<B> for GLWEKeySwitchKey<C, B>
where
    MatZnxDft<C, B>: MatZnxDftToMut<B>,
{
    fn to_mut(&mut self) -> MatZnxDft<&mut [u8], B> {
        self.data.to_mut()
    }
}

impl<C, B: Backend> MatZnxDftToRef<B> for GLWEKeySwitchKey<C, B>
where
    MatZnxDft<C, B>: MatZnxDftToRef<B>,
{
    fn to_ref(&self) -> MatZnxDft<&[u8], B> {
        self.data.to_ref()
    }
}

impl GLWEKeySwitchKey<Vec<u8>, FFT64> {
    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, size: usize) -> usize {
        GLWECiphertext::encrypt_sk_scratch_space(module, size)
            + module.bytes_of_vec_znx(2, size)
            + module.bytes_of_vec_znx(1, size)
            + module.bytes_of_vec_znx_dft(2, size)
    }
}

pub fn encrypt_glwe_key_switch_key_sk<C, P, S>(
    module: &Module<FFT64>,
    ct: &mut GLWEKeySwitchKey<C, FFT64>,
    pt: &ScalarZnx<P>,
    sk_dft: &SecretKeyFourier<S, FFT64>,
    source_xa: &mut Source,
    source_xe: &mut Source,
    sigma: f64,
    bound: f64,
    scratch: &mut Scratch,
) where
    MatZnxDft<C, FFT64>: MatZnxDftToMut<FFT64>,
    ScalarZnx<P>: ScalarZnxToRef,
    ScalarZnxDft<S, FFT64>: ScalarZnxDftToRef<FFT64>,
{
    let rows: usize = ct.rows();
    let size: usize = ct.size();
    let log_base2k: usize = ct.basek();

    let (tmp_znx_pt, scrach_1) = scratch.tmp_vec_znx(module, 1, size);
    let (tmp_znx_ct, scrach_2) = scrach_1.tmp_vec_znx(module, 2, size);
    let (mut vec_znx_dft_ct, scratch_3) = scrach_2.tmp_vec_znx_dft(module, 2, size);

    let mut vec_znx_pt: GLWEPlaintext<&mut [u8]> = GLWEPlaintext {
        data: tmp_znx_pt,
        log_base2k: log_base2k,
        log_k: ct.k(),
    };

    let mut vec_znx_ct: GLWECiphertext<&mut [u8]> = GLWECiphertext {
        data: tmp_znx_ct,
        log_base2k: log_base2k,
        log_k: ct.k(),
    };

    (0..rows).for_each(|row_i| {
        // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
        module.vec_znx_add_scalar_inplace(&mut vec_znx_pt, 0, row_i, pt, 0);
        module.vec_znx_normalize_inplace(log_base2k, &mut vec_znx_pt, 0, scratch_3);

        // rlwe encrypt of vec_znx_pt into vec_znx_ct
        vec_znx_ct.encrypt_sk(
            module,
            &vec_znx_pt,
            sk_dft,
            source_xa,
            source_xe,
            sigma,
            bound,
            scratch_3,
        );

        vec_znx_pt.data.zero(); // zeroes for next iteration

        // Switch vec_znx_ct into DFT domain
        module.vec_znx_dft(&mut vec_znx_dft_ct, 0, &vec_znx_ct, 0);
        module.vec_znx_dft(&mut vec_znx_dft_ct, 1, &vec_znx_ct, 1);

        // Stores vec_znx_dft_ct into thw i-th row of the MatZnxDft
        module.vmp_prepare_row(ct, row_i, 0, &vec_znx_dft_ct);
    });
}

impl<C> GLWEKeySwitchKey<C, FFT64> {
    pub fn encrypt_sk<P, S>(
        &mut self,
        module: &Module<FFT64>,
        pt: &ScalarZnx<P>,
        sk_dft: &SecretKeyFourier<S, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        bound: f64,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<C, FFT64>: MatZnxDftToMut<FFT64>,
        ScalarZnx<P>: ScalarZnxToRef,
        ScalarZnxDft<S, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        encrypt_glwe_key_switch_key_sk(
            module, self, pt, sk_dft, source_xa, source_xe, sigma, bound, scratch,
        )
    }
}

impl<C> GetRow<FFT64> for GLWEKeySwitchKey<C, FFT64>
where
    MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64>,
{
    fn get_row<R>(&self, module: &Module<FFT64>, row_i: usize, col_j: usize, res: &mut GLWECiphertextFourier<R, FFT64>)
    where
        VecZnxDft<R, FFT64>: VecZnxDftToMut<FFT64>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(col_j, 0);
        }
        module.vmp_extract_row(res, self, row_i, col_j);
    }
}

impl<C> SetRow<FFT64> for GLWEKeySwitchKey<C, FFT64>
where
    MatZnxDft<C, FFT64>: MatZnxDftToMut<FFT64>,
{
    fn set_row<R>(&mut self, module: &Module<FFT64>, row_i: usize, col_j: usize, a: &GLWECiphertextFourier<R, FFT64>)
    where
        VecZnxDft<R, FFT64>: VecZnxDftToRef<FFT64>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(col_j, 0);
        }
        module.vmp_prepare_row(self, row_i, col_j, a);
    }
}

impl KeySwitchScratchSpace for GLWEKeySwitchKey<Vec<u8>, FFT64> {
    fn keyswitch_scratch_space(module: &Module<FFT64>, res_size: usize, lhs: usize, rhs: usize) -> usize {
        <GLWEKeySwitchKey<Vec<u8>, FFT64> as VecGLWEProductScratchSpace>::prod_with_vec_glwe_scratch_space(
            module, res_size, lhs, rhs,
        )
    }
}

impl<DataSelf, DataLhs, DataRhs> KeySwitch<DataLhs, DataRhs> for GLWEKeySwitchKey<DataSelf, FFT64>
where
    MatZnxDft<DataSelf, FFT64>: MatZnxDftToMut<FFT64> + MatZnxDftToRef<FFT64>,
    MatZnxDft<DataLhs, FFT64>: MatZnxDftToRef<FFT64>,
    MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
{
    type Lhs = GLWEKeySwitchKey<DataLhs, FFT64>;
    type Rhs = GLWEKeySwitchKey<DataRhs, FFT64>;

    fn keyswitch(&mut self, module: &Module<FFT64>, lhs: &Self::Lhs, rhs: &Self::Rhs, scratch: &mut Scratch) {
        rhs.prod_with_vec_glwe(module, self, lhs, scratch);
    }
}

impl KeySwitchInplaceScratchSpace for GLWEKeySwitchKey<Vec<u8>, FFT64> {
    fn keyswitch_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, rhs: usize) -> usize {
        <GLWEKeySwitchKey<Vec<u8>, FFT64> as VecGLWEProductScratchSpace>::prod_with_vec_glwe_inplace_scratch_space(
            module, res_size, rhs,
        )
    }
}

impl<DataSelf, DataRhs> KeySwitchInplace<DataRhs> for GLWEKeySwitchKey<DataSelf, FFT64>
where
    MatZnxDft<DataSelf, FFT64>: MatZnxDftToMut<FFT64> + MatZnxDftToRef<FFT64>,
    MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
{
    type Rhs = GLWEKeySwitchKey<DataRhs, FFT64>;

    fn keyswitch_inplace(&mut self, module: &Module<FFT64>, rhs: &Self::Rhs, scratch: &mut Scratch) {
        rhs.prod_with_vec_glwe(module, self, rhs, scratch);
    }
}

impl ExternalProductScratchSpace for GLWEKeySwitchKey<Vec<u8>, FFT64> {
    fn external_product_scratch_space(module: &Module<FFT64>, res_size: usize, lhs: usize, rhs: usize) -> usize {
        <GGSWCiphertext<Vec<u8>, FFT64> as VecGLWEProductScratchSpace>::prod_with_vec_glwe_scratch_space(
            module, res_size, lhs, rhs,
        )
    }
}

impl<DataSelf, DataLhs, DataRhs> ExternalProduct<DataLhs, DataRhs> for GLWEKeySwitchKey<DataSelf, FFT64>
where
    MatZnxDft<DataSelf, FFT64>: MatZnxDftToMut<FFT64> + MatZnxDftToRef<FFT64>,
    MatZnxDft<DataLhs, FFT64>: MatZnxDftToRef<FFT64>,
    MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
{
    type Lhs = GLWEKeySwitchKey<DataLhs, FFT64>;
    type Rhs = GGSWCiphertext<DataRhs, FFT64>;

    fn external_product(&mut self, module: &Module<FFT64>, lhs: &Self::Lhs, rhs: &Self::Rhs, scratch: &mut Scratch) {
        rhs.prod_with_vec_glwe(module, self, lhs, scratch);
    }
}

impl ExternalProductInplaceScratchSpace for GLWEKeySwitchKey<Vec<u8>, FFT64> {
    fn external_product_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, rhs: usize) -> usize {
        <GGSWCiphertext<Vec<u8>, FFT64> as VecGLWEProductScratchSpace>::prod_with_glwe_inplace_scratch_space(
            module, res_size, rhs,
        )
    }
}

impl<DataSelf, DataRhs> ExternalProductInplace<DataRhs> for GLWEKeySwitchKey<DataSelf, FFT64>
where
    MatZnxDft<DataSelf, FFT64>: MatZnxDftToMut<FFT64> + MatZnxDftToRef<FFT64>,
    MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
{
    type Rhs = GGSWCiphertext<DataRhs, FFT64>;

    fn external_product_inplace(&mut self, module: &Module<FFT64>, rhs: &Self::Rhs, scratch: &mut Scratch) {
        rhs.prod_with_vec_glwe_inplace(module, self, scratch);
    }
}

impl VecGLWEProductScratchSpace for GLWEKeySwitchKey<Vec<u8>, FFT64> {
    fn prod_with_glwe_scratch_space(module: &Module<FFT64>, res_size: usize, a_size: usize, grlwe_size: usize) -> usize {
        module.bytes_of_vec_znx_dft(2, grlwe_size)
            + (module.vec_znx_big_normalize_tmp_bytes()
                | (module.vmp_apply_tmp_bytes(res_size, a_size, a_size, 1, 2, grlwe_size)
                    + module.bytes_of_vec_znx_dft(1, a_size)))
    }
}

impl<C> VecGLWEProduct for GLWEKeySwitchKey<C, FFT64>
where
    MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64> + ZnxInfos,
{
    fn prod_with_glwe<R, A>(
        &self,
        module: &Module<FFT64>,
        res: &mut GLWECiphertext<R>,
        a: &GLWECiphertext<A>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64>,
        VecZnx<R>: VecZnxToMut,
        VecZnx<A>: VecZnxToRef,
    {
        let log_base2k: usize = self.basek();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.basek(), log_base2k);
            assert_eq!(a.basek(), log_base2k);
            assert_eq!(self.n(), module.n());
            assert_eq!(res.n(), module.n());
            assert_eq!(a.n(), module.n());
        }

        let (mut res_dft, scratch1) = scratch.tmp_vec_znx_dft(module, 2, self.size()); // Todo optimise

        {
            let (mut a1_dft, scratch2) = scratch1.tmp_vec_znx_dft(module, 1, a.size());
            module.vec_znx_dft(&mut a1_dft, 0, a, 1);
            module.vmp_apply(&mut res_dft, &a1_dft, self, scratch2);
        }

        let mut res_big: VecZnxBig<&mut [u8], FFT64> = module.vec_znx_idft_consume(res_dft);

        module.vec_znx_big_add_small_inplace(&mut res_big, 0, a, 0);

        module.vec_znx_big_normalize(log_base2k, res, 0, &res_big, 0, scratch1);
        module.vec_znx_big_normalize(log_base2k, res, 1, &res_big, 1, scratch1);
    }
}
