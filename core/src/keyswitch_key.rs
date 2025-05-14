use base2k::{
    Backend, FFT64, MatZnxDft, MatZnxDftOps, MatZnxDftToMut, MatZnxDftToRef, Module, ScalarZnx, ScalarZnxDft, ScalarZnxDftToRef,
    ScalarZnxToRef, Scratch, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef,
};
use sampling::source::Source;

use crate::{
    elem::{GetRow, Infos, SetRow},
    gglwe_ciphertext::GGLWECiphertext,
    ggsw_ciphertext::GGSWCiphertext,
    glwe_ciphertext_fourier::GLWECiphertextFourier,
    keys::{SecretKey, SecretKeyFourier},
    vec_glwe_product::{VecGLWEProduct, VecGLWEProductScratchSpace},
};

pub struct GLWESwitchingKey<Data, B: Backend>(pub(crate) GGLWECiphertext<Data, B>);

impl GLWESwitchingKey<Vec<u8>, FFT64> {
    pub fn new(module: &Module<FFT64>, base2k: usize, k: usize, rows: usize, rank_in: usize, rank_out: usize) -> Self {
        GLWESwitchingKey(GGLWECiphertext::new(
            module, base2k, k, rows, rank_in, rank_out,
        ))
    }
}

impl<T, B: Backend> Infos for GLWESwitchingKey<T, B> {
    type Inner = MatZnxDft<T, B>;

    fn inner(&self) -> &Self::Inner {
        &self.0.inner()
    }

    fn basek(&self) -> usize {
        self.0.basek()
    }

    fn k(&self) -> usize {
        self.0.k()
    }
}

impl<DataSelf, B: Backend> MatZnxDftToMut<B> for GLWESwitchingKey<DataSelf, B>
where
    MatZnxDft<DataSelf, B>: MatZnxDftToMut<B>,
{
    fn to_mut(&mut self) -> MatZnxDft<&mut [u8], B> {
        self.0.data.to_mut()
    }
}

impl<DataSelf, B: Backend> MatZnxDftToRef<B> for GLWESwitchingKey<DataSelf, B>
where
    MatZnxDft<DataSelf, B>: MatZnxDftToRef<B>,
{
    fn to_ref(&self) -> MatZnxDft<&[u8], B> {
        self.0.data.to_ref()
    }
}

impl<C> GetRow<FFT64> for GLWESwitchingKey<C, FFT64>
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

impl<C> SetRow<FFT64> for GLWESwitchingKey<C, FFT64>
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

impl GLWESwitchingKey<Vec<u8>, FFT64> {
    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, rank: usize, size: usize) -> usize {
        GGLWECiphertext::encrypt_sk_scratch_space(module, rank, size)
    }

    pub fn encrypt_pk_scratch_space(module: &Module<FFT64>, rank: usize, pk_size: usize) -> usize {
        GGLWECiphertext::encrypt_pk_scratch_space(module, rank, pk_size)
    }
}

impl<DataSelf> GLWESwitchingKey<DataSelf, FFT64>
where
    MatZnxDft<DataSelf, FFT64>: MatZnxDftToMut<FFT64> + MatZnxDftToRef<FFT64>,
{
    pub fn encrypt_sk<DataSkIn, DataSkOut>(
        &mut self,
        module: &Module<FFT64>,
        sk_in: &SecretKey<DataSkIn>,
        sk_out_dft: &SecretKeyFourier<DataSkOut, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        ScalarZnx<DataSkIn>: ScalarZnxToRef,
        ScalarZnxDft<DataSkOut, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        self.0.encrypt_sk(
            module,
            &sk_in.data,
            sk_out_dft,
            source_xa,
            source_xe,
            sigma,
            scratch,
        );
    }
}

impl GLWESwitchingKey<Vec<u8>, FFT64> {
    pub fn keyswitch_scratch_space(
        module: &Module<FFT64>,
        res_size: usize,
        lhs: usize,
        rhs: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize {
        <GGLWECiphertext<Vec<u8>, FFT64> as VecGLWEProductScratchSpace>::prod_with_vec_glwe_scratch_space(
            module, res_size, lhs, rhs, rank_in, rank_out,
        )
    }

    pub fn keyswitch_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, rhs: usize, rank: usize) -> usize {
        <GGLWECiphertext<Vec<u8>, FFT64> as VecGLWEProductScratchSpace>::prod_with_vec_glwe_inplace_scratch_space(
            module, res_size, rhs, rank,
        )
    }

    pub fn external_product_scratch_space(module: &Module<FFT64>, res_size: usize, lhs: usize, rhs: usize, rank: usize) -> usize {
        <GGSWCiphertext<Vec<u8>, FFT64> as VecGLWEProductScratchSpace>::prod_with_vec_glwe_scratch_space(
            module, res_size, lhs, rhs, rank, rank,
        )
    }

    pub fn external_product_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, rhs: usize, rank: usize) -> usize {
        <GGSWCiphertext<Vec<u8>, FFT64> as VecGLWEProductScratchSpace>::prod_with_glwe_inplace_scratch_space(
            module, res_size, rhs, rank,
        )
    }
}

impl<DataSelf> GLWESwitchingKey<DataSelf, FFT64>
where
    MatZnxDft<DataSelf, FFT64>: MatZnxDftToMut<FFT64> + MatZnxDftToRef<FFT64>,
{
    pub fn keyswitch<DataLhs, DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWESwitchingKey<DataLhs, FFT64>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut base2k::Scratch,
    ) where
        MatZnxDft<DataLhs, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        rhs.0
            .prod_with_vec_glwe(module, &mut self.0, &lhs.0, scratch);
    }

    pub fn keyswitch_inplace<DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut base2k::Scratch,
    ) where
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        rhs.0
            .prod_with_vec_glwe_inplace(module, &mut self.0, scratch);
    }

    pub fn external_product<DataLhs, DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWESwitchingKey<DataLhs, FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataLhs, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        rhs.prod_with_vec_glwe(module, &mut self.0, &lhs.0, scratch);
    }

    pub fn external_product_inplace<DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        rhs.prod_with_vec_glwe_inplace(module, &mut self.0, scratch);
    }
}
