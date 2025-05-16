use base2k::{
    Backend, FFT64, MatZnxDft, MatZnxDftOps, MatZnxDftToMut, MatZnxDftToRef, Module, ScalarZnx, ScalarZnxDftOps, ScalarZnxOps,
    ScalarZnxToRef, Scratch, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef,
};
use sampling::source::Source;

use crate::{
    elem::{GetRow, Infos, SetRow},
    gglwe_ciphertext::GGLWECiphertext,
    ggsw_ciphertext::GGSWCiphertext,
    glwe_ciphertext_fourier::GLWECiphertextFourier,
    keys::{SecretKey, SecretKeyFourier},
    keyswitch_key::GLWESwitchingKey,
};

pub struct AutomorphismKey<Data, B: Backend> {
    pub(crate) key: GLWESwitchingKey<Data, B>,
    pub(crate) p: i64,
}

impl AutomorphismKey<Vec<u8>, FFT64> {
    pub fn new(module: &Module<FFT64>, basek: usize, p: i64, k: usize, rows: usize, rank: usize) -> Self {
        AutomorphismKey {
            key: GLWESwitchingKey::new(module, basek, k, rows, rank, rank),
            p: p,
        }
    }
}

impl<T, B: Backend> Infos for AutomorphismKey<T, B> {
    type Inner = MatZnxDft<T, B>;

    fn inner(&self) -> &Self::Inner {
        &self.key.inner()
    }

    fn basek(&self) -> usize {
        self.key.basek()
    }

    fn k(&self) -> usize {
        self.key.k()
    }
}

impl<T, B: Backend> AutomorphismKey<T, B> {
    pub fn p(&self) -> i64 {
        self.p
    }

    pub fn rank(&self) -> usize {
        self.key.rank()
    }

    pub fn rank_in(&self) -> usize {
        self.key.rank_in()
    }

    pub fn rank_out(&self) -> usize {
        self.key.rank_out()
    }
}

impl<DataSelf, B: Backend> MatZnxDftToMut<B> for AutomorphismKey<DataSelf, B>
where
    MatZnxDft<DataSelf, B>: MatZnxDftToMut<B>,
{
    fn to_mut(&mut self) -> MatZnxDft<&mut [u8], B> {
        self.key.to_mut()
    }
}

impl<DataSelf, B: Backend> MatZnxDftToRef<B> for AutomorphismKey<DataSelf, B>
where
    MatZnxDft<DataSelf, B>: MatZnxDftToRef<B>,
{
    fn to_ref(&self) -> MatZnxDft<&[u8], B> {
        self.key.to_ref()
    }
}

impl<C> GetRow<FFT64> for AutomorphismKey<C, FFT64>
where
    MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64>,
{
    fn get_row<R>(&self, module: &Module<FFT64>, row_i: usize, col_j: usize, res: &mut GLWECiphertextFourier<R, FFT64>)
    where
        VecZnxDft<R, FFT64>: VecZnxDftToMut<FFT64>,
    {
        module.vmp_extract_row(res, self, row_i, col_j);
    }
}

impl<C> SetRow<FFT64> for AutomorphismKey<C, FFT64>
where
    MatZnxDft<C, FFT64>: MatZnxDftToMut<FFT64>,
{
    fn set_row<R>(&mut self, module: &Module<FFT64>, row_i: usize, col_j: usize, a: &GLWECiphertextFourier<R, FFT64>)
    where
        VecZnxDft<R, FFT64>: VecZnxDftToRef<FFT64>,
    {
        module.vmp_prepare_row(self, row_i, col_j, a);
    }
}

impl AutomorphismKey<Vec<u8>, FFT64> {
    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, rank: usize, size: usize) -> usize {
        GGLWECiphertext::encrypt_sk_scratch_space(module, rank, size)
    }

    pub fn encrypt_pk_scratch_space(module: &Module<FFT64>, rank: usize, pk_size: usize) -> usize {
        GGLWECiphertext::encrypt_pk_scratch_space(module, rank, pk_size)
    }

    pub fn keyswitch_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        in_size: usize,
        ksk_size: usize,
        rank: usize,
    ) -> usize {
        GLWESwitchingKey::keyswitch_scratch_space(module, out_size, rank, in_size, rank, ksk_size)
    }

    pub fn keyswitch_inplace_scratch_space(module: &Module<FFT64>, out_size: usize, out_rank: usize, ksk_size: usize) -> usize {
        GLWESwitchingKey::keyswitch_inplace_scratch_space(module, out_size, out_rank, ksk_size)
    }

    pub fn external_product_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        in_size: usize,
        ggsw_size: usize,
        rank: usize,
    ) -> usize {
        GLWESwitchingKey::external_product_scratch_space(module, out_size, in_size, ggsw_size, rank)
    }

    pub fn external_product_inplace_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        ggsw_size: usize,
        rank: usize,
    ) -> usize {
        GLWESwitchingKey::external_product_inplace_scratch_space(module, out_size, ggsw_size, rank)
    }
}

impl<DataSelf> AutomorphismKey<DataSelf, FFT64>
where
    MatZnxDft<DataSelf, FFT64>: MatZnxDftToMut<FFT64> + MatZnxDftToRef<FFT64>,
{
    pub fn encrypt_sk<DataSk>(
        &mut self,
        module: &Module<FFT64>,
        p: i64,
        sk: &SecretKey<DataSk>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        ScalarZnx<DataSk>: ScalarZnxToRef,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), module.n());
            assert_eq!(sk.n(), module.n());
            assert_eq!(self.rank_out(), self.rank_in());
            assert_eq!(sk.rank(), self.rank());
        }

        let (sk_out_dft_data, scratch_1) = scratch.tmp_scalar_znx_dft(module, sk.rank());

        let mut sk_out_dft: SecretKeyFourier<&mut [u8], FFT64> = SecretKeyFourier {
            data: sk_out_dft_data,
            dist: sk.dist,
        };

        {
            (0..self.rank()).for_each(|i| {
                let (mut sk_inv_auto, _) = scratch_1.tmp_scalar_znx(module, 1);
                module.scalar_znx_automorphism(module.galois_element_inv(p), &mut sk_inv_auto, 0, sk, i);
                module.svp_prepare(&mut sk_out_dft, i, &sk_inv_auto, 0);
            });
        }

        self.key.encrypt_sk(
            module,
            &sk,
            &sk_out_dft,
            source_xa,
            source_xe,
            sigma,
            scratch_1,
        );

        self.p = p;
    }
}

impl<DataSelf> AutomorphismKey<DataSelf, FFT64>
where
    MatZnxDft<DataSelf, FFT64>: MatZnxDftToMut<FFT64> + MatZnxDftToRef<FFT64>,
{
    pub fn keyswitch<DataLhs, DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &AutomorphismKey<DataLhs, FFT64>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut base2k::Scratch,
    ) where
        MatZnxDft<DataLhs, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        self.key.keyswitch(module, &lhs.key, rhs, scratch);
    }

    pub fn keyswitch_inplace<DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut base2k::Scratch,
    ) where
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        self.key.keyswitch_inplace(module, &rhs.key, scratch);
    }

    pub fn external_product<DataLhs, DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &AutomorphismKey<DataLhs, FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataLhs, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        self.key.external_product(module, &lhs.key, rhs, scratch);
    }

    pub fn external_product_inplace<DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        self.key.external_product_inplace(module, rhs, scratch);
    }
}
