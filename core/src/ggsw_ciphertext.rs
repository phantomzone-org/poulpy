use base2k::{
    Backend, FFT64, MatZnxDft, MatZnxDftAlloc, MatZnxDftOps, MatZnxDftScratch, MatZnxDftToMut, MatZnxDftToRef, Module, ScalarZnx,
    ScalarZnxDft, ScalarZnxDftToRef, ScalarZnxToRef, Scratch, VecZnx, VecZnxAlloc, VecZnxBig, VecZnxBigOps, VecZnxBigScratch,
    VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxDftToMut, VecZnxDftToRef, VecZnxOps, VecZnxToMut, VecZnxToRef, ZnxInfos,
    ZnxZero,
};
use sampling::source::Source;

use crate::{
    elem::{GetRow, Infos, SetRow},
    gglwe_ciphertext::GGLWECiphertext,
    glwe_ciphertext::GLWECiphertext,
    glwe_ciphertext_fourier::GLWECiphertextFourier,
    glwe_plaintext::GLWEPlaintext,
    keys::SecretKeyFourier,
    keyswitch_key::GLWESwitchingKey,
    utils::derive_size,
    vec_glwe_product::{VecGLWEProduct, VecGLWEProductScratchSpace},
};

pub struct GGSWCiphertext<C, B: Backend> {
    pub data: MatZnxDft<C, B>,
    pub log_base2k: usize,
    pub log_k: usize,
}

impl<B: Backend> GGSWCiphertext<Vec<u8>, B> {
    pub fn new(module: &Module<B>, log_base2k: usize, log_k: usize, rows: usize, rank: usize) -> Self {
        Self {
            data: module.new_mat_znx_dft(rows, rank + 1, rank + 1, derive_size(log_base2k, log_k)),
            log_base2k: log_base2k,
            log_k: log_k,
        }
    }
}

impl<T, B: Backend> Infos for GGSWCiphertext<T, B> {
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

impl<T, B: Backend> GGSWCiphertext<T, B> {
    pub fn rank(&self) -> usize {
        self.data.cols_out() - 1
    }
}

impl<C, B: Backend> MatZnxDftToMut<B> for GGSWCiphertext<C, B>
where
    MatZnxDft<C, B>: MatZnxDftToMut<B>,
{
    fn to_mut(&mut self) -> MatZnxDft<&mut [u8], B> {
        self.data.to_mut()
    }
}

impl<C, B: Backend> MatZnxDftToRef<B> for GGSWCiphertext<C, B>
where
    MatZnxDft<C, B>: MatZnxDftToRef<B>,
{
    fn to_ref(&self) -> MatZnxDft<&[u8], B> {
        self.data.to_ref()
    }
}

impl GGSWCiphertext<Vec<u8>, FFT64> {
    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, rank: usize, size: usize) -> usize {
        GLWECiphertext::encrypt_sk_scratch_space(module, size)
            + module.bytes_of_vec_znx(rank + 1, size)
            + module.bytes_of_vec_znx(1, size)
            + module.bytes_of_vec_znx_dft(rank + 1, size)
    }

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

impl<DataSelf> GGSWCiphertext<DataSelf, FFT64>
where
    MatZnxDft<DataSelf, FFT64>: MatZnxDftToMut<FFT64> + MatZnxDftToRef<FFT64>,
{
    pub fn encrypt_sk<DataPt, DataSk>(
        &mut self,
        module: &Module<FFT64>,
        pt: &ScalarZnx<DataPt>,
        sk_dft: &SecretKeyFourier<DataSk, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        ScalarZnx<DataPt>: ScalarZnxToRef,
        ScalarZnxDft<DataSk, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), sk_dft.rank());
            assert_eq!(self.n(), module.n());
            assert_eq!(pt.n(), module.n());
            assert_eq!(sk_dft.n(), module.n());
        }

        let size: usize = self.size();
        let log_base2k: usize = self.basek();
        let k: usize = self.k();
        let cols: usize = self.rank() + 1;

        let (tmp_znx_pt, scratch_1) = scratch.tmp_vec_znx(module, 1, size);
        let (tmp_znx_ct, scrach_2) = scratch_1.tmp_vec_znx(module, cols, size);

        let mut vec_znx_pt: GLWEPlaintext<&mut [u8]> = GLWEPlaintext {
            data: tmp_znx_pt,
            basek: log_base2k,
            k: k,
        };

        let mut vec_znx_ct: GLWECiphertext<&mut [u8]> = GLWECiphertext {
            data: tmp_znx_ct,
            basek: log_base2k,
            k,
        };

        (0..self.rows()).for_each(|row_j| {
            // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
            module.vec_znx_add_scalar_inplace(&mut vec_znx_pt, 0, row_j, pt, 0);
            module.vec_znx_normalize_inplace(log_base2k, &mut vec_znx_pt, 0, scrach_2);

            (0..cols).for_each(|col_i| {
                // rlwe encrypt of vec_znx_pt into vec_znx_ct

                vec_znx_ct.encrypt_sk_private(
                    module,
                    Some((&vec_znx_pt, col_i)),
                    sk_dft,
                    source_xa,
                    source_xe,
                    sigma,
                    scrach_2,
                );

                // Switch vec_znx_ct into DFT domain
                {
                    let (mut vec_znx_dft_ct, _) = scrach_2.tmp_vec_znx_dft(module, cols, size);

                    (0..cols).for_each(|i| {
                        module.vec_znx_dft(&mut vec_znx_dft_ct, i, &vec_znx_ct, i);
                    });

                    module.vmp_prepare_row(self, row_j, col_i, &vec_znx_dft_ct);
                }
            });

            vec_znx_pt.data.zero(); // zeroes for next iteration
        });
    }

    pub fn keyswitch<DataLhs, DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GGSWCiphertext<DataLhs, FFT64>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataLhs, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        rhs.0.prod_with_vec_glwe(module, self, lhs, scratch);
    }

    pub fn keyswitch_inplace<DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        rhs.0.prod_with_vec_glwe_inplace(module, self, scratch);
    }

    pub fn external_product<DataLhs, DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GGSWCiphertext<DataLhs, FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataLhs, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        rhs.prod_with_vec_glwe(module, self, lhs, scratch);
    }

    pub fn external_product_inplace<DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        rhs.prod_with_vec_glwe_inplace(module, self, scratch);
    }
}

impl<DataSelf> GetRow<FFT64> for GGSWCiphertext<DataSelf, FFT64>
where
    MatZnxDft<DataSelf, FFT64>: MatZnxDftToRef<FFT64>,
{
    fn get_row<DataRes>(
        &self,
        module: &Module<FFT64>,
        row_i: usize,
        col_j: usize,
        res: &mut GLWECiphertextFourier<DataRes, FFT64>,
    ) where
        VecZnxDft<DataRes, FFT64>: VecZnxDftToMut<FFT64>,
    {
        module.vmp_extract_row(res, self, row_i, col_j);
    }
}

impl<DataSelf> SetRow<FFT64> for GGSWCiphertext<DataSelf, FFT64>
where
    MatZnxDft<DataSelf, FFT64>: MatZnxDftToMut<FFT64>,
{
    fn set_row<DataRes>(&mut self, module: &Module<FFT64>, row_i: usize, col_j: usize, a: &GLWECiphertextFourier<DataRes, FFT64>)
    where
        VecZnxDft<DataRes, FFT64>: VecZnxDftToRef<FFT64>,
    {
        module.vmp_prepare_row(self, row_i, col_j, a);
    }
}

impl VecGLWEProductScratchSpace for GGSWCiphertext<Vec<u8>, FFT64> {
    fn prod_with_glwe_scratch_space(
        module: &Module<FFT64>,
        res_size: usize,
        a_size: usize,
        rgsw_size: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize {
        module.bytes_of_vec_znx_dft(rank_out + 1, rgsw_size)
            + ((module.bytes_of_vec_znx_dft(rank_in + 1, a_size)
                + module.vmp_apply_tmp_bytes(
                    res_size,
                    a_size,
                    a_size,
                    rank_in + 1,
                    rank_out + 1,
                    rgsw_size,
                ))
                | module.vec_znx_big_normalize_tmp_bytes())
    }
}

impl<C> VecGLWEProduct for GGSWCiphertext<C, FFT64>
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
        VecZnx<R>: VecZnxToMut,
        VecZnx<A>: VecZnxToRef,
    {
        let log_base2k: usize = self.basek();

        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), a.rank());
            assert_eq!(self.rank(), res.rank());
            assert_eq!(res.basek(), log_base2k);
            assert_eq!(a.basek(), log_base2k);
            assert_eq!(self.n(), module.n());
            assert_eq!(res.n(), module.n());
            assert_eq!(a.n(), module.n());
        }

        let cols: usize = self.rank() + 1;

        let (mut res_dft, scratch1) = scratch.tmp_vec_znx_dft(module, cols, self.size()); // Todo optimise

        {
            let (mut a_dft, scratch2) = scratch1.tmp_vec_znx_dft(module, cols, a.size());
            (0..cols).for_each(|col_i| {
                module.vec_znx_dft(&mut a_dft, col_i, a, col_i);
            });
            module.vmp_apply(&mut res_dft, &a_dft, self, scratch2);
        }

        let res_big: VecZnxBig<&mut [u8], FFT64> = module.vec_znx_idft_consume(res_dft);

        (0..cols).for_each(|i| {
            module.vec_znx_big_normalize(log_base2k, res, i, &res_big, i, scratch1);
        });
    }
}
