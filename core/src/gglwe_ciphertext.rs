use base2k::{
    Backend, FFT64, MatZnxDft, MatZnxDftAlloc, MatZnxDftOps, MatZnxDftScratch, MatZnxDftToMut, MatZnxDftToRef, Module, ScalarZnx,
    ScalarZnxDft, ScalarZnxDftToRef, ScalarZnxToRef, Scratch, VecZnx, VecZnxAlloc, VecZnxBig, VecZnxBigOps, VecZnxBigScratch,
    VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxDftToMut, VecZnxDftToRef, VecZnxOps, VecZnxToMut, VecZnxToRef, ZnxInfos,
    ZnxZero,
};
use sampling::source::Source;

use crate::{
    elem::{GetRow, Infos, SetRow},
    glwe_ciphertext::GLWECiphertext,
    glwe_ciphertext_fourier::GLWECiphertextFourier,
    glwe_plaintext::GLWEPlaintext,
    keys::SecretKeyFourier,
    utils::derive_size,
    vec_glwe_product::{VecGLWEProduct, VecGLWEProductScratchSpace},
};

pub struct GGLWECiphertext<C, B: Backend> {
    pub(crate) data: MatZnxDft<C, B>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
}

impl<B: Backend> GGLWECiphertext<Vec<u8>, B> {
    pub fn new(module: &Module<B>, base2k: usize, k: usize, rows: usize, rank_in: usize, rank_out: usize) -> Self {
        Self {
            data: module.new_mat_znx_dft(rows, rank_in, rank_out + 1, derive_size(base2k, k)),
            basek: base2k,
            k,
        }
    }
}

impl<T, B: Backend> Infos for GGLWECiphertext<T, B> {
    type Inner = MatZnxDft<T, B>;

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

impl<T, B: Backend> GGLWECiphertext<T, B> {
    pub fn rank(&self) -> usize {
        self.data.cols_out() - 1
    }
}

impl<C, B: Backend> MatZnxDftToMut<B> for GGLWECiphertext<C, B>
where
    MatZnxDft<C, B>: MatZnxDftToMut<B>,
{
    fn to_mut(&mut self) -> MatZnxDft<&mut [u8], B> {
        self.data.to_mut()
    }
}

impl<C, B: Backend> MatZnxDftToRef<B> for GGLWECiphertext<C, B>
where
    MatZnxDft<C, B>: MatZnxDftToRef<B>,
{
    fn to_ref(&self) -> MatZnxDft<&[u8], B> {
        self.data.to_ref()
    }
}

impl GGLWECiphertext<Vec<u8>, FFT64> {
    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, rank: usize, size: usize) -> usize {
        GLWECiphertext::encrypt_sk_scratch_space(module, rank, size)
            + module.bytes_of_vec_znx(rank + 1, size)
            + module.bytes_of_vec_znx(1, size)
            + module.bytes_of_vec_znx_dft(rank + 1, size)
    }

    pub fn encrypt_pk_scratch_space(_module: &Module<FFT64>, _rank: usize, _pk_size: usize) -> usize {
        unimplemented!()
    }
}

impl<DataSelf> GGLWECiphertext<DataSelf, FFT64>
where
    MatZnxDft<DataSelf, FFT64>: MatZnxDftToMut<FFT64> + ZnxInfos,
{
    pub fn encrypt_sk<DataPt, DataSk>(
        &mut self,
        module: &Module<FFT64>,
        pt: &ScalarZnx<DataPt>,
        sk_dft: &SecretKeyFourier<DataSk, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        bound: f64,
        scratch: &mut Scratch,
    ) where
        ScalarZnx<DataPt>: ScalarZnxToRef,
        ScalarZnxDft<DataSk, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), sk_dft.rank());
            assert_eq!(self.n(), module.n());
            assert_eq!(sk_dft.n(), module.n());
            assert_eq!(pt.n(), module.n());
        }

        let rows: usize = self.rows();
        let size: usize = self.size();
        let basek: usize = self.basek();
        let k: usize = self.k();

        let cols: usize = self.rank() + 1;

        let (tmp_znx_pt, scrach_1) = scratch.tmp_vec_znx(module, 1, size);
        let (tmp_znx_ct, scrach_2) = scrach_1.tmp_vec_znx(module, cols, size);
        let (tmp_znx_dft_ct, scratch_3) = scrach_2.tmp_vec_znx_dft(module, cols, size);

        let mut vec_znx_pt: GLWEPlaintext<&mut [u8]> = GLWEPlaintext {
            data: tmp_znx_pt,
            basek,
            k,
        };

        let mut vec_znx_ct: GLWECiphertext<&mut [u8]> = GLWECiphertext {
            data: tmp_znx_ct,
            basek,
            k,
        };

        let mut vec_znx_ct_dft: GLWECiphertextFourier<&mut [u8], FFT64> = GLWECiphertextFourier {
            data: tmp_znx_dft_ct,
            basek,
            k,
        };

        (0..rows).for_each(|row_i| {
            // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
            module.vec_znx_add_scalar_inplace(&mut vec_znx_pt, 0, row_i, pt, 0);
            module.vec_znx_normalize_inplace(basek, &mut vec_znx_pt, 0, scratch_3);

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
            vec_znx_ct.dft(module, &mut vec_znx_ct_dft);

            // Stores vec_znx_dft_ct into thw i-th row of the MatZnxDft
            module.vmp_prepare_row(self, row_i, 0, &vec_znx_ct_dft);
        });
    }
}

impl<C> GetRow<FFT64> for GGLWECiphertext<C, FFT64>
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

impl<C> SetRow<FFT64> for GGLWECiphertext<C, FFT64>
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

impl VecGLWEProductScratchSpace for GGLWECiphertext<Vec<u8>, FFT64> {
    fn prod_with_glwe_scratch_space(module: &Module<FFT64>, res_size: usize, a_size: usize, grlwe_size: usize) -> usize {
        module.bytes_of_vec_znx_dft(2, grlwe_size)
            + (module.vec_znx_big_normalize_tmp_bytes()
                | (module.vmp_apply_tmp_bytes(res_size, a_size, a_size, 1, 2, grlwe_size)
                    + module.bytes_of_vec_znx_dft(1, a_size)))
    }
}

impl<C> VecGLWEProduct for GGLWECiphertext<C, FFT64>
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
