use base2k::{
    Backend, FFT64, MatZnxDft, MatZnxDftAlloc, MatZnxDftOps, MatZnxDftToMut, MatZnxDftToRef, Module, ScalarZnx, ScalarZnxDft,
    ScalarZnxDftToRef, ScalarZnxToRef, Scratch, VecZnxAlloc, VecZnxDft, VecZnxDftAlloc, VecZnxDftToMut, VecZnxDftToRef,
    VecZnxOps, ZnxInfos, ZnxZero,
};
use sampling::source::Source;

use crate::{
    elem::{GetRow, Infos, SetRow},
    glwe_ciphertext::GLWECiphertext,
    glwe_ciphertext_fourier::GLWECiphertextFourier,
    glwe_plaintext::GLWEPlaintext,
    keys::SecretKeyFourier,
    utils::derive_size,
};

pub struct GGLWECiphertext<C, B: Backend> {
    pub(crate) data: MatZnxDft<C, B>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
}

impl<B: Backend> GGLWECiphertext<Vec<u8>, B> {
    pub fn new(module: &Module<B>, basek: usize, k: usize, rows: usize, rank_in: usize, rank_out: usize) -> Self {
        Self {
            data: module.new_mat_znx_dft(rows, rank_in, rank_out + 1, derive_size(basek, k)),
            basek: basek,
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

    pub fn rank_in(&self) -> usize {
        self.data.cols_in()
    }

    pub fn rank_out(&self) -> usize {
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
        GLWECiphertext::encrypt_sk_scratch_space(module, size)
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
        scratch: &mut Scratch,
    ) where
        ScalarZnx<DataPt>: ScalarZnxToRef,
        ScalarZnxDft<DataSk, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank_in(), pt.cols());
            assert_eq!(self.rank_out(), sk_dft.rank());
            assert_eq!(self.n(), module.n());
            assert_eq!(sk_dft.n(), module.n());
            assert_eq!(pt.n(), module.n());
        }

        let rows: usize = self.rows();
        let size: usize = self.size();
        let basek: usize = self.basek();
        let k: usize = self.k();

        let cols_in: usize = self.rank_in();
        let cols_out: usize = self.rank_out() + 1;

        let (tmp_znx_pt, scrach_1) = scratch.tmp_vec_znx(module, 1, size);
        let (tmp_znx_ct, scrach_2) = scrach_1.tmp_vec_znx(module, cols_out, size);
        let (tmp_znx_dft_ct, scratch_3) = scrach_2.tmp_vec_znx_dft(module, cols_out, size);

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

        // For each input column (i.e. rank) produces a GGLWE ciphertext of rank_out+1 columns
        //
        // Example for ksk rank 2 to rank 3:
        //
        // (-(a0*s0 + a1*s1 + a2*s2) + s0', a0, a1, a2)
        // (-(b0*s0 + b1*s1 + b2*s2) + s0', b0, b1, b2)
        //
        // Example ksk rank 2 to rank 1
        //
        // (-(a*s) + s0, a)
        // (-(b*s) + s1, b)
        (0..cols_in).for_each(|col_i| {
            (0..rows).for_each(|row_i| {
                // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
                vec_znx_pt.data.zero(); // zeroes for next iteration
                module.vec_znx_add_scalar_inplace(&mut vec_znx_pt, 0, row_i, pt, col_i); // Selects the i-th
                module.vec_znx_normalize_inplace(basek, &mut vec_znx_pt, 0, scratch_3);

                // rlwe encrypt of vec_znx_pt into vec_znx_ct
                vec_znx_ct.encrypt_sk(
                    module,
                    &vec_znx_pt,
                    sk_dft,
                    source_xa,
                    source_xe,
                    sigma,
                    scratch_3,
                );

                // Switch vec_znx_ct into DFT domain
                vec_znx_ct.dft(module, &mut vec_znx_ct_dft);

                // Stores vec_znx_dft_ct into thw i-th row of the MatZnxDft
                module.vmp_prepare_row(self, row_i, col_i, &vec_znx_ct_dft);
            });
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
        module.vmp_prepare_row(self, row_i, col_j, a);
    }
}
