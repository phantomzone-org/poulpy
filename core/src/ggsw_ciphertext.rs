use base2k::{
    Backend, FFT64, MatZnxDft, MatZnxDftAlloc, MatZnxDftOps, MatZnxDftToMut, MatZnxDftToRef, Module, ScalarZnx, ScalarZnxDft,
    ScalarZnxDftToRef, ScalarZnxToRef, Scratch, VecZnxAlloc, VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxDftToMut,
    VecZnxDftToRef, VecZnxOps, ZnxInfos, ZnxZero,
};
use sampling::source::Source;

use crate::{
    automorphism::AutomorphismKey,
    elem::{GetRow, Infos, SetRow},
    glwe_ciphertext::GLWECiphertext,
    glwe_ciphertext_fourier::GLWECiphertextFourier,
    glwe_plaintext::GLWEPlaintext,
    keys::SecretKeyFourier,
    utils::derive_size,
};

pub struct GGSWCiphertext<C, B: Backend> {
    pub data: MatZnxDft<C, B>,
    pub basek: usize,
    pub k: usize,
}

impl<B: Backend> GGSWCiphertext<Vec<u8>, B> {
    pub fn new(module: &Module<B>, basek: usize, k: usize, rows: usize, rank: usize) -> Self {
        Self {
            data: module.new_mat_znx_dft(rows, rank + 1, rank + 1, derive_size(basek, k)),
            basek: basek,
            k: k,
        }
    }
}

impl<T, B: Backend> Infos for GGSWCiphertext<T, B> {
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

    pub fn automorphism_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        in_size: usize,
        auto_key_size: usize,
        rank: usize,
    ) -> usize {
        let size: usize = in_size.min(out_size);
        let tmp_dft: usize = module.bytes_of_vec_znx_dft(rank + 1, size);
        let tmp_idft: usize = module.bytes_of_vec_znx(rank + 1, size);
        let vmp: usize = GLWECiphertext::keyswitch_from_fourier_scratch_space(module, size, rank, size, rank, auto_key_size);
        tmp_dft + tmp_idft + vmp
    }

    pub fn external_product_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        in_size: usize,
        ggsw_size: usize,
        rank: usize,
    ) -> usize {
        let tmp_in: usize = module.bytes_of_vec_znx_dft(rank + 1, in_size);
        let tmp_out: usize = module.bytes_of_vec_znx_dft(rank + 1, out_size);
        let ggsw: usize = GLWECiphertextFourier::external_product_scratch_space(module, out_size, in_size, ggsw_size, rank);
        tmp_in + tmp_out + ggsw
    }

    pub fn external_product_inplace_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        ggsw_size: usize,
        rank: usize,
    ) -> usize {
        let tmp: usize = module.bytes_of_vec_znx_dft(rank + 1, out_size);
        let ggsw: usize = GLWECiphertextFourier::external_product_inplace_scratch_space(module, out_size, ggsw_size, rank);
        tmp + ggsw
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
        let basek: usize = self.basek();
        let k: usize = self.k();
        let cols: usize = self.rank() + 1;

        let (tmp_znx_pt, scratch_1) = scratch.tmp_vec_znx(module, 1, size);
        let (tmp_znx_ct, scrach_2) = scratch_1.tmp_vec_znx(module, cols, size);

        let mut vec_znx_pt: GLWEPlaintext<&mut [u8]> = GLWEPlaintext {
            data: tmp_znx_pt,
            basek: basek,
            k: k,
        };

        let mut vec_znx_ct: GLWECiphertext<&mut [u8]> = GLWECiphertext {
            data: tmp_znx_ct,
            basek: basek,
            k,
        };

        (0..self.rows()).for_each(|row_j| {
            vec_znx_pt.data.zero();

            // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
            module.vec_znx_add_scalar_inplace(&mut vec_znx_pt, 0, row_j, pt, 0);
            module.vec_znx_normalize_inplace(basek, &mut vec_znx_pt, 0, scrach_2);

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
        });
    }

    pub fn automorphism<DataLhs, DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GGSWCiphertext<DataLhs, FFT64>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataLhs, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.rank(),
                lhs.rank(),
                "ggsw_out rank: {} != ggsw_in rank: {}",
                self.rank(),
                lhs.rank()
            );
            assert_eq!(
                self.rank(),
                rhs.rank(),
                "ggsw_in rank: {} != auto_key rank: {}",
                self.rank(),
                rhs.rank()
            );
        }

        let size: usize = self.size().min(lhs.size());
        let cols: usize = self.rank() + 1;

        let (tmp_dft_data, scratch1) = scratch.tmp_vec_znx_dft(module, cols, size);

        let mut tmp_dft: GLWECiphertextFourier<&mut [u8], FFT64> = GLWECiphertextFourier::<&mut [u8], FFT64> {
            data: tmp_dft_data,
            basek: lhs.basek(),
            k: lhs.k(),
        };

        let (tmp_idft_data, scratch2) = scratch1.tmp_vec_znx(module, cols, size);

        let mut tmp_idft: GLWECiphertext<&mut [u8]> = GLWECiphertext::<&mut [u8]> {
            data: tmp_idft_data,
            basek: self.basek(),
            k: self.k(),
        };

        (0..cols).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                lhs.get_row(module, row_j, col_i, &mut tmp_dft);
                tmp_idft.keyswitch_from_fourier(module, &tmp_dft, &rhs.key, scratch2);
                (0..cols).for_each(|i| {
                    module.vec_znx_automorphism_inplace(rhs.p(), &mut tmp_idft, i);
                });
                self.set_row(module, row_j, col_i, &tmp_dft);
            });
        });

        tmp_dft.data.zero();

        (self.rows().min(lhs.rows())..self.rows()).for_each(|row_i| {
            (0..self.rank() + 1).for_each(|col_j| {
                self.set_row(module, row_i, col_j, &tmp_dft);
            });
        });
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
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.rank(),
                lhs.rank(),
                "ggsw_out rank: {} != ggsw_in rank: {}",
                self.rank(),
                lhs.rank()
            );
            assert_eq!(
                self.rank(),
                rhs.rank(),
                "ggsw_in rank: {} != ggsw_apply rank: {}",
                self.rank(),
                rhs.rank()
            );
        }

        let (tmp_in_data, scratch1) = scratch.tmp_vec_znx_dft(module, lhs.rank() + 1, lhs.size());

        let mut tmp_in: GLWECiphertextFourier<&mut [u8], FFT64> = GLWECiphertextFourier::<&mut [u8], FFT64> {
            data: tmp_in_data,
            basek: lhs.basek(),
            k: lhs.k(),
        };

        let (tmp_out_data, scratch2) = scratch1.tmp_vec_znx_dft(module, self.rank() + 1, self.size());

        let mut tmp_out: GLWECiphertextFourier<&mut [u8], FFT64> = GLWECiphertextFourier::<&mut [u8], FFT64> {
            data: tmp_out_data,
            basek: self.basek(),
            k: self.k(),
        };

        (0..self.rank() + 1).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                lhs.get_row(module, row_j, col_i, &mut tmp_in);
                tmp_out.external_product(module, &tmp_in, rhs, scratch2);
                self.set_row(module, row_j, col_i, &tmp_out);
            });
        });

        tmp_out.data.zero();

        (self.rows().min(lhs.rows())..self.rows()).for_each(|row_i| {
            (0..self.rank() + 1).for_each(|col_j| {
                self.set_row(module, row_i, col_j, &tmp_out);
            });
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
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.rank(),
                rhs.rank(),
                "ggsw_out rank: {} != ggsw_apply: {}",
                self.rank(),
                rhs.rank()
            );
        }

        let (tmp_data, scratch1) = scratch.tmp_vec_znx_dft(module, self.rank() + 1, self.size());

        let mut tmp: GLWECiphertextFourier<&mut [u8], FFT64> = GLWECiphertextFourier::<&mut [u8], FFT64> {
            data: tmp_data,
            basek: self.basek(),
            k: self.k(),
        };

        (0..self.rank() + 1).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                self.get_row(module, row_j, col_i, &mut tmp);
                tmp.external_product_inplace(module, rhs, scratch1);
                self.set_row(module, row_j, col_i, &tmp);
            });
        });
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
