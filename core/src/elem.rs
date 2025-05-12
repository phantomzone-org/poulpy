use base2k::{
    Backend, FFT64, MatZnxDft, MatZnxDftToRef, Module, Scratch, VecZnx, VecZnxAlloc, VecZnxDft, VecZnxDftAlloc, VecZnxDftOps,
    VecZnxDftToMut, VecZnxDftToRef, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxZero,
};

use crate::{
    grlwe::GRLWECt,
    rgsw::RGSWCt,
    rlwe::{RLWECt, RLWECtDft},
    utils::derive_size,
};

pub trait Infos {
    type Inner: ZnxInfos;

    fn inner(&self) -> &Self::Inner;

    /// Returns the ring degree of the polynomials.
    fn n(&self) -> usize {
        self.inner().n()
    }

    /// Returns the base two logarithm of the ring dimension of the polynomials.
    fn log_n(&self) -> usize {
        self.inner().log_n()
    }

    /// Returns the number of rows.
    fn rows(&self) -> usize {
        self.inner().rows()
    }

    /// Returns the number of polynomials in each row.
    fn cols(&self) -> usize {
        self.inner().cols()
    }

    /// Returns the number of size per polynomial.
    fn size(&self) -> usize {
        let size: usize = self.inner().size();
        debug_assert_eq!(size, derive_size(self.log_base2k(), self.log_k()));
        size
    }

    /// Returns the total number of small polynomials.
    fn poly_count(&self) -> usize {
        self.rows() * self.cols() * self.size()
    }

    /// Returns the base 2 logarithm of the ciphertext base.
    fn log_base2k(&self) -> usize;

    /// Returns the bit precision of the ciphertext.
    fn log_k(&self) -> usize;
}

pub trait GetRow<B: Backend> {
    fn get_row<R>(&self, module: &Module<B>, row_i: usize, col_j: usize, res: &mut RLWECtDft<R, B>)
    where
        VecZnxDft<R, B>: VecZnxDftToMut<B>;
}

pub trait SetRow<B: Backend> {
    fn set_row<A>(&mut self, module: &Module<B>, row_i: usize, col_j: usize, a: &RLWECtDft<A, B>)
    where
        VecZnxDft<A, B>: VecZnxDftToRef<B>;
}

pub trait ProdInplaceScratchSpace {
    fn prod_by_grlwe_inplace_scratch_space(module: &Module<FFT64>, lhs: usize, rhs: usize) -> usize;
    fn prod_by_rgsw_inplace_scratch_space(module: &Module<FFT64>, lhs: usize, rhs: usize) -> usize;
}

pub trait ProdInplace<MUT, REF>
where
    MatZnxDft<REF, FFT64>: MatZnxDftToRef<FFT64>,
{
    fn prod_by_grlwe_inplace(&mut self, module: &Module<FFT64>, rhs: &GRLWECt<REF, FFT64>, scratch: &mut Scratch);
    fn prod_by_rgsw_inplace(&mut self, module: &Module<FFT64>, rhs: &RGSWCt<REF, FFT64>, scratch: &mut Scratch);
}

pub trait ProdScratchSpace {
    fn prod_by_grlwe_scratch_space(module: &Module<FFT64>, res_size: usize, lhs: usize, rhs: usize) -> usize;
    fn prod_by_rgsw_scratch_space(module: &Module<FFT64>, res_size: usize, lhs: usize, rhs: usize) -> usize;
}

pub trait Product<MUT, REF>
where
    MatZnxDft<REF, FFT64>: MatZnxDftToRef<FFT64>,
{
    type Lhs;

    fn prod_by_grlwe(&mut self, module: &Module<FFT64>, lhs: &Self::Lhs, rhs: &GRLWECt<REF, FFT64>, scratch: &mut Scratch);
    fn prod_by_rgsw(&mut self, module: &Module<FFT64>, lhs: &Self::Lhs, rhs: &RGSWCt<REF, FFT64>, scratch: &mut Scratch);
}

pub(crate) trait MatRLWEProductScratchSpace {
    fn prod_with_rlwe_scratch_space(module: &Module<FFT64>, res_size: usize, a_size: usize, grlwe_size: usize) -> usize;

    fn prod_with_rlwe_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, mat_size: usize) -> usize {
        Self::prod_with_rlwe_scratch_space(module, res_size, res_size, mat_size)
    }

    fn prod_with_rlwe_dft_scratch_space(module: &Module<FFT64>, res_size: usize, a_size: usize, mat_size: usize) -> usize {
        (Self::prod_with_rlwe_scratch_space(module, res_size, a_size, mat_size) | module.vec_znx_idft_tmp_bytes())
            + module.bytes_of_vec_znx(2, a_size)
            + module.bytes_of_vec_znx(2, res_size)
    }

    fn prod_with_rlwe_dft_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, mat_size: usize) -> usize {
        (Self::prod_with_rlwe_inplace_scratch_space(module, res_size, mat_size) | module.vec_znx_idft_tmp_bytes())
            + module.bytes_of_vec_znx(2, res_size)
    }

    fn prod_with_mat_rlwe_scratch_space(module: &Module<FFT64>, res_size: usize, a_size: usize, mat_size: usize) -> usize {
        Self::prod_with_rlwe_dft_scratch_space(module, res_size, a_size, mat_size)
            + module.bytes_of_vec_znx_dft(2, a_size)
            + module.bytes_of_vec_znx_dft(2, res_size)
    }

    fn prod_with_mat_rlwe_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, mat_size: usize) -> usize {
        Self::prod_with_rlwe_dft_inplace_scratch_space(module, res_size, mat_size) + module.bytes_of_vec_znx_dft(2, res_size)
    }
}

pub(crate) trait MatRLWEProduct: Infos {
    fn prod_with_rlwe<MUT, REF>(&self, module: &Module<FFT64>, res: &mut RLWECt<MUT>, a: &RLWECt<REF>, scratch: &mut Scratch)
    where
        VecZnx<MUT>: VecZnxToMut,
        VecZnx<REF>: VecZnxToRef;

    fn prod_with_rlwe_inplace<MUT>(&self, module: &Module<FFT64>, res: &mut RLWECt<MUT>, scratch: &mut Scratch)
    where
        VecZnx<MUT>: VecZnxToMut + VecZnxToRef,
    {
        unsafe {
            let res_ptr: *mut RLWECt<MUT> = res as *mut RLWECt<MUT>; // This is ok because [Self::mul_rlwe] only updates res at the end.
            self.prod_with_rlwe(&module, &mut *res_ptr, &*res_ptr, scratch);
        }
    }

    fn prod_with_rlwe_dft<MUT, REF>(
        &self,
        module: &Module<FFT64>,
        res: &mut RLWECtDft<MUT, FFT64>,
        a: &RLWECtDft<REF, FFT64>,
        scratch: &mut Scratch,
    ) where
        VecZnxDft<MUT, FFT64>: VecZnxDftToMut<FFT64> + VecZnxDftToRef<FFT64> + ZnxInfos,
        VecZnxDft<REF, FFT64>: VecZnxDftToRef<FFT64> + ZnxInfos,
    {
        let log_base2k: usize = self.log_base2k();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.log_base2k(), log_base2k);
            assert_eq!(self.n(), module.n());
            assert_eq!(res.n(), module.n());
        }

        let (a_data, scratch_1) = scratch.tmp_vec_znx(module, 2, a.size());

        let mut a_idft: RLWECt<&mut [u8]> = RLWECt::<&mut [u8]> {
            data: a_data,
            log_base2k: a.log_base2k(),
            log_k: a.log_k(),
        };

        a.idft(module, &mut a_idft, scratch_1);

        let (res_data, scratch_2) = scratch_1.tmp_vec_znx(module, 2, res.size());

        let mut res_idft: RLWECt<&mut [u8]> = RLWECt::<&mut [u8]> {
            data: res_data,
            log_base2k: res.log_base2k(),
            log_k: res.log_k(),
        };

        self.prod_with_rlwe(module, &mut res_idft, &a_idft, scratch_2);

        module.vec_znx_dft(res, 0, &res_idft, 0);
        module.vec_znx_dft(res, 1, &res_idft, 1);
    }

    fn prod_with_rlwe_dft_inplace<MUT>(&self, module: &Module<FFT64>, res: &mut RLWECtDft<MUT, FFT64>, scratch: &mut Scratch)
    where
        VecZnxDft<MUT, FFT64>: VecZnxDftToRef<FFT64> + VecZnxDftToMut<FFT64>,
    {
        let log_base2k: usize = self.log_base2k();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.log_base2k(), log_base2k);
            assert_eq!(self.n(), module.n());
            assert_eq!(res.n(), module.n());
        }

        let (res_data, scratch_1) = scratch.tmp_vec_znx(module, 2, res.size());

        let mut res_idft: RLWECt<&mut [u8]> = RLWECt::<&mut [u8]> {
            data: res_data,
            log_base2k: res.log_base2k(),
            log_k: res.log_k(),
        };

        res.idft(module, &mut res_idft, scratch_1);

        self.prod_with_rlwe_inplace(module, &mut res_idft, scratch_1);

        module.vec_znx_dft(res, 0, &res_idft, 0);
        module.vec_znx_dft(res, 1, &res_idft, 1);
    }

    fn prod_with_mat_rlwe<RES, LHS>(&self, module: &Module<FFT64>, res: &mut RES, a: &LHS, scratch: &mut Scratch)
    where
        LHS: GetRow<FFT64> + Infos,
        RES: SetRow<FFT64> + Infos,
    {
        let (tmp_row_data, scratch1) = scratch.tmp_vec_znx_dft(module, 2, a.size());

        let mut tmp_a_row: RLWECtDft<&mut [u8], FFT64> = RLWECtDft::<&mut [u8], FFT64> {
            data: tmp_row_data,
            log_base2k: a.log_base2k(),
            log_k: a.log_k(),
        };

        let (tmp_res_data, scratch2) = scratch1.tmp_vec_znx_dft(module, 2, res.size());

        let mut tmp_res_row: RLWECtDft<&mut [u8], FFT64> = RLWECtDft::<&mut [u8], FFT64> {
            data: tmp_res_data,
            log_base2k: res.log_base2k(),
            log_k: res.log_k(),
        };

        let min_rows: usize = res.rows().min(a.rows());

        (0..res.rows()).for_each(|row_i| {
            (0..res.cols()).for_each(|col_j| {
                a.get_row(module, row_i, col_j, &mut tmp_a_row);
                self.prod_with_rlwe_dft(module, &mut tmp_res_row, &tmp_a_row, scratch2);
                res.set_row(module, row_i, col_j, &tmp_res_row);
            });
        });

        tmp_res_row.data.zero();

        (min_rows..res.rows()).for_each(|row_i| {
            (0..self.cols()).for_each(|col_j| {
                res.set_row(module, row_i, col_j, &tmp_res_row);
            });
        });
    }

    fn prod_with_mat_rlwe_inplace<RES>(&self, module: &Module<FFT64>, res: &mut RES, scratch: &mut Scratch)
    where
        RES: GetRow<FFT64> + SetRow<FFT64> + Infos,
    {
        let (tmp_row_data, scratch1) = scratch.tmp_vec_znx_dft(module, 2, res.size());

        let mut tmp_row: RLWECtDft<&mut [u8], FFT64> = RLWECtDft::<&mut [u8], FFT64> {
            data: tmp_row_data,
            log_base2k: res.log_base2k(),
            log_k: res.log_k(),
        };

        (0..res.rows()).for_each(|row_i| {
            (0..res.cols()).for_each(|col_j| {
                res.get_row(module, row_i, col_j, &mut tmp_row);
                self.prod_with_rlwe_dft_inplace(module, &mut tmp_row, scratch1);
                res.set_row(module, row_i, col_j, &tmp_row);
            });
        });
    }
}
