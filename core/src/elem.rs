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

pub trait ProdByScratchSpace {
    fn prod_by_grlwe_scratch_space(module: &Module<FFT64>, lhs: usize, rhs: usize) -> usize;
    fn prod_by_rgsw_scratch_space(module: &Module<FFT64>, lhs: usize, rhs: usize) -> usize;
}

pub trait ProdBy<D> {
    fn prod_by_grlwe<R>(&mut self, module: &Module<FFT64>, rhs: &GRLWECt<R, FFT64>, scratch: &mut Scratch)
    where
        MatZnxDft<R, FFT64>: MatZnxDftToRef<FFT64>;

    fn prod_by_rgsw<R>(&mut self, module: &Module<FFT64>, rhs: &RGSWCt<R, FFT64>, scratch: &mut Scratch)
    where
        MatZnxDft<R, FFT64>: MatZnxDftToRef<FFT64>;
}

pub trait FromProdByScratchSpace {
    fn from_prod_by_grlwe_scratch_space(module: &Module<FFT64>, res_size: usize, lhs: usize, rhs: usize) -> usize;
    fn from_prod_by_rgsw_scratch_space(module: &Module<FFT64>, res_size: usize, lhs: usize, rhs: usize) -> usize;
}

pub trait FromProdBy<D, L> {
    fn from_prod_by_grlwe<R>(&mut self, module: &Module<FFT64>, lhs: &L, rhs: &GRLWECt<R, FFT64>, scratch: &mut Scratch)
    where
        MatZnxDft<R, FFT64>: MatZnxDftToRef<FFT64>;

    fn from_prod_by_rgsw<R>(&mut self, module: &Module<FFT64>, lhs: &L, rhs: &RGSWCt<R, FFT64>, scratch: &mut Scratch)
    where
        MatZnxDft<R, FFT64>: MatZnxDftToRef<FFT64>;
}

pub(crate) trait MatZnxDftProducts<D, C>: Infos
where
    MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64> + ZnxInfos,
{
    fn mul_rlwe<R, A>(&self, module: &Module<FFT64>, res: &mut RLWECt<R>, a: &RLWECt<A>, scratch: &mut Scratch)
    where
        MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64>,
        VecZnx<R>: VecZnxToMut,
        VecZnx<A>: VecZnxToRef;

    fn mul_rlwe_scratch_space(module: &Module<FFT64>, res_size: usize, a_size: usize, grlwe_size: usize) -> usize;

    fn mul_rlwe_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, mat_size: usize) -> usize {
        Self::mul_rlwe_scratch_space(module, res_size, res_size, mat_size)
    }

    fn mul_rlwe_dft_scratch_space(module: &Module<FFT64>, res_size: usize, a_size: usize, mat_size: usize) -> usize {
        (Self::mul_rlwe_scratch_space(module, res_size, a_size, mat_size) | module.vec_znx_idft_tmp_bytes())
            + module.bytes_of_vec_znx(2, a_size)
            + module.bytes_of_vec_znx(2, res_size)
    }

    fn mul_rlwe_dft_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, mat_size: usize) -> usize {
        (Self::mul_rlwe_inplace_scratch_space(module, res_size, mat_size) | module.vec_znx_idft_tmp_bytes())
            + module.bytes_of_vec_znx(2, res_size)
    }

    fn mul_mat_rlwe_scratch_space(module: &Module<FFT64>, res_size: usize, a_size: usize, mat_size: usize) -> usize {
        Self::mul_rlwe_dft_inplace_scratch_space(module, res_size, mat_size) + module.bytes_of_vec_znx_dft(2, a_size)
    }

    fn mul_mat_rlwe_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, mat_size: usize) -> usize {
        Self::mul_rlwe_dft_inplace_scratch_space(module, res_size, mat_size) + module.bytes_of_vec_znx_dft(2, res_size)
    }

    fn mul_rlwe_inplace<R>(&self, module: &Module<FFT64>, res: &mut RLWECt<R>, scratch: &mut Scratch)
    where
        MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64> + ZnxInfos,
        VecZnx<R>: VecZnxToMut + VecZnxToRef,
    {
        unsafe {
            let res_ptr: *mut RLWECt<R> = res as *mut RLWECt<R>; // This is ok because [Self::mul_rlwe] only updates res at the end.
            self.mul_rlwe(&module, &mut *res_ptr, &*res_ptr, scratch);
        }
    }

    fn mul_rlwe_dft<R, A>(
        &self,
        module: &Module<FFT64>,
        res: &mut RLWECtDft<R, FFT64>,
        a: &RLWECtDft<A, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64> + ZnxInfos,
        VecZnxDft<R, FFT64>: VecZnxDftToMut<FFT64> + VecZnxDftToRef<FFT64> + ZnxInfos,
        VecZnxDft<A, FFT64>: VecZnxDftToRef<FFT64> + ZnxInfos,
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

        self.mul_rlwe(module, &mut res_idft, &a_idft, scratch_2);

        module.vec_znx_dft(res, 0, &res_idft, 0);
        module.vec_znx_dft(res, 1, &res_idft, 1);
    }

    fn mul_rlwe_dft_inplace<R>(&self, module: &Module<FFT64>, res: &mut RLWECtDft<R, FFT64>, scratch: &mut Scratch)
    where
        VecZnxDft<R, FFT64>: VecZnxDftToRef<FFT64> + VecZnxDftToMut<FFT64>,
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

        self.mul_rlwe_inplace(module, &mut res_idft, scratch_1);

        module.vec_znx_dft(res, 0, &res_idft, 0);
        module.vec_znx_dft(res, 1, &res_idft, 1);
    }

    fn mul_mat_rlwe<R, A>(&self, module: &Module<FFT64>, res: &mut R, a: &A, scratch: &mut Scratch)
    where
        A: GetRow<FFT64> + Infos,
        R: SetRow<FFT64> + Infos,
    {
        let (tmp_row_data, scratch1) = scratch.tmp_vec_znx_dft(module, 2, a.size());

        let mut tmp_row: RLWECtDft<&mut [u8], FFT64> = RLWECtDft::<&mut [u8], FFT64> {
            data: tmp_row_data,
            log_base2k: a.log_base2k(),
            log_k: a.log_k(),
        };

        let min_rows: usize = res.rows().min(a.rows());

        (0..res.rows()).for_each(|row_i| {
            (0..res.cols()).for_each(|col_j| {
                a.get_row(module, row_i, col_j, &mut tmp_row);
                self.mul_rlwe_dft_inplace(module, &mut tmp_row, scratch1);
                res.set_row(module, row_i, col_j, &tmp_row);
            });
        });

        tmp_row.data.zero();

        (min_rows..res.rows()).for_each(|row_i| {
            (0..self.cols()).for_each(|col_j| {
                res.set_row(module, row_i, col_j, &tmp_row);
            });
        });
    }

    fn mul_mat_rlwe_inplace<R>(&self, module: &Module<FFT64>, res: &mut R, scratch: &mut Scratch)
    where
        R: GetRow<FFT64> + SetRow<FFT64> + Infos,
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
                self.mul_rlwe_dft_inplace(module, &mut tmp_row, scratch1);
                res.set_row(module, row_i, col_j, &tmp_row);
            });
        });
    }
}
