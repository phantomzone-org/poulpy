use base2k::{
    FFT64, Module, Scratch, VecZnx, VecZnxAlloc, VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxDftToMut, VecZnxDftToRef,
    VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxZero,
};

use crate::{
    elem::{GetRow, Infos, SetRow},
    glwe_ciphertext::GLWECiphertext,
    glwe_ciphertext_fourier::GLWECiphertextFourier,
};

pub(crate) trait VecGLWEProductScratchSpace {
    fn prod_with_glwe_scratch_space(
        module: &Module<FFT64>,
        res_size: usize,
        lhs: usize,
        rhs: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize;

    fn prod_with_glwe_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, rhs: usize, rank: usize) -> usize {
        Self::prod_with_glwe_scratch_space(module, res_size, res_size, rhs, rank, rank)
    }

    fn prod_with_glwe_dft_scratch_space(
        module: &Module<FFT64>,
        res_size: usize,
        lhs: usize,
        rhs: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize {
        (Self::prod_with_glwe_scratch_space(module, res_size, lhs, rhs, rank_in, rank_out) | module.vec_znx_idft_tmp_bytes())
            + module.bytes_of_vec_znx(rank_in, lhs)
            + module.bytes_of_vec_znx(rank_out, res_size)
    }

    fn prod_with_glwe_dft_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, rhs: usize, rank: usize) -> usize {
        (Self::prod_with_glwe_inplace_scratch_space(module, res_size, rhs, rank) | module.vec_znx_idft_tmp_bytes())
            + module.bytes_of_vec_znx(rank + 1, res_size)
    }

    fn prod_with_vec_glwe_scratch_space(
        module: &Module<FFT64>,
        res_size: usize,
        lhs: usize,
        rhs: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize {
        Self::prod_with_glwe_dft_scratch_space(module, res_size, lhs, rhs, rank_in, rank_out)
            + module.bytes_of_vec_znx_dft(rank_in + 1, lhs)
            + module.bytes_of_vec_znx_dft(rank_out + 1, res_size)
    }

    fn prod_with_vec_glwe_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, rhs: usize, rank: usize) -> usize {
        Self::prod_with_glwe_dft_inplace_scratch_space(module, res_size, rhs, rank)
            + module.bytes_of_vec_znx_dft(rank + 1, res_size)
    }
}

pub(crate) trait VecGLWEProduct: Infos {
    fn prod_with_glwe<MUT, REF>(
        &self,
        module: &Module<FFT64>,
        res: &mut GLWECiphertext<MUT>,
        a: &GLWECiphertext<REF>,
        scratch: &mut Scratch,
    ) where
        VecZnx<MUT>: VecZnxToMut,
        VecZnx<REF>: VecZnxToRef;

    fn prod_with_glwe_inplace<MUT>(&self, module: &Module<FFT64>, res: &mut GLWECiphertext<MUT>, scratch: &mut Scratch)
    where
        VecZnx<MUT>: VecZnxToMut + VecZnxToRef,
    {
        unsafe {
            let res_ptr: *mut GLWECiphertext<MUT> = res as *mut GLWECiphertext<MUT>; // This is ok because [Self::mul_rlwe] only updates res at the end.
            self.prod_with_glwe(&module, &mut *res_ptr, &*res_ptr, scratch);
        }
    }

    fn prod_with_glwe_fourier<MUT, REF>(
        &self,
        module: &Module<FFT64>,
        res: &mut GLWECiphertextFourier<MUT, FFT64>,
        a: &GLWECiphertextFourier<REF, FFT64>,
        scratch: &mut Scratch,
    ) where
        VecZnxDft<MUT, FFT64>: VecZnxDftToMut<FFT64> + VecZnxDftToRef<FFT64> + ZnxInfos,
        VecZnxDft<REF, FFT64>: VecZnxDftToRef<FFT64> + ZnxInfos,
    {
        let log_base2k: usize = self.basek();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.basek(), log_base2k);
            assert_eq!(self.n(), module.n());
            assert_eq!(res.n(), module.n());
        }

        let (a_data, scratch_1) = scratch.tmp_vec_znx(module, a.rank() + 1, a.size());

        let mut a_idft: GLWECiphertext<&mut [u8]> = GLWECiphertext::<&mut [u8]> {
            data: a_data,
            basek: a.basek(),
            k: a.k(),
        };

        a.idft(module, &mut a_idft, scratch_1);

        let (res_data, scratch_2) = scratch_1.tmp_vec_znx(module, res.rank() + 1, res.size());

        let mut res_idft: GLWECiphertext<&mut [u8]> = GLWECiphertext::<&mut [u8]> {
            data: res_data,
            basek: res.basek(),
            k: res.k(),
        };

        self.prod_with_glwe(module, &mut res_idft, &a_idft, scratch_2);

        res_idft.dft(module, res);
    }

    fn prod_with_glwe_fourier_inplace<MUT>(
        &self,
        module: &Module<FFT64>,
        res: &mut GLWECiphertextFourier<MUT, FFT64>,
        scratch: &mut Scratch,
    ) where
        VecZnxDft<MUT, FFT64>: VecZnxDftToRef<FFT64> + VecZnxDftToMut<FFT64>,
    {
        let log_base2k: usize = self.basek();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.basek(), log_base2k);
            assert_eq!(self.n(), module.n());
            assert_eq!(res.n(), module.n());
        }

        let (res_data, scratch_1) = scratch.tmp_vec_znx(module, res.rank() + 1, res.size());

        let mut res_idft: GLWECiphertext<&mut [u8]> = GLWECiphertext::<&mut [u8]> {
            data: res_data,
            basek: res.basek(),
            k: res.k(),
        };

        res.idft(module, &mut res_idft, scratch_1);

        self.prod_with_glwe_inplace(module, &mut res_idft, scratch_1);

        res_idft.dft(module, res);
    }

    fn prod_with_vec_glwe<RES, LHS>(&self, module: &Module<FFT64>, res: &mut RES, a: &LHS, scratch: &mut Scratch)
    where
        LHS: GetRow<FFT64> + Infos,
        RES: SetRow<FFT64> + Infos,
    {
        let (tmp_row_data, scratch1) = scratch.tmp_vec_znx_dft(module, a.cols(), a.size());

        let mut tmp_a_row: GLWECiphertextFourier<&mut [u8], FFT64> = GLWECiphertextFourier::<&mut [u8], FFT64> {
            data: tmp_row_data,
            basek: a.basek(),
            k: a.k(),
        };

        let (tmp_res_data, scratch2) = scratch1.tmp_vec_znx_dft(module, res.cols(), res.size());

        let mut tmp_res_row: GLWECiphertextFourier<&mut [u8], FFT64> = GLWECiphertextFourier::<&mut [u8], FFT64> {
            data: tmp_res_data,
            basek: res.basek(),
            k: res.k(),
        };

        let min_rows: usize = res.rows().min(a.rows());

        (0..res.rows()).for_each(|row_i| {
            (0..res.cols()).for_each(|col_j| {
                a.get_row(module, row_i, col_j, &mut tmp_a_row);
                self.prod_with_glwe_fourier(module, &mut tmp_res_row, &tmp_a_row, scratch2);
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

    fn prod_with_vec_glwe_inplace<RES>(&self, module: &Module<FFT64>, res: &mut RES, scratch: &mut Scratch)
    where
        RES: GetRow<FFT64> + SetRow<FFT64> + Infos,
    {
        let (tmp_row_data, scratch1) = scratch.tmp_vec_znx_dft(module, res.cols(), res.size());

        let mut tmp_row: GLWECiphertextFourier<&mut [u8], FFT64> = GLWECiphertextFourier::<&mut [u8], FFT64> {
            data: tmp_row_data,
            basek: res.basek(),
            k: res.k(),
        };

        (0..res.rows()).for_each(|row_i| {
            (0..res.cols()).for_each(|col_j| {
                res.get_row(module, row_i, col_j, &mut tmp_row);
                self.prod_with_glwe_fourier_inplace(module, &mut tmp_row, scratch1);
                res.set_row(module, row_i, col_j, &tmp_row);
            });
        });
    }
}
