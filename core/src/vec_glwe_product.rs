use base2k::{
    FFT64, Module, Scratch, VecZnx, VecZnxAlloc, VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxDftToMut, VecZnxDftToRef,
    VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxZero,
};

use crate::{
    elem::{GetRow, Infos, SetRow},
    glwe::{GLWECiphertext, GLWECiphertextFourier},
};

pub(crate) trait VecGLWEProductScratchSpace {
    fn prod_with_glwe_scratch_space(module: &Module<FFT64>, res_size: usize, lhs: usize, rhs: usize) -> usize;

    fn prod_with_glwe_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, rhs: usize) -> usize {
        Self::prod_with_glwe_scratch_space(module, res_size, res_size, rhs)
    }

    fn prod_with_glwe_dft_scratch_space(module: &Module<FFT64>, res_size: usize, lhs: usize, rhs: usize) -> usize {
        (Self::prod_with_glwe_scratch_space(module, res_size, lhs, rhs) | module.vec_znx_idft_tmp_bytes())
            + module.bytes_of_vec_znx(2, lhs)
            + module.bytes_of_vec_znx(2, res_size)
    }

    fn prod_with_glwe_dft_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, rhs: usize) -> usize {
        (Self::prod_with_glwe_inplace_scratch_space(module, res_size, rhs) | module.vec_znx_idft_tmp_bytes())
            + module.bytes_of_vec_znx(2, res_size)
    }

    fn prod_with_vec_glwe_scratch_space(module: &Module<FFT64>, res_size: usize, lhs: usize, rhs: usize) -> usize {
        Self::prod_with_glwe_dft_scratch_space(module, res_size, lhs, rhs)
            + module.bytes_of_vec_znx_dft(2, lhs)
            + module.bytes_of_vec_znx_dft(2, res_size)
    }

    fn prod_with_vec_glwe_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, rhs: usize) -> usize {
        Self::prod_with_glwe_dft_inplace_scratch_space(module, res_size, rhs) + module.bytes_of_vec_znx_dft(2, res_size)
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

        let (a_data, scratch_1) = scratch.tmp_vec_znx(module, 2, a.size());

        let mut a_idft: GLWECiphertext<&mut [u8]> = GLWECiphertext::<&mut [u8]> {
            data: a_data,
            log_base2k: a.basek(),
            log_k: a.k(),
        };

        a.idft(module, &mut a_idft, scratch_1);

        let (res_data, scratch_2) = scratch_1.tmp_vec_znx(module, 2, res.size());

        let mut res_idft: GLWECiphertext<&mut [u8]> = GLWECiphertext::<&mut [u8]> {
            data: res_data,
            log_base2k: res.basek(),
            log_k: res.k(),
        };

        self.prod_with_glwe(module, &mut res_idft, &a_idft, scratch_2);

        module.vec_znx_dft(res, 0, &res_idft, 0);
        module.vec_znx_dft(res, 1, &res_idft, 1);
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

        let (res_data, scratch_1) = scratch.tmp_vec_znx(module, 2, res.size());

        let mut res_idft: GLWECiphertext<&mut [u8]> = GLWECiphertext::<&mut [u8]> {
            data: res_data,
            log_base2k: res.basek(),
            log_k: res.k(),
        };

        res.idft(module, &mut res_idft, scratch_1);

        self.prod_with_glwe_inplace(module, &mut res_idft, scratch_1);

        module.vec_znx_dft(res, 0, &res_idft, 0);
        module.vec_znx_dft(res, 1, &res_idft, 1);
    }

    fn prod_with_vec_glwe<RES, LHS>(&self, module: &Module<FFT64>, res: &mut RES, a: &LHS, scratch: &mut Scratch)
    where
        LHS: GetRow<FFT64> + Infos,
        RES: SetRow<FFT64> + Infos,
    {
        let (tmp_row_data, scratch1) = scratch.tmp_vec_znx_dft(module, 2, a.size());

        let mut tmp_a_row: GLWECiphertextFourier<&mut [u8], FFT64> = GLWECiphertextFourier::<&mut [u8], FFT64> {
            data: tmp_row_data,
            log_base2k: a.basek(),
            log_k: a.k(),
        };

        let (tmp_res_data, scratch2) = scratch1.tmp_vec_znx_dft(module, 2, res.size());

        let mut tmp_res_row: GLWECiphertextFourier<&mut [u8], FFT64> = GLWECiphertextFourier::<&mut [u8], FFT64> {
            data: tmp_res_data,
            log_base2k: res.basek(),
            log_k: res.k(),
        };

        let min_rows: usize = res.rows().min(a.rows());

        (0..res.rows()).for_each(|row_i| {
            (0..res.rank()).for_each(|col_j| {
                a.get_row(module, row_i, col_j, &mut tmp_a_row);
                self.prod_with_glwe_fourier(module, &mut tmp_res_row, &tmp_a_row, scratch2);
                res.set_row(module, row_i, col_j, &tmp_res_row);
            });
        });

        tmp_res_row.data.zero();

        (min_rows..res.rows()).for_each(|row_i| {
            (0..self.rank()).for_each(|col_j| {
                res.set_row(module, row_i, col_j, &tmp_res_row);
            });
        });
    }

    fn prod_with_vec_glwe_inplace<RES>(&self, module: &Module<FFT64>, res: &mut RES, scratch: &mut Scratch)
    where
        RES: GetRow<FFT64> + SetRow<FFT64> + Infos,
    {
        let (tmp_row_data, scratch1) = scratch.tmp_vec_znx_dft(module, 2, res.size());

        let mut tmp_row: GLWECiphertextFourier<&mut [u8], FFT64> = GLWECiphertextFourier::<&mut [u8], FFT64> {
            data: tmp_row_data,
            log_base2k: res.basek(),
            log_k: res.k(),
        };

        (0..res.rows()).for_each(|row_i| {
            (0..res.rank()).for_each(|col_j| {
                res.get_row(module, row_i, col_j, &mut tmp_row);
                self.prod_with_glwe_fourier_inplace(module, &mut tmp_row, scratch1);
                res.set_row(module, row_i, col_j, &tmp_row);
            });
        });
    }
}
