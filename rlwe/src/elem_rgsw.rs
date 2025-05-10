use base2k::{
    Backend, FFT64, MatZnxDft, MatZnxDftAlloc, MatZnxDftOps, MatZnxDftScratch, MatZnxDftToMut, MatZnxDftToRef, Module, ScalarZnx,
    ScalarZnxDft, ScalarZnxDftToRef, ScalarZnxToRef, Scratch, VecZnx, VecZnxAlloc, VecZnxBig, VecZnxBigOps, VecZnxBigScratch,
    VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxDftToMut, VecZnxDftToRef, VecZnxOps, VecZnxToMut, VecZnxToRef, ZnxInfos,
    ZnxZero,
};
use sampling::source::Source;

use crate::{
    elem::Infos,
    elem_grlwe::GRLWECt,
    elem_rlwe::{RLWECt, RLWECtDft, RLWEPt, encrypt_rlwe_sk},
    keys::SecretKeyDft,
    utils::derive_size,
};

pub struct RGSWCt<C, B: Backend> {
    pub data: MatZnxDft<C, B>,
    pub log_base2k: usize,
    pub log_k: usize,
}

impl<B: Backend> RGSWCt<Vec<u8>, B> {
    pub fn new(module: &Module<B>, log_base2k: usize, log_k: usize, rows: usize) -> Self {
        Self {
            data: module.new_mat_znx_dft(rows, 2, 2, derive_size(log_base2k, log_k)),
            log_base2k: log_base2k,
            log_k: log_k,
        }
    }
}

impl<T, B: Backend> Infos for RGSWCt<T, B> {
    type Inner = MatZnxDft<T, B>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn log_base2k(&self) -> usize {
        self.log_base2k
    }

    fn log_k(&self) -> usize {
        self.log_k
    }
}

impl<C, B: Backend> MatZnxDftToMut<B> for RGSWCt<C, B>
where
    MatZnxDft<C, B>: MatZnxDftToMut<B>,
{
    fn to_mut(&mut self) -> MatZnxDft<&mut [u8], B> {
        self.data.to_mut()
    }
}

impl<C, B: Backend> MatZnxDftToRef<B> for RGSWCt<C, B>
where
    MatZnxDft<C, B>: MatZnxDftToRef<B>,
{
    fn to_ref(&self) -> MatZnxDft<&[u8], B> {
        self.data.to_ref()
    }
}

impl RGSWCt<Vec<u8>, FFT64> {
    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, size: usize) -> usize {
        RLWECt::encrypt_sk_scratch_space(module, size)
            + module.bytes_of_vec_znx(2, size)
            + module.bytes_of_vec_znx(1, size)
            + module.bytes_of_vec_znx_dft(2, size)
    }

    pub fn mul_rlwe_scratch_space(module: &Module<FFT64>, res_size: usize, a_size: usize, rgsw_size: usize) -> usize {
        module.bytes_of_vec_znx_dft(2, rgsw_size)
            + ((module.bytes_of_vec_znx_dft(2, a_size) + module.vmp_apply_tmp_bytes(res_size, a_size, a_size, 2, 2, rgsw_size))
                | module.vec_znx_big_normalize_tmp_bytes())
    }

    pub fn mul_rlwe_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, rgsw_size: usize) -> usize {
        Self::mul_rlwe_scratch_space(module, res_size, res_size, rgsw_size)
    }
}

impl<C> RGSWCt<C, FFT64>
where
    MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64>,
{
    pub fn get_row<R>(&self, module: &Module<FFT64>, row_i: usize, col_j: usize, res: &mut RLWECtDft<R, FFT64>)
    where
        VecZnxDft<R, FFT64>: VecZnxDftToMut<FFT64>,
    {
        module.vmp_extract_row(res, self, row_i, col_j);
    }
}

impl<C> RGSWCt<C, FFT64>
where
    MatZnxDft<C, FFT64>: MatZnxDftToMut<FFT64>,
{
    pub fn set_row<R>(&mut self, module: &Module<FFT64>, row_i: usize, col_j: usize, a: &RLWECtDft<R, FFT64>)
    where
        VecZnxDft<R, FFT64>: VecZnxDftToRef<FFT64>,
    {
        module.vmp_prepare_row(self, row_i, col_j, a);
    }
}

pub fn encrypt_rgsw_sk<C, P, S>(
    module: &Module<FFT64>,
    ct: &mut RGSWCt<C, FFT64>,
    pt: &ScalarZnx<P>,
    sk_dft: &SecretKeyDft<S, FFT64>,
    source_xa: &mut Source,
    source_xe: &mut Source,
    sigma: f64,
    bound: f64,
    scratch: &mut Scratch,
) where
    MatZnxDft<C, FFT64>: MatZnxDftToMut<FFT64>,
    ScalarZnx<P>: ScalarZnxToRef,
    ScalarZnxDft<S, FFT64>: ScalarZnxDftToRef<FFT64>,
{
    let size: usize = ct.size();
    let log_base2k: usize = ct.log_base2k();

    let (tmp_znx_pt, scratch_1) = scratch.tmp_vec_znx(module, 1, size);
    let (tmp_znx_ct, scrach_2) = scratch_1.tmp_vec_znx(module, 2, size);

    let mut vec_znx_pt: RLWEPt<&mut [u8]> = RLWEPt {
        data: tmp_znx_pt,
        log_base2k: log_base2k,
        log_k: ct.log_k(),
    };

    let mut vec_znx_ct: RLWECt<&mut [u8]> = RLWECt {
        data: tmp_znx_ct,
        log_base2k: log_base2k,
        log_k: ct.log_k(),
    };

    (0..ct.rows()).for_each(|row_j| {
        // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
        module.vec_znx_add_scalar_inplace(&mut vec_znx_pt, 0, row_j, pt, 0);
        module.vec_znx_normalize_inplace(log_base2k, &mut vec_znx_pt, 0, scrach_2);

        (0..ct.cols()).for_each(|col_i| {
            // rlwe encrypt of vec_znx_pt into vec_znx_ct
            encrypt_rlwe_sk(
                module,
                &mut vec_znx_ct,
                Some((&vec_znx_pt, col_i)),
                sk_dft,
                source_xa,
                source_xe,
                sigma,
                bound,
                scrach_2,
            );

            // Switch vec_znx_ct into DFT domain
            {
                let (mut vec_znx_dft_ct, _) = scrach_2.tmp_vec_znx_dft(module, 2, size);
                module.vec_znx_dft(&mut vec_znx_dft_ct, 0, &vec_znx_ct, 0);
                module.vec_znx_dft(&mut vec_znx_dft_ct, 1, &vec_znx_ct, 1);
                module.vmp_prepare_row(ct, row_j, col_i, &vec_znx_dft_ct);
            }
        });

        vec_znx_pt.data.zero(); // zeroes for next iteration
    });
}

impl<C> RGSWCt<C, FFT64> {
    pub fn encrypt_sk<P, S>(
        &mut self,
        module: &Module<FFT64>,
        pt: &ScalarZnx<P>,
        sk_dft: &SecretKeyDft<S, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        bound: f64,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<C, FFT64>: MatZnxDftToMut<FFT64>,
        ScalarZnx<P>: ScalarZnxToRef,
        ScalarZnxDft<S, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        encrypt_rgsw_sk(
            module, self, pt, sk_dft, source_xa, source_xe, sigma, bound, scratch,
        )
    }

    pub fn mul_rlwe<R, A>(&self, module: &Module<FFT64>, res: &mut RLWECt<R>, a: &RLWECt<A>, scratch: &mut Scratch)
    where
        MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64>,
        VecZnx<R>: VecZnxToMut,
        VecZnx<A>: VecZnxToRef,
    {
        let log_base2k: usize = self.log_base2k();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.log_base2k(), log_base2k);
            assert_eq!(a.log_base2k(), log_base2k);
            assert_eq!(self.n(), module.n());
            assert_eq!(res.n(), module.n());
            assert_eq!(a.n(), module.n());
        }

        let (mut res_dft, scratch1) = scratch.tmp_vec_znx_dft(module, 2, self.size()); // Todo optimise

        {
            let (mut a_dft, scratch2) = scratch1.tmp_vec_znx_dft(module, 2, a.size());
            module.vec_znx_dft(&mut a_dft, 0, a, 0);
            module.vec_znx_dft(&mut a_dft, 1, a, 1);
            module.vmp_apply(&mut res_dft, &a_dft, self, scratch2);
        }

        let res_big: VecZnxBig<&mut [u8], FFT64> = module.vec_znx_idft_consume(res_dft);

        module.vec_znx_big_normalize(log_base2k, res, 0, &res_big, 0, scratch1);
        module.vec_znx_big_normalize(log_base2k, res, 1, &res_big, 1, scratch1);
    }

    pub fn mul_rlwe_inplace<R>(&self, module: &Module<FFT64>, res: &mut RLWECt<R>, scratch: &mut Scratch)
    where
        MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64> + ZnxInfos,
        VecZnx<R>: VecZnxToMut + VecZnxToRef,
    {
        unsafe {
            let res_ptr: *mut RLWECt<R> = res as *mut RLWECt<R>; // This is ok because [Self::mul_rlwe] only updates res at the end.
            self.mul_rlwe(&module, &mut *res_ptr, &*res_ptr, scratch);
        }
    }

    pub fn mul_rlwe_dft<R, A>(
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

    pub fn mul_rlwe_dft_inplace<R>(&self, module: &Module<FFT64>, res: &mut RLWECtDft<R, FFT64>, scratch: &mut Scratch)
    where
        MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64> + ZnxInfos,
        VecZnxDft<R, FFT64>: VecZnxDftToMut<FFT64> + VecZnxDftToRef<FFT64> + ZnxInfos,
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

    pub fn mul_grlwe<R, A>(
        &self,
        module: &Module<FFT64>,
        res: &mut GRLWECt<R, FFT64>,
        a: &GRLWECt<A, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64> + ZnxInfos,
        MatZnxDft<R, FFT64>: MatZnxDftToMut<FFT64> + MatZnxDftToRef<FFT64> + ZnxInfos,
        MatZnxDft<A, FFT64>: MatZnxDftToRef<FFT64> + ZnxInfos,
    {
        let (tmp_row_data, scratch1) = scratch.tmp_vec_znx_dft(module, 2, a.size());

        let mut tmp_row: RLWECtDft<&mut [u8], FFT64> = RLWECtDft::<&mut [u8], FFT64> {
            data: tmp_row_data,
            log_base2k: a.log_base2k(),
            log_k: a.log_k(),
        };

        let min_rows: usize = res.rows().min(a.rows());

        (0..min_rows).for_each(|row_i| {
            a.get_row(module, row_i, &mut tmp_row);
            self.mul_rlwe_dft_inplace(module, &mut tmp_row, scratch1);
            res.set_row(module, row_i, &tmp_row);
        });

        tmp_row.data.zero();

        (min_rows..res.rows()).for_each(|row_i| {
            res.set_row(module, row_i, &tmp_row);
        })
    }

    pub fn mul_grlwe_inplace<R>(&self, module: &Module<FFT64>, res: &mut GRLWECt<R, FFT64>, scratch: &mut Scratch)
    where
        MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64> + ZnxInfos,
        MatZnxDft<R, FFT64>: MatZnxDftToMut<FFT64> + MatZnxDftToRef<FFT64> + ZnxInfos,
    {
        let (tmp_row_data, scratch1) = scratch.tmp_vec_znx_dft(module, 2, res.size());

        let mut tmp_row: RLWECtDft<&mut [u8], FFT64> = RLWECtDft::<&mut [u8], FFT64> {
            data: tmp_row_data,
            log_base2k: res.log_base2k(),
            log_k: res.log_k(),
        };

        (0..res.rows()).for_each(|row_i| {
            res.get_row(module, row_i, &mut tmp_row);
            self.mul_rlwe_dft_inplace(module, &mut tmp_row, scratch1);
            res.set_row(module, row_i, &tmp_row);
        });
    }

    pub fn mul_rgsw<R, A>(&self, module: &Module<FFT64>, res: &mut RGSWCt<R, FFT64>, a: &RGSWCt<A, FFT64>, scratch: &mut Scratch)
    where
        MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64> + ZnxInfos,
        MatZnxDft<R, FFT64>: MatZnxDftToMut<FFT64> + MatZnxDftToRef<FFT64> + ZnxInfos,
        MatZnxDft<A, FFT64>: MatZnxDftToRef<FFT64> + ZnxInfos,
    {
        let (tmp_row_data, scratch1) = scratch.tmp_vec_znx_dft(module, 2, a.size());

        let mut tmp_row: RLWECtDft<&mut [u8], FFT64> = RLWECtDft::<&mut [u8], FFT64> {
            data: tmp_row_data,
            log_base2k: a.log_base2k(),
            log_k: a.log_k(),
        };

        let min_rows: usize = res.rows().min(a.rows());

        (0..min_rows).for_each(|row_i| {
            a.get_row(module, row_i, 0, &mut tmp_row);
            self.mul_rlwe_dft_inplace(module, &mut tmp_row, scratch1);
            res.set_row(module, row_i, 0, &tmp_row);
        });

        (0..min_rows).for_each(|row_i| {
            a.get_row(module, row_i, 1, &mut tmp_row);
            self.mul_rlwe_dft_inplace(module, &mut tmp_row, scratch1);
            res.set_row(module, row_i, 1, &tmp_row);
        });

        tmp_row.data.zero();

        (min_rows..res.rows()).for_each(|row_i| {
            res.set_row(module, row_i, 0, &tmp_row);
            res.set_row(module, row_i, 1, &tmp_row);
        })
    }

    pub fn mul_rgsw_inplace<R>(&self, module: &Module<FFT64>, res: &mut RGSWCt<R, FFT64>, scratch: &mut Scratch)
    where
        MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64> + ZnxInfos,
        MatZnxDft<R, FFT64>: MatZnxDftToMut<FFT64> + MatZnxDftToRef<FFT64> + ZnxInfos,
    {
        let (tmp_row_data, scratch1) = scratch.tmp_vec_znx_dft(module, 2, res.size());

        let mut tmp_row: RLWECtDft<&mut [u8], FFT64> = RLWECtDft::<&mut [u8], FFT64> {
            data: tmp_row_data,
            log_base2k: res.log_base2k(),
            log_k: res.log_k(),
        };

        (0..res.rows()).for_each(|row_i| {
            res.get_row(module, row_i, 0, &mut tmp_row);
            self.mul_rlwe_dft_inplace(module, &mut tmp_row, scratch1);
            res.set_row(module, row_i, 0, &tmp_row);
        });

        (0..res.rows()).for_each(|row_i| {
            res.get_row(module, row_i, 1, &mut tmp_row);
            self.mul_rlwe_dft_inplace(module, &mut tmp_row, scratch1);
            res.set_row(module, row_i, 1, &tmp_row);
        });
    }
}
