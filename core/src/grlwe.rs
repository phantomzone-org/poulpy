use base2k::{
    Backend, FFT64, MatZnxDft, MatZnxDftAlloc, MatZnxDftOps, MatZnxDftScratch, MatZnxDftToMut, MatZnxDftToRef, Module, ScalarZnx,
    ScalarZnxDft, ScalarZnxDftToRef, ScalarZnxToRef, Scratch, VecZnx, VecZnxAlloc, VecZnxBig, VecZnxBigOps, VecZnxBigScratch,
    VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxDftToMut, VecZnxDftToRef, VecZnxOps, VecZnxToMut, VecZnxToRef, ZnxInfos,
    ZnxZero,
};
use sampling::source::Source;

use crate::{
    elem::{GetRow, Infos, MatZnxDftProducts, SetRow},
    keys::SecretKeyDft,
    rlwe::{RLWECt, RLWECtDft, RLWEPt},
    utils::derive_size,
};

pub struct GRLWECt<C, B: Backend> {
    pub data: MatZnxDft<C, B>,
    pub log_base2k: usize,
    pub log_k: usize,
}

impl<B: Backend> GRLWECt<Vec<u8>, B> {
    pub fn new(module: &Module<B>, log_base2k: usize, log_k: usize, rows: usize) -> Self {
        Self {
            data: module.new_mat_znx_dft(rows, 1, 2, derive_size(log_base2k, log_k)),
            log_base2k: log_base2k,
            log_k: log_k,
        }
    }
}

impl<C> GRLWECt<C, FFT64>
where
    MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64>,
{
    pub fn get_row<R>(&self, module: &Module<FFT64>, row_i: usize, res: &mut RLWECtDft<R, FFT64>)
    where
        VecZnxDft<R, FFT64>: VecZnxDftToMut<FFT64>,
    {
        module.vmp_extract_row(res, self, row_i, 0);
    }
}

impl<C> GRLWECt<C, FFT64>
where
    MatZnxDft<C, FFT64>: MatZnxDftToMut<FFT64>,
{
    pub fn set_row<R>(&mut self, module: &Module<FFT64>, row_i: usize, a: &RLWECtDft<R, FFT64>)
    where
        VecZnxDft<R, FFT64>: VecZnxDftToRef<FFT64>,
    {
        module.vmp_prepare_row(self, row_i, 0, a);
    }
}

impl<T, B: Backend> Infos for GRLWECt<T, B> {
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

impl<C, B: Backend> MatZnxDftToMut<B> for GRLWECt<C, B>
where
    MatZnxDft<C, B>: MatZnxDftToMut<B>,
{
    fn to_mut(&mut self) -> MatZnxDft<&mut [u8], B> {
        self.data.to_mut()
    }
}

impl<C, B: Backend> MatZnxDftToRef<B> for GRLWECt<C, B>
where
    MatZnxDft<C, B>: MatZnxDftToRef<B>,
{
    fn to_ref(&self) -> MatZnxDft<&[u8], B> {
        self.data.to_ref()
    }
}

impl GRLWECt<Vec<u8>, FFT64> {
    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, size: usize) -> usize {
        RLWECt::encrypt_sk_scratch_space(module, size)
            + module.bytes_of_vec_znx(2, size)
            + module.bytes_of_vec_znx(1, size)
            + module.bytes_of_vec_znx_dft(2, size)
    }

    pub fn mul_rlwe_scratch_space(module: &Module<FFT64>, res_size: usize, a_size: usize, grlwe_size: usize) -> usize {
        module.bytes_of_vec_znx_dft(2, grlwe_size)
            + (module.vec_znx_big_normalize_tmp_bytes()
                | (module.vmp_apply_tmp_bytes(res_size, a_size, a_size, 1, 2, grlwe_size)
                    + module.bytes_of_vec_znx_dft(1, a_size)))
    }

    pub fn mul_rlwe_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, grlwe_size: usize) -> usize {
        Self::mul_rlwe_scratch_space(module, res_size, res_size, grlwe_size)
    }

    pub fn mul_rlwe_dft_scratch_space(module: &Module<FFT64>, res_size: usize, a_size: usize, grlwe_size: usize) -> usize {
        (Self::mul_rlwe_scratch_space(module, res_size, a_size, grlwe_size) | module.vec_znx_idft_tmp_bytes())
            + module.bytes_of_vec_znx(2, a_size)
            + module.bytes_of_vec_znx(2, res_size)
    }

    pub fn mul_rlwe_dft_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, grlwe_size: usize) -> usize {
        (Self::mul_rlwe_inplace_scratch_space(module, res_size, grlwe_size) | module.vec_znx_idft_tmp_bytes())
            + module.bytes_of_vec_znx(2, res_size)
    }

    pub fn mul_grlwe_scratch_space(module: &Module<FFT64>, res_size: usize, a_size: usize, grlwe_size: usize) -> usize {
        Self::mul_rlwe_dft_inplace_scratch_space(module, res_size, grlwe_size) + module.bytes_of_vec_znx_dft(2, a_size)
    }

    pub fn mul_grlwe_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, a_size: usize, grlwe_size: usize) -> usize {
        Self::mul_rlwe_dft_inplace_scratch_space(module, res_size, grlwe_size) + module.bytes_of_vec_znx_dft(2, a_size)
    }
}

pub fn encrypt_grlwe_sk<C, P, S>(
    module: &Module<FFT64>,
    ct: &mut GRLWECt<C, FFT64>,
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
    let rows: usize = ct.rows();
    let size: usize = ct.size();
    let log_base2k: usize = ct.log_base2k();

    let (tmp_znx_pt, scrach_1) = scratch.tmp_vec_znx(module, 1, size);
    let (tmp_znx_ct, scrach_2) = scrach_1.tmp_vec_znx(module, 2, size);
    let (mut vec_znx_dft_ct, scratch_3) = scrach_2.tmp_vec_znx_dft(module, 2, size);

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

    (0..rows).for_each(|row_i| {
        // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
        module.vec_znx_add_scalar_inplace(&mut vec_znx_pt, 0, row_i, pt, 0);
        module.vec_znx_normalize_inplace(log_base2k, &mut vec_znx_pt, 0, scratch_3);

        // rlwe encrypt of vec_znx_pt into vec_znx_ct
        vec_znx_ct.encrypt_sk(
            module,
            Some(&vec_znx_pt),
            sk_dft,
            source_xa,
            source_xe,
            sigma,
            bound,
            scratch_3,
        );

        vec_znx_pt.data.zero(); // zeroes for next iteration

        // Switch vec_znx_ct into DFT domain
        module.vec_znx_dft(&mut vec_znx_dft_ct, 0, &vec_znx_ct, 0);
        module.vec_znx_dft(&mut vec_znx_dft_ct, 1, &vec_znx_ct, 1);

        // Stores vec_znx_dft_ct into thw i-th row of the MatZnxDft
        module.vmp_prepare_row(ct, row_i, 0, &vec_znx_dft_ct);
    });
}

impl<C> GRLWECt<C, FFT64> {
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
        encrypt_grlwe_sk(
            module, self, pt, sk_dft, source_xa, source_xe, sigma, bound, scratch,
        )
    }

    pub fn mul_rlwe<R, A>(&self, module: &Module<FFT64>, res: &mut RLWECt<R>, a: &RLWECt<A>, scratch: &mut Scratch)
    where
        MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64>,
        VecZnx<R>: VecZnxToMut,
        VecZnx<A>: VecZnxToRef,
    {
        MatZnxDftProducts::mul_rlwe(self, module, res, a, scratch);
    }

    pub fn mul_rlwe_inplace<R>(&self, module: &Module<FFT64>, res: &mut RLWECt<R>, scratch: &mut Scratch)
    where
        MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64> + ZnxInfos,
        VecZnx<R>: VecZnxToMut + VecZnxToRef,
    {
        MatZnxDftProducts::mul_rlwe_inplace(self, module, res, scratch);
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
        MatZnxDftProducts::mul_rlwe_dft(self, module, res, a, scratch);
    }

    pub fn mul_rlwe_dft_inplace<R>(&self, module: &Module<FFT64>, res: &mut RLWECtDft<R, FFT64>, scratch: &mut Scratch)
    where
        MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64> + ZnxInfos,
        VecZnxDft<R, FFT64>: VecZnxDftToRef<FFT64> + VecZnxDftToMut<FFT64>,
    {
        MatZnxDftProducts::mul_rlwe_dft_inplace(self, module, res, scratch);
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
        MatZnxDftProducts::mul_grlwe(self, module, res, a, scratch);
    }

    pub fn mul_grlwe_inplace<R>(&self, module: &Module<FFT64>, res: &mut R, scratch: &mut Scratch)
    where
        MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64> + ZnxInfos,
        R: GetRow<FFT64> + SetRow<FFT64> + Infos,
    {
        MatZnxDftProducts::mul_grlwe_inplace(self, module, res, scratch);
    }
}

impl<C> GetRow<FFT64> for GRLWECt<C, FFT64>
where
    MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64>,
{
    fn get_row<R>(&self, module: &Module<FFT64>, row_i: usize, col_j: usize, res: &mut RLWECtDft<R, FFT64>)
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

impl<C> SetRow<FFT64> for GRLWECt<C, FFT64>
where
    MatZnxDft<C, FFT64>: MatZnxDftToMut<FFT64>,
{
    fn set_row<R>(&mut self, module: &Module<FFT64>, row_i: usize, col_j: usize, a: &RLWECtDft<R, FFT64>)
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

impl<C> MatZnxDftProducts<GRLWECt<C, FFT64>, C> for GRLWECt<C, FFT64>
where
    MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64> + ZnxInfos,
{
    fn mul_rlwe<R, A>(&self, module: &Module<FFT64>, res: &mut RLWECt<R>, a: &RLWECt<A>, scratch: &mut Scratch)
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
