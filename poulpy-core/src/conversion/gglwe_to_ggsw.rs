use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftZero, VecZnxIdftApply, VecZnxIdftApplyTmpBytes,
        VecZnxNormalize, VecZnxNormalizeTmpBytes,
    },
    layouts::{
        Backend, HostDataMut, Module, ScratchArena, VecZnx, VecZnxBackendRef, VecZnxBigReborrowBackendRef, VecZnxDftBackendRef,
        VecZnxDftReborrowBackendRef, VecZnxReborrowBackendRef,
    },
};

pub use crate::api::{GGSWExpandRows, GGSWFromGGLWE};
use crate::{
    GGLWEProduct, GLWECopy, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToBackendRef, GGSWBackendMut, GGSWInfos, GGSWToBackendMut, GLWEInfos, LWEInfos,
        gglwe_at_backend_ref_from_ref, ggsw_at_backend_mut_from_mut, ggsw_at_backend_ref_from_mut,
        prepared::{GGLWEToGGSWKeyPreparedBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
    },
};

pub(crate) trait GGSWFromGGLWEDefault<BE: Backend>: GGSWExpandRowsDefault<BE> + GLWECopy<BE>
where
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    for<'s> BE::BufMut<'s>: HostDataMut,
{
    fn ggsw_from_gglwe_tmp_bytes_default<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos,
    {
        let lvl_0: usize = self.ggsw_expand_rows_tmp_bytes_default(res_infos, tsk_infos);
        lvl_0
    }

    fn ggsw_from_gglwe_default<'s, R, A, T>(&self, res: &mut R, a: &A, tsk: &T, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        assert_eq!(res.rank(), a.rank_out());
        assert_eq!(res.dnum(), a.dnum());
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(tsk.n(), self.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(
            scratch.available() >= self.ggsw_from_gglwe_tmp_bytes_default(res, tsk),
            "scratch.available(): {} < GGSWFromGGLWE::ggsw_from_gglwe_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_from_gglwe_tmp_bytes_default(res, tsk)
        );

        {
            let res = &mut res.to_backend_mut();
            let a = &a.to_backend_ref();
            for row in 0..res.dnum().into() {
                self.glwe_copy(
                    &mut ggsw_at_backend_mut_from_mut::<BE>(res, row, 0),
                    &gglwe_at_backend_ref_from_ref::<BE>(a, row, 0),
                );
            }
        }

        let mut res_backend: GGSWBackendMut<'_, BE> = res.to_backend_mut();
        self.ggsw_expand_row_default(&mut res_backend, tsk, scratch)
    }
}

impl<BE: Backend> GGSWFromGGLWEDefault<BE> for Module<BE>
where
    Self: GGSWExpandRowsDefault<BE> + GLWECopy<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    for<'s> BE::BufMut<'s>: HostDataMut,
{
}

pub(crate) trait GGSWExpandRowsDefault<BE: Backend>:
    GGLWEProduct<BE>
    + VecZnxBigNormalize<BE>
    + VecZnxBigNormalizeTmpBytes
    + VecZnxBigBytesOf
    + VecZnxDftBytesOf
    + VecZnxDftApply<BE>
    + VecZnxNormalize<BE>
    + VecZnxNormalizeTmpBytes
    + VecZnxBigAddSmallAssign<BE>
    + VecZnxIdftApply<BE>
    + VecZnxIdftApplyTmpBytes
    + VecZnxDftZero<BE>
where
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn ggsw_expand_rows_tmp_bytes_default<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, res_infos.n());
        assert_eq!(self.n() as u32, tsk_infos.n());

        let tsk_base2k: usize = tsk_infos.base2k().into();

        let rank: usize = res_infos.rank().into();
        let cols: usize = rank + 1;

        let res_size: usize = res_infos.size();
        let a_size: usize = res_infos.max_k().as_usize().div_ceil(tsk_base2k);

        let lvl_0: usize = self.bytes_of_vec_znx_dft(cols - 1, a_size) + VecZnx::bytes_of(self.n(), 1, a_size);
        let lvl_1_res_dft: usize = self.bytes_of_vec_znx_dft(cols, a_size);
        let lvl_1_gglwe_prod: usize = self.gglwe_product_dft_tmp_bytes(res_size, a_size, tsk_infos);
        let lvl_1_big: usize = self.bytes_of_vec_znx_big(cols, res_size)
            + self
                .vec_znx_idft_apply_tmp_bytes()
                .max(self.vec_znx_big_normalize_tmp_bytes());
        let lvl_1: usize = lvl_1_res_dft + lvl_1_gglwe_prod.max(lvl_1_big);
        let lvl_2: usize = self.vec_znx_normalize_tmp_bytes();

        lvl_0 + lvl_1.max(lvl_2)
    }

    fn ggsw_expand_row_default<'s, 'r, T>(&self, res: &mut GGSWBackendMut<'r, BE>, tsk: &T, scratch: &mut ScratchArena<'s, BE>)
    where
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let res_base2k: usize = res.base2k().into();
        let tsk_base2k: usize = tsk.base2k().into();

        assert!(
            scratch.available() >= self.ggsw_expand_rows_tmp_bytes_default(res, tsk),
            "scratch.available(): {} < GGSWExpandRows::ggsw_expand_rows_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_expand_rows_tmp_bytes_default(res, tsk)
        );

        let rank: usize = res.rank().into();
        let cols: usize = rank + 1;

        let res_conv_size: usize = res.max_k().as_usize().div_ceil(tsk_base2k);
        {
            let (mut a_dft, scratch_1) = scratch.borrow().take_vec_znx_dft(self, cols - 1, res_conv_size);
            let (mut a_0, mut scratch_2) = scratch_1.take_vec_znx(self.n(), 1, res_conv_size);

            // Keyswitch the j-th row of the col 0
            for row in 0..res.dnum().as_usize() {
                {
                    let glwe_mi_1 = ggsw_at_backend_ref_from_mut::<BE>(&*res, row, 0);

                    for i in 0..cols - 1 {
                        self.vec_znx_normalize(
                            &mut a_0,
                            tsk_base2k,
                            0,
                            0,
                            &glwe_mi_1.data,
                            res_base2k,
                            i + 1,
                            &mut scratch_2.borrow(),
                        );
                        let a_0_ref: VecZnxBackendRef<'_, BE> =
                            <VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&a_0);
                        self.vec_znx_dft_apply(1, 0, &mut a_dft, i, &a_0_ref, 0);
                    }
                    self.vec_znx_normalize(
                        &mut a_0,
                        tsk_base2k,
                        0,
                        0,
                        &glwe_mi_1.data,
                        res_base2k,
                        0,
                        &mut scratch_2.borrow(),
                    );
                }

                let a_0_ref: VecZnxBackendRef<'_, BE> =
                    <VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&a_0);
                let a_dft_ref: VecZnxDftBackendRef<'_, BE> =
                    <poulpy_hal::layouts::VecZnxDft<BE::BufMut<'_>, BE> as VecZnxDftReborrowBackendRef<BE>>::reborrow_backend_ref(
                        &a_dft,
                    );
                let mut scratch_row = scratch_2.borrow();
                ggsw_expand_rows_internal(self, row, res, &a_0_ref, &a_dft_ref, tsk, &mut scratch_row);
            }
        }
    }
}

impl<BE: Backend> GGSWExpandRowsDefault<BE> for Module<BE>
where
    Self: GGLWEProduct<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigBytesOf
        + VecZnxDftBytesOf
        + VecZnxDftApply<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxDftZero<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
}

fn ggsw_expand_rows_internal<'r, 'a, 'b, M, T, BE: Backend>(
    module: &M,
    row: usize,
    res: &mut GGSWBackendMut<'r, BE>,
    a_0: &VecZnxBackendRef<'a, BE>,
    a_dft: &VecZnxDftBackendRef<'b, BE>,
    tsk: &T,
    scratch: &mut ScratchArena<'_, BE>,
) where
    M: GGLWEProduct<BE>
        + ModuleN
        + VecZnxBigBytesOf
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>,
    T: GGLWEToGGSWKeyPreparedToBackendRef<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    let tsk: GGLWEToGGSWKeyPreparedBackendRef<'_, BE> = tsk.to_backend_ref();
    let cols: usize = res.rank().as_usize() + 1;

    // Example for rank 3:
    //
    // Note: M is a vector (m, Bm, B^2m, B^3m, ...), so each column is
    // actually composed of that many dnum and we focus on a specific row here
    // implicitely given ci_dft.
    //
    // # Input
    //
    // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0    , a1    , a2    )
    // col 1: (0, 0, 0, 0)
    // col 2: (0, 0, 0, 0)
    // col 3: (0, 0, 0, 0)
    //
    // # Output
    //
    // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0       , a1       , a2       )
    // col 1: (-(b0s0 + b1s1 + b2s2)       , b0 + M[i], b1       , b2       )
    // col 2: (-(c0s0 + c1s1 + c2s2)       , c0       , c1 + M[i], c2       )
    // col 3: (-(d0s0 + d1s1 + d2s2)       , d0       , d1       , d2 + M[i])
    for col in 1..cols {
        let scratch_row = scratch.borrow();
        let (mut res_dft, mut scratch_1) = scratch_row.take_vec_znx_dft(module, cols, tsk.size()); // Todo optimise
        for j in 0..cols {
            module.vec_znx_dft_zero(&mut res_dft, j);
        }

        // Performs a key-switch for each combination of s[i]*s[j], i.e. for a0, a1, a2
        //
        // # Example for col=1
        //
        // a0 * (-(f0s0 + f1s1 + f1s2) + s0^2, f0, f1, f2) = (-(a0f0s0 + a0f1s1 + a0f1s2) + a0s0^2, a0f0, a0f1, a0f2)
        // +
        // a1 * (-(g0s0 + g1s1 + g1s2) + s0s1, g0, g1, g2) = (-(a1g0s0 + a1g1s1 + a1g1s2) + a1s0s1, a1g0, a1g1, a1g2)
        // +
        // a2 * (-(h0s0 + h1s1 + h1s2) + s0s2, h0, h1, h2) = (-(a2h0s0 + a2h1s1 + a2h1s2) + a2s0s2, a2h0, a2h1, a2h2)
        // =
        // (-(x0s0 + x1s1 + x2s2) + s0(a0s0 + a1s1 + a2s2), x0, x1, x2)
        {
            let mut scratch_prod = scratch_1.borrow();
            module.gglwe_product_dft(&mut res_dft, a_dft, tsk.at(col - 1), &mut scratch_prod);
        }

        let (mut res_big, mut scratch_2) = scratch_1.take_vec_znx_big(module, cols, res_dft.size);
        let res_dft_ref = res_dft.reborrow_backend_ref();
        for j in 0..cols {
            scratch_2 = scratch_2.apply_mut(|scratch| module.vec_znx_idft_apply(&mut res_big, j, &res_dft_ref, j, scratch));
        }

        // Adds -(sum a[i] * s[i]) + m)  on the i-th column of tmp_idft_i
        //
        // (-(x0s0 + x1s1 + x2s2) + a0s0s0 + a1s0s1 + a2s0s2, x0, x1, x2)
        // +
        // (0, -(a0s0 + a1s1 + a2s2) + M[i], 0, 0)
        // =
        // (-(x0s0 + x1s1 + x2s2) + s0(a0s0 + a1s1 + a2s2), x0 -(a0s0 + a1s1 + a2s2) + M[i], x1, x2)
        // =
        // (-(x0s0 + x1s1 + x2s2), x0 + M[i], x1, x2)
        module.vec_znx_big_add_small_assign(&mut res_big, col, a_0, 0);
        let res_big_ref = res_big.reborrow_backend_ref();

        let res_base2k: usize = res.base2k().as_usize();

        for j in 0..cols {
            let mut res_col = ggsw_at_backend_mut_from_mut::<BE>(res, row, col);
            let scratch_norm = &mut scratch_2.borrow();
            module.vec_znx_big_normalize(
                &mut res_col.data,
                res_base2k,
                0,
                j,
                &res_big_ref,
                tsk.base2k().as_usize(),
                j,
                scratch_norm,
            );
        }
    }
}
