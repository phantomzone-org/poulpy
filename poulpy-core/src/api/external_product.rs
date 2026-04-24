use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, ScratchArena, VecZnxDft, ZnxZero},
};

use crate::{
    ScratchArenaTakeCore,
    layouts::{
        GGLWE, GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGLWEToMut, GGLWEToRef, GGSW, GGSWInfos, GGSWToBackendMut,
        GGSWToBackendRef, GGSWToMut, GGSWToRef, GLWEBackendMut, GLWEBackendRef, GLWEInfos, prepared::GGSWPreparedToBackendRef,
    },
};

pub trait GLWEExternalProduct<BE: Backend> {
    fn glwe_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos;

    fn glwe_external_product_inplace<'s, 'r, D>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &D,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        D: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
        BE: 's;

    fn glwe_external_product<'s, 'r, 'a, D>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        lhs: &GLWEBackendRef<'a, BE>,
        rhs: &D,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        D: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
        BE: 's;
}

pub trait GLWEExternalProductInternal<BE: Backend> {
    fn glwe_external_product_internal_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos;

    fn glwe_external_product_dft<'s, 'r, 'a, G>(
        &self,
        res_dft: &mut VecZnxDft<<BE as Backend>::BufMut<'r>, BE>,
        a: &GLWEBackendRef<'a, BE>,
        ggsw: &G,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        G: GGSWPreparedToBackendRef<BE>,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
        BE: 's;

    fn glwe_external_product_internal<'s, 'r, 'a, G>(
        &self,
        res_dft: VecZnxDft<<BE as Backend>::BufMut<'r>, BE>,
        a: &GLWEBackendRef<'a, BE>,
        ggsw: &G,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        G: GGSWPreparedToBackendRef<BE>,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
        BE: 's;
}

pub trait GGLWEExternalProduct<BE: Backend>
where
    Self: GLWEExternalProduct<BE>,
{
    fn gglwe_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos,
    {
        self.glwe_external_product_tmp_bytes(res_infos, a_infos, b_infos)
    }

    fn gglwe_external_product<'s, R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToMut + GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToRef + GGLWEToBackendRef<BE> + GGLWEInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
        BE: 's,
    {
        assert_eq!(
            res.rank_in(),
            a.rank_in(),
            "res input rank_in: {} != a input rank_in: {}",
            res.rank_in(),
            a.rank_in()
        );
        assert_eq!(
            a.rank_out(),
            b.rank(),
            "a output rank_out: {} != b rank: {}",
            a.rank_out(),
            b.rank()
        );
        assert_eq!(
            res.rank_out(),
            b.rank(),
            "res output rank_out: {} != b rank: {}",
            res.rank_out(),
            b.rank()
        );
        assert_eq!(res.base2k(), a.base2k());
        assert!(
            scratch.available() >= self.gglwe_external_product_tmp_bytes(res, a, b),
            "scratch.available(): {} < GGLWEExternalProduct::gglwe_external_product_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_external_product_tmp_bytes(res, a, b)
        );

        let min_dnum: usize = res.dnum().min(a.dnum()).into();
        let res_dnum: usize = res.dnum().into();
        let res_rank_in: usize = res.rank_in().into();

        {
            let mut res = res.to_backend_mut();
            let a = a.to_backend_ref();
            for row in 0..min_dnum {
                for col in 0..res_rank_in {
                    self.glwe_external_product(
                        &mut crate::layouts::gglwe_at_backend_mut_from_mut::<BE>(&mut res, row, col),
                        &crate::layouts::gglwe_at_backend_ref_from_ref::<BE>(&a, row, col),
                        b,
                        &mut scratch.borrow(),
                    );
                }
            }
        }

        if min_dnum < res_dnum {
            let res: &mut GGLWE<&mut [u8]> = &mut res.to_mut();
            for row in min_dnum..res_dnum {
                for col in 0..res_rank_in {
                    res.at_mut(row, col).data_mut().zero();
                }
            }
        }
    }

    fn gglwe_external_product_inplace<'s, R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToMut + GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
        BE: 's,
    {
        assert_eq!(
            res.rank_out(),
            a.rank(),
            "res output rank: {} != a rank: {}",
            res.rank_out(),
            a.rank()
        );
        assert!(
            scratch.available() >= self.gglwe_external_product_tmp_bytes(res, res, a),
            "scratch.available(): {} < GGLWEExternalProduct::gglwe_external_product_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_external_product_tmp_bytes(res, res, a)
        );

        let res_dnum: usize = res.dnum().into();
        let res_rank_in: usize = res.rank_in().into();
        let mut res = res.to_backend_mut();
        for row in 0..res_dnum {
            for col in 0..res_rank_in {
                self.glwe_external_product_inplace(
                    &mut crate::layouts::gglwe_at_backend_mut_from_mut::<BE>(&mut res, row, col),
                    a,
                    &mut scratch.borrow(),
                );
            }
        }
    }
}

pub trait GGSWExternalProduct<BE: Backend>
where
    Self: GLWEExternalProduct<BE> + ModuleN,
{
    fn ggsw_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos,
    {
        self.glwe_external_product_tmp_bytes(res_infos, a_infos, b_infos)
    }

    fn ggsw_external_product<'s, R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToMut + GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToRef + GGSWToBackendRef<BE> + GGSWInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
        BE: 's,
    {
        assert_eq!(res.rank(), a.rank(), "res rank: {} != a rank: {}", res.rank(), a.rank());
        assert_eq!(res.rank(), b.rank(), "res rank: {} != b rank: {}", res.rank(), b.rank());
        assert_eq!(res.base2k(), a.base2k());
        assert!(
            scratch.available() >= self.ggsw_external_product_tmp_bytes(res, a, b),
            "scratch.available(): {} < GGSWExternalProduct::ggsw_external_product_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_external_product_tmp_bytes(res, a, b)
        );

        let min_dnum: usize = res.dnum().min(a.dnum()).into();
        let res_dnum: usize = res.dnum().into();
        let res_rank: usize = (res.rank() + 1).into();

        {
            let mut res = res.to_backend_mut();
            let a = a.to_backend_ref();
            for row in 0..min_dnum {
                for col in 0..res_rank {
                    self.glwe_external_product(
                        &mut crate::layouts::ggsw_at_backend_mut_from_mut::<BE>(&mut res, row, col),
                        &crate::layouts::ggsw_at_backend_ref_from_ref::<BE>(&a, row, col),
                        b,
                        &mut scratch.borrow(),
                    );
                }
            }
        }

        if min_dnum < res_dnum {
            let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
            for row in min_dnum..res_dnum {
                for col in 0..res_rank {
                    res.at_mut(row, col).data.zero();
                }
            }
        }
    }

    fn ggsw_external_product_inplace<'s, R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToMut + GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
        BE: 's,
    {
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(res.rank(), a.rank(), "res rank: {} != a rank: {}", res.rank(), a.rank());
        assert!(
            scratch.available() >= self.ggsw_external_product_tmp_bytes(res, res, a),
            "scratch.available(): {} < GGSWExternalProduct::ggsw_external_product_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_external_product_tmp_bytes(res, res, a)
        );

        let res_dnum: usize = res.dnum().into();
        let res_rank: usize = (res.rank() + 1).into();
        let mut res = res.to_backend_mut();
        for row in 0..res_dnum {
            for col in 0..res_rank {
                self.glwe_external_product_inplace(
                    &mut crate::layouts::ggsw_at_backend_mut_from_mut::<BE>(&mut res, row, col),
                    a,
                    &mut scratch.borrow(),
                );
            }
        }
    }
}
