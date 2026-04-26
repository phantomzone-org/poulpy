use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    ScratchArenaTakeCore,
    api::GGSWExpandRows,
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWBackendMut, GGSWBackendRef, GGSWInfos, GLWEBackendMut,
        GLWEBackendRef, GLWEInfos, GetGaloisElement, SetGaloisElement,
        prepared::{GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
    },
};

pub trait GLWEAutomorphism<BE: Backend> {
    fn glwe_automorphism_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_automorphism<'s, 'r, 'a, K>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &GLWEBackendRef<'a, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn glwe_automorphism_inplace<'s, 'r, K>(&self, res: &mut GLWEBackendMut<'r, BE>, key: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn glwe_automorphism_add<'s, 'r, 'a, K>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &GLWEBackendRef<'a, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn glwe_automorphism_add_inplace<'s, 'r, K>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn glwe_automorphism_sub<'s, 'r, 'a, K>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &GLWEBackendRef<'a, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn glwe_automorphism_sub_negate<'s, 'r, 'a, K>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &GLWEBackendRef<'a, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn glwe_automorphism_sub_inplace<'s, 'r, K>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn glwe_automorphism_sub_negate_inplace<'s, 'r, K>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;
}

pub trait GGSWAutomorphism<BE: Backend>
where
    Self: GLWEAutomorphism<BE> + GGSWExpandRows<BE>,
{
    fn ggsw_automorphism_tmp_bytes<R, A, K, T>(&self, res_infos: &R, a_infos: &A, key_infos: &K, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        let lvl_0: usize = self
            .glwe_automorphism_tmp_bytes(res_infos, a_infos, key_infos)
            .max(self.ggsw_expand_rows_tmp_bytes(res_infos, tsk_infos));
        lvl_0
    }

    fn ggsw_automorphism<'s, 'r, 'a, K, T>(
        &self,
        res: &mut GGSWBackendMut<'r, BE>,
        a: &GGSWBackendRef<'a, BE>,
        key: &K,
        tsk: &T,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        assert_eq!(res.dsize(), a.dsize());
        assert_eq!(res.base2k, a.base2k);
        assert!(res.dnum() <= a.dnum());
        assert!(
            scratch.available() >= self.ggsw_automorphism_tmp_bytes(res, a, key, tsk),
            "scratch.available(): {} < GGSWAutomorphism::ggsw_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_automorphism_tmp_bytes(res, a, key, tsk)
        );

        for row in 0..res.dnum().as_usize() {
            self.glwe_automorphism(
                &mut crate::layouts::ggsw_at_backend_mut_from_mut::<BE>(res, row, 0),
                &crate::layouts::ggsw_at_backend_ref_from_ref::<BE>(a, row, 0),
                key,
                scratch,
            );
        }

        self.ggsw_expand_row(res, tsk, scratch)
    }

    fn ggsw_automorphism_inplace<'s, 'r, K, T>(
        &self,
        res: &mut GGSWBackendMut<'r, BE>,
        key: &K,
        tsk: &T,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        assert!(
            scratch.available() >= self.ggsw_automorphism_tmp_bytes(res, res, key, tsk),
            "scratch.available(): {} < GGSWAutomorphism::ggsw_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_automorphism_tmp_bytes(res, res, key, tsk)
        );

        for row in 0..res.dnum().as_usize() {
            self.glwe_automorphism_inplace(
                &mut crate::layouts::ggsw_at_backend_mut_from_mut::<BE>(res, row, 0),
                key,
                scratch,
            );
        }

        self.ggsw_expand_row(res, tsk, scratch)
    }
}

pub trait GLWEAutomorphismKeyAutomorphism<BE: Backend> {
    fn glwe_automorphism_key_automorphism_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos;

    fn glwe_automorphism_key_automorphism<'s, R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToBackendMut<BE> + SetGaloisElement + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos;

    fn glwe_automorphism_key_automorphism_inplace<'s, R, K>(&self, res: &mut R, key: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToBackendMut<BE> + SetGaloisElement + GetGaloisElement + GGLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos;
}
