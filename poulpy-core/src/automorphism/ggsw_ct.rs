use poulpy_hal::layouts::{Backend, Module, ScratchArena};

pub use crate::api::GGSWAutomorphism;
use crate::{
    GGSWExpandRows, ScratchArenaTakeCore,
    automorphism::glwe_ct::GLWEAutomorphism,
    layouts::{
        GGLWEInfos, GGSWBackendMut, GGSWBackendRef, GGSWInfos, GetGaloisElement, ggsw_at_backend_mut_from_mut,
        ggsw_at_backend_ref_from_ref,
        prepared::{GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
    },
};

#[doc(hidden)]
pub trait GGSWAutomorphismDefault<BE: Backend>
where
    Self: GLWEAutomorphism<BE> + GGSWExpandRows<BE>,
{
    fn ggsw_automorphism_tmp_bytes_default<R, A, K, T>(&self, res_infos: &R, a_infos: &A, key_infos: &K, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        self.glwe_automorphism_tmp_bytes(res_infos, a_infos, key_infos)
            .max(self.ggsw_expand_rows_tmp_bytes(res_infos, tsk_infos))
    }

    fn ggsw_automorphism_default<'s, 'r, 'a, K, T>(
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
    {
        assert_eq!(res.dsize(), a.dsize());
        assert_eq!(res.base2k, a.base2k);
        assert!(res.dnum() <= a.dnum());
        assert!(
            scratch.available() >= self.ggsw_automorphism_tmp_bytes_default(res, a, key, tsk),
            "scratch.available(): {} < GGSWAutomorphism::ggsw_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_automorphism_tmp_bytes_default(res, a, key, tsk)
        );

        for row in 0..res.dnum().as_usize() {
            self.glwe_automorphism(
                &mut ggsw_at_backend_mut_from_mut::<BE>(res, row, 0),
                &ggsw_at_backend_ref_from_ref::<BE>(a, row, 0),
                key,
                &mut scratch.borrow(),
            );
        }

        self.ggsw_expand_row(res, tsk, scratch)
    }

    fn ggsw_automorphism_inplace_default<'s, 'r, K, T>(
        &self,
        res: &mut GGSWBackendMut<'r, BE>,
        key: &K,
        tsk: &T,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    {
        assert!(
            scratch.available() >= self.ggsw_automorphism_tmp_bytes_default(res, res, key, tsk),
            "scratch.available(): {} < GGSWAutomorphism::ggsw_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_automorphism_tmp_bytes_default(res, res, key, tsk)
        );

        for row in 0..res.dnum().as_usize() {
            self.glwe_automorphism_inplace(
                &mut ggsw_at_backend_mut_from_mut::<BE>(res, row, 0),
                key,
                &mut scratch.borrow(),
            );
        }

        self.ggsw_expand_row(res, tsk, scratch)
    }
}

impl<BE: Backend> GGSWAutomorphismDefault<BE> for Module<BE>
where
    Self: GLWEAutomorphism<BE> + GGSWExpandRows<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
}
