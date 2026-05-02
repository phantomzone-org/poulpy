use poulpy_hal::layouts::{Backend, Module, ScratchArena};

pub use crate::api::GGSWAutomorphism;
use crate::{
    ScratchArenaTakeCore,
    automorphism::glwe_ct::GLWEAutomorphismDefault,
    conversion::GGSWExpandRowsDefault,
    layouts::{
        GGLWEInfos, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GetGaloisElement, ggsw_at_backend_mut_from_mut,
        ggsw_at_backend_ref_from_ref,
        prepared::{GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
    },
};

#[doc(hidden)]
pub trait GGSWAutomorphismDefault<BE: Backend>
where
    Self: GLWEAutomorphismDefault<BE> + GGSWExpandRowsDefault<BE>,
{
    fn ggsw_automorphism_tmp_bytes_default<R, A, K, T>(&self, res_infos: &R, a_infos: &A, key_infos: &K, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        self.glwe_automorphism_tmp_bytes_default(res_infos, a_infos, key_infos)
            .max(self.ggsw_expand_rows_tmp_bytes_default(res_infos, tsk_infos))
    }

    fn ggsw_automorphism_default<'s, R, A, K, T>(&self, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    {
        assert_eq!(res.dsize(), a.dsize());
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.dnum() <= a.dnum());
        assert!(
            scratch.available() >= self.ggsw_automorphism_tmp_bytes_default(res, a, key, tsk),
            "scratch.available(): {} < GGSWAutomorphism::ggsw_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_automorphism_tmp_bytes_default(res, a, key, tsk)
        );

        let mut res_backend = res.to_backend_mut();
        let a_backend = a.to_backend_ref();

        for row in 0..res_backend.dnum().as_usize() {
            self.glwe_automorphism_default(
                &mut ggsw_at_backend_mut_from_mut::<BE>(&mut res_backend, row, 0),
                &ggsw_at_backend_ref_from_ref::<BE>(&a_backend, row, 0),
                key,
                &mut scratch.borrow(),
            );
        }

        self.ggsw_expand_row_default(&mut res_backend, tsk, scratch)
    }

    fn ggsw_automorphism_assign_default<'s, R, K, T>(&self, res: &mut R, key: &K, tsk: &T, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
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

        let mut res_backend = res.to_backend_mut();

        for row in 0..res_backend.dnum().as_usize() {
            self.glwe_automorphism_assign_default(
                &mut ggsw_at_backend_mut_from_mut::<BE>(&mut res_backend, row, 0),
                key,
                &mut scratch.borrow(),
            );
        }

        self.ggsw_expand_row_default(&mut res_backend, tsk, scratch)
    }
}

impl<BE: Backend> GGSWAutomorphismDefault<BE> for Module<BE>
where
    Self: GLWEAutomorphismDefault<BE> + GGSWExpandRowsDefault<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
}
