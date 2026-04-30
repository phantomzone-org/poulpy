use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, Module, ScratchArena},
};

pub use crate::api::GGSWKeyswitch;
use crate::{
    GGSWExpandRows, ScratchArenaTakeCore,
    keyswitching::GLWEKeyswitch,
    layouts::{
        GGLWEInfos, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, LWEInfos, ggsw_at_backend_mut_from_mut,
        ggsw_at_backend_ref_from_ref,
        prepared::{GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
    },
};

#[doc(hidden)]
pub trait GGSWKeyswitchDefault<BE: Backend>
where
    Self: ModuleN + GLWEKeyswitch<BE> + GGSWExpandRows<BE>,
{
    fn ggsw_keyswitch_tmp_bytes_default<R, A, K, T>(&self, res_infos: &R, a_infos: &A, key_infos: &K, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        assert_eq!(key_infos.rank_in(), key_infos.rank_out());
        assert_eq!(tsk_infos.rank_in(), tsk_infos.rank_out());
        assert_eq!(key_infos.rank_in(), tsk_infos.rank_in());
        assert_eq!(self.n() as u32, res_infos.n());
        assert_eq!(self.n() as u32, a_infos.n());
        assert_eq!(self.n() as u32, key_infos.n());
        assert_eq!(self.n() as u32, tsk_infos.n());

        self.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos)
            .max(self.ggsw_expand_rows_tmp_bytes(res_infos, tsk_infos))
    }

    fn ggsw_keyswitch_assign_default<'s, R, K, T>(&self, res: &mut R, key: &K, tsk: &T, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    {
        let mut res = res.to_backend_mut();
        assert!(
            scratch.available() >= self.ggsw_keyswitch_tmp_bytes_default(&res, &res, key, tsk),
            "scratch.available(): {} < GGSWKeyswitch::ggsw_keyswitch_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_keyswitch_tmp_bytes_default(&res, &res, key, tsk)
        );

        for row in 0..res.dnum().into() {
            self.glwe_keyswitch_assign(
                &mut ggsw_at_backend_mut_from_mut::<BE>(&mut res, row, 0),
                key,
                &mut scratch.borrow(),
            );
        }

        self.ggsw_expand_row(&mut res, tsk, scratch)
    }

    fn ggsw_keyswitch_default<'s, R, A, K, T>(&self, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    {
        let mut res = res.to_backend_mut();
        let a = a.to_backend_ref();

        assert!(res.dnum() <= a.dnum());
        assert_eq!(res.dsize(), a.dsize());
        assert_eq!(res.base2k(), a.base2k());
        assert!(
            scratch.available() >= self.ggsw_keyswitch_tmp_bytes_default(&res, &a, key, tsk),
            "scratch.available(): {} < GGSWKeyswitch::ggsw_keyswitch_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_keyswitch_tmp_bytes_default(&res, &a, key, tsk)
        );

        for row in 0..a.dnum().into() {
            self.glwe_keyswitch(
                &mut ggsw_at_backend_mut_from_mut::<BE>(&mut res, row, 0),
                &ggsw_at_backend_ref_from_ref::<BE>(&a, row, 0),
                key,
                &mut scratch.borrow(),
            );
        }

        self.ggsw_expand_row(&mut res, tsk, scratch)
    }
}

impl<BE: Backend> GGSWKeyswitchDefault<BE> for Module<BE>
where
    Self: ModuleN + GLWEKeyswitch<BE> + GGSWExpandRows<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
}
