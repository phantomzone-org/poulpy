use poulpy_hal::layouts::{Backend, HostDataMut, Module, ScratchArena};

pub use crate::api::GGLWEKeyswitch;
use crate::{
    ScratchArenaTakeCore,
    keyswitching::GLWEKeyswitch,
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGLWEToMut, GGLWEToRef, GLWESwitchingKey,
        gglwe_at_backend_mut_from_mut, gglwe_at_backend_ref_from_ref, prepared::GGLWEPreparedToBackendRef,
    },
};

impl<DataSelf: HostDataMut> GLWESwitchingKey<DataSelf> {}

#[doc(hidden)]
pub trait GGLWEKeyswitchDefault<BE: Backend>
where
    Self: GLWEKeyswitch<BE>,
{
    fn gglwe_keyswitch_tmp_bytes_default<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
    {
        self.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos)
    }

    fn gglwe_keyswitch_default<'s, R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToMut + GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToRef + GGLWEToBackendRef<BE> + GGLWEInfos,
        B: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        assert_eq!(
            res.rank_in(),
            a.rank_in(),
            "res input rank: {} != a input rank: {}",
            res.rank_in(),
            a.rank_in()
        );
        assert_eq!(
            a.rank_out(),
            b.rank_in(),
            "res output rank: {} != b input rank: {}",
            a.rank_out(),
            b.rank_in()
        );
        assert_eq!(
            res.rank_out(),
            b.rank_out(),
            "res output rank: {} != b output rank: {}",
            res.rank_out(),
            b.rank_out()
        );
        assert!(res.dnum() <= a.dnum(), "res.dnum()={} > a.dnum()={}", res.dnum(), a.dnum());
        assert_eq!(res.dsize(), a.dsize(), "res dsize: {} != a dsize: {}", res.dsize(), a.dsize());
        assert_eq!(res.base2k(), a.base2k());
        assert!(
            scratch.available() >= self.gglwe_keyswitch_tmp_bytes_default(res, a, b),
            "scratch.available(): {} < GGLWEKeyswitch::gglwe_keyswitch_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_keyswitch_tmp_bytes_default(res, a, b)
        );

        let mut res = res.to_backend_mut();
        let a = a.to_backend_ref();

        for row in 0..res.dnum().into() {
            for col in 0..res.rank_in().into() {
                self.glwe_keyswitch(
                    &mut gglwe_at_backend_mut_from_mut::<BE>(&mut res, row, col),
                    &gglwe_at_backend_ref_from_ref::<BE>(&a, row, col),
                    b,
                    &mut scratch.borrow(),
                );
            }
        }
    }

    fn gglwe_keyswitch_inplace_default<'s, R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToMut + GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        let mut res = res.to_backend_mut();

        assert_eq!(
            res.rank_out(),
            a.rank_out(),
            "res output rank: {} != a output rank: {}",
            res.rank_out(),
            a.rank_out()
        );
        assert!(
            scratch.available() >= self.gglwe_keyswitch_tmp_bytes_default(&res, &res, a),
            "scratch.available(): {} < GGLWEKeyswitch::gglwe_keyswitch_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_keyswitch_tmp_bytes_default(&res, &res, a)
        );

        for row in 0..res.dnum().into() {
            for col in 0..res.rank_in().into() {
                self.glwe_keyswitch_inplace(
                    &mut gglwe_at_backend_mut_from_mut::<BE>(&mut res, row, col),
                    a,
                    &mut scratch.borrow(),
                );
            }
        }
    }
}

impl<BE: Backend> GGLWEKeyswitchDefault<BE> for Module<BE> where Self: GLWEKeyswitch<BE> {}
