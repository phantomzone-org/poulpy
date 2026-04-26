use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, HostDataMut, ScratchArena},
};

use crate::{
    ScratchArenaTakeCore,
    api::{GGSWExpandRows, LWESampleExtract},
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWE, GLWEBackendMut,
        GLWEBackendRef, GLWEInfos, GLWELayout, LWEInfos, LWEToBackendMut, LWEToBackendRef, Rank, TorusPrecision,
        gglwe_at_backend_mut_from_mut, gglwe_at_backend_ref_from_ref, glwe_backend_ref_from_mut,
        prepared::{GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
    },
};

pub trait GLWEKeyswitch<BE: Backend> {
    fn glwe_keyswitch_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, key_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos;

    fn glwe_keyswitch<'s, 'r, 'a, K>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &GLWEBackendRef<'a, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn glwe_keyswitch_inplace<'s, 'r, K>(&self, res: &mut GLWEBackendMut<'r, BE>, key: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;
}

pub trait GGLWEKeyswitch<BE: Backend>
where
    Self: GLWEKeyswitch<BE>,
{
    fn gglwe_keyswitch_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
    {
        self.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos)
    }

    fn gglwe_keyswitch<'s, R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
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
            scratch.available() >= self.gglwe_keyswitch_tmp_bytes(res, a, b),
            "scratch.available(): {} < GGLWEKeyswitch::gglwe_keyswitch_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_keyswitch_tmp_bytes(res, a, b)
        );

        let mut res = res.to_backend_mut();
        let a = a.to_backend_ref();

        for row in 0..res.dnum().into() {
            for col in 0..res.rank_in().into() {
                self.glwe_keyswitch(
                    &mut gglwe_at_backend_mut_from_mut::<BE>(&mut res, row, col),
                    &gglwe_at_backend_ref_from_ref::<BE>(&a, row, col),
                    b,
                    scratch,
                );
            }
        }
    }

    fn gglwe_keyswitch_inplace<'s, R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
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
            scratch.available() >= self.gglwe_keyswitch_tmp_bytes(&res, &res, a),
            "scratch.available(): {} < GGLWEKeyswitch::gglwe_keyswitch_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_keyswitch_tmp_bytes(&res, &res, a)
        );

        for row in 0..res.dnum().into() {
            for col in 0..res.rank_in().into() {
                self.glwe_keyswitch_inplace(&mut gglwe_at_backend_mut_from_mut::<BE>(&mut res, row, col), a, scratch);
            }
        }
    }
}

pub trait GGSWKeyswitch<BE: Backend>
where
    Self: GLWEKeyswitch<BE> + GGSWExpandRows<BE>,
{
    fn ggsw_keyswitch_tmp_bytes<R, A, K, T>(&self, res_infos: &R, a_infos: &A, key_infos: &K, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos;

    fn ggsw_keyswitch<'s, R, A, K, T>(&self, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;

    fn ggsw_keyswitch_inplace<'s, R, K, T>(&self, res: &mut R, key: &K, tsk: &T, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;
}

pub trait LWEKeySwitch<BE: Backend>
where
    Self: GLWEKeyswitch<BE> + LWESampleExtract<BE> + ModuleN,
{
    fn lwe_keyswitch_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, key_infos.n());

        let max_k: TorusPrecision = a_infos.max_k().max(res_infos.max_k());

        let glwe_a_infos: GLWELayout = GLWELayout {
            n: self.n().into(),
            base2k: a_infos.base2k(),
            k: max_k,
            rank: Rank(1),
        };

        let glwe_res_infos: GLWELayout = GLWELayout {
            n: self.n().into(),
            base2k: res_infos.base2k(),
            k: max_k,
            rank: Rank(1),
        };

        let lvl_0: usize = GLWE::<Vec<u8>>::bytes_of_from_infos(&glwe_a_infos);
        let lvl_1: usize = GLWE::<Vec<u8>>::bytes_of_from_infos(&glwe_res_infos);
        let lvl_2: usize = self.glwe_keyswitch_tmp_bytes(&glwe_res_infos, &glwe_a_infos, key_infos);

        lvl_0 + lvl_1 + lvl_2
    }

    fn lwe_keyswitch<'s, R, A, K>(&self, res: &mut R, a: &A, ksk: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    {
        assert!(res.n().as_usize() <= self.n());
        assert!(a.n().as_usize() <= self.n());
        assert_eq!(ksk.n(), self.n() as u32);
        assert!(
            scratch.available() >= self.lwe_keyswitch_tmp_bytes(res, a, ksk),
            "scratch.available(): {} < LWEKeySwitch::lwe_keyswitch_tmp_bytes: {}",
            scratch.available(),
            self.lwe_keyswitch_tmp_bytes(res, a, ksk)
        );

        let scratch = scratch.borrow();
        let a_backend = a.to_backend_ref();

        let (mut glwe_in, scratch_1) = scratch.take_glwe(&GLWELayout {
            n: ksk.n(),
            base2k: a.base2k(),
            k: a.max_k(),
            rank: Rank(1),
        });
        self.vec_znx_zero_backend(&mut glwe_in.data, 0);
        self.vec_znx_zero_backend(&mut glwe_in.data, 1);

        let n_lwe: usize = a.n().into();

        for i in 0..a.size() {
            self.vec_znx_copy_range_backend(&mut glwe_in.data, 0, i, 0, &a_backend.data, 0, i, 0, 1);
            self.vec_znx_copy_range_backend(&mut glwe_in.data, 1, i, 0, &a_backend.data, 0, i, 1, n_lwe);
        }

        let (mut glwe_out, mut scratch_2) = scratch_1.take_glwe(&GLWELayout {
            n: ksk.n(),
            base2k: res.base2k(),
            k: res.max_k(),
            rank: Rank(1),
        });

        let glwe_in_ref = glwe_backend_ref_from_mut::<BE>(&glwe_in);
        self.glwe_keyswitch(&mut glwe_out, &glwe_in_ref, ksk, &mut scratch_2);
        let glwe_out_ref = glwe_backend_ref_from_mut::<BE>(&glwe_out);
        self.lwe_sample_extract(res, &glwe_out_ref);
    }
}
