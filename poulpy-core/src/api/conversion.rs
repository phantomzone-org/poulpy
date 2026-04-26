use poulpy_hal::{
    api::{ModuleN, VecZnxCopyRangeBackend, VecZnxZeroBackend},
    layouts::{Backend, ScratchArena},
};

use crate::{
    ScratchArenaTakeCore,
    api::{GLWEKeyswitch, GLWERotate},
    layouts::{
        GGLWEInfos, GGLWEToBackendRef, GGSWBackendMut, GGSWInfos, GGSWToBackendMut, GLWE, GLWEBackendRef, GLWEInfos,
        GLWELayout, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, LWEToBackendMut, LWEToBackendRef, Rank,
        glwe_backend_mut_from_mut,
        prepared::{GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
    },
};

pub trait LWESampleExtract<BE: Backend>
where
    Self: ModuleN + VecZnxCopyRangeBackend<BE> + VecZnxZeroBackend<BE>,
{
    fn lwe_sample_extract<R>(&self, res: &mut R, a: &GLWEBackendRef<'_, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
    {
        let mut res = res.to_backend_mut();

        assert!(res.n() <= a.n());
        assert_eq!(a.n(), self.n() as u32);
        assert!(res.base2k() == a.base2k());

        let min_size: usize = res.size().min(a.size());
        let n: usize = res.n().into();

        self.vec_znx_zero_backend(&mut res.data, 0);
        (0..min_size).for_each(|i| {
            self.vec_znx_copy_range_backend(&mut res.data, 0, i, 0, &a.data, 0, i, 0, 1);
            self.vec_znx_copy_range_backend(&mut res.data, 0, i, 1, &a.data, 1, i, 0, n);
        });
    }
}

pub trait GLWEFromLWE<BE: Backend>
where
    Self: GLWEKeyswitch<BE>,
{
    fn glwe_from_lwe_tmp_bytes<R, A, K>(&self, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn glwe_from_lwe<'s, R, A, K>(&self, res: &mut R, lwe: &A, ksk: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;
}

pub trait LWEFromGLWE<BE: Backend>
where
    Self: GLWEKeyswitch<BE> + LWESampleExtract<BE> + GLWERotate<BE>,
{
    fn lwe_from_glwe_tmp_bytes<R, A, K>(&self, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, glwe_infos.n());
        assert_eq!(self.n() as u32, key_infos.n());

        let res_infos: GLWELayout = GLWELayout {
            n: self.n().into(),
            base2k: lwe_infos.base2k(),
            k: lwe_infos.max_k(),
            rank: Rank(1),
        };

        let lvl_0: usize = GLWE::<Vec<u8>>::bytes_of(self.n().into(), lwe_infos.base2k(), lwe_infos.max_k(), 1u32.into());
        let lvl_1: usize = self.glwe_keyswitch_tmp_bytes(&res_infos, glwe_infos, key_infos);
        let lvl_2: usize = GLWE::<Vec<u8>>::bytes_of_from_infos(glwe_infos);

        lvl_0 + lvl_1 + lvl_2
    }

    fn lwe_from_glwe<'s, R, A, K>(&self, res: &mut R, a: &A, a_idx: usize, key: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        BE: 's,
    {
        let a_backend = a.to_backend_ref();

        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(key.n(), self.n() as u32);
        assert!(res.n() <= self.n() as u32);
        assert!(
            scratch.available() >= self.lwe_from_glwe_tmp_bytes(res, a, key),
            "scratch.available(): {} < LWEFromGLWE::lwe_from_glwe_tmp_bytes: {}",
            scratch.available(),
            self.lwe_from_glwe_tmp_bytes(res, a, key)
        );

        let glwe_layout: GLWELayout = GLWELayout {
            n: self.n().into(),
            base2k: res.base2k(),
            k: res.max_k(),
            rank: Rank(1),
        };

        let (mut tmp_glwe_rank_1, mut scratch_1) = scratch.borrow().take_glwe(&glwe_layout);

        self.glwe_keyswitch(&mut tmp_glwe_rank_1, &a_backend, key, &mut scratch_1);
        if a_idx != 0 {
            let mut tmp_glwe_rank_1_backend = glwe_backend_mut_from_mut::<BE>(&mut tmp_glwe_rank_1);
            self.glwe_rotate_inplace(-(a_idx as i64), &mut tmp_glwe_rank_1_backend, &mut scratch_1);
        }

        let tmp_glwe_rank_1_ref = crate::layouts::glwe_backend_ref_from_mut::<BE>(&tmp_glwe_rank_1);
        self.lwe_sample_extract(res, &tmp_glwe_rank_1_ref);
    }
}

pub trait GGSWFromGGLWE<BE: Backend> {
    fn ggsw_from_gglwe_tmp_bytes<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_from_gglwe<'s, R, A, T>(&self, res: &mut R, a: &A, tsk: &T, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;
}

pub trait GGSWExpandRows<BE: Backend> {
    fn ggsw_expand_rows_tmp_bytes<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_expand_row<'s, 'r, T>(&self, res: &mut GGSWBackendMut<'r, BE>, tsk: &T, scratch: &mut ScratchArena<'s, BE>)
    where
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;
}
