use poulpy_hal::{
    api::{ModuleN, VecZnxCopyRangeBackend, VecZnxZeroBackend},
    layouts::{Backend, Module, ScratchArena},
};

use crate::{
    GLWERotate, ScratchArenaTakeCore,
    default::keyswitching::GLWEKeyswitchDefault,
    layouts::{
        GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWEToBackendRef, LWEInfos, LWEToBackendMut, Rank, glwe_backend_mut_from_mut,
        glwe_backend_ref_from_mut, prepared::GGLWEPreparedToBackendRef,
    },
};

#[doc(hidden)]
pub(crate) trait LWESampleExtractDefault<BE: Backend>
where
    Self: ModuleN + VecZnxCopyRangeBackend<BE> + VecZnxZeroBackend<BE>,
{
    fn lwe_sample_extract_default<R, A>(&self, res: &mut R, a: &A)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        let mut res = res.to_backend_mut();
        let a = a.to_backend_ref();

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

impl<BE: Backend> LWESampleExtractDefault<BE> for Module<BE> where
    Self: ModuleN + VecZnxCopyRangeBackend<BE> + VecZnxZeroBackend<BE>
{
}

#[doc(hidden)]
pub(crate) trait LWEFromGLWEDefault<BE: Backend>
where
    Self: ModuleN + GLWEKeyswitchDefault<BE> + GLWERotate<BE> + VecZnxCopyRangeBackend<BE> + VecZnxZeroBackend<BE>,
{
    fn lwe_from_glwe_tmp_bytes_default<R, A, K>(&self, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
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
        let lvl_1: usize = self.glwe_keyswitch_tmp_bytes_default(&res_infos, glwe_infos, key_infos);
        let lvl_2: usize = GLWE::<Vec<u8>>::bytes_of_from_infos(glwe_infos);

        lvl_0 + lvl_1 + lvl_2
    }

    fn lwe_from_glwe_default<'s, R, A, K>(&self, res: &mut R, a: &A, a_idx: usize, key: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
    {
        let a_backend = a.to_backend_ref();

        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(key.n(), self.n() as u32);
        assert!(res.n() <= self.n() as u32);
        assert!(
            scratch.available() >= self.lwe_from_glwe_tmp_bytes_default(res, a, key),
            "scratch.available(): {} < LWEFromGLWE::lwe_from_glwe_tmp_bytes: {}",
            scratch.available(),
            self.lwe_from_glwe_tmp_bytes_default(res, a, key)
        );

        let glwe_layout: GLWELayout = GLWELayout {
            n: self.n().into(),
            base2k: res.base2k(),
            k: res.max_k(),
            rank: Rank(1),
        };

        let scratch = scratch.borrow();
        let (mut tmp_glwe_rank_1, mut scratch_1) = scratch.take_glwe(&glwe_layout);

        self.glwe_keyswitch_default(&mut tmp_glwe_rank_1, &a_backend, key, &mut scratch_1);
        if a_idx != 0 {
            let mut tmp_glwe_rank_1_backend = glwe_backend_mut_from_mut::<BE>(&mut tmp_glwe_rank_1);
            self.glwe_rotate_assign(-(a_idx as i64), &mut tmp_glwe_rank_1_backend, &mut scratch_1);
        }

        let mut res_backend = res.to_backend_mut();
        let tmp_glwe_rank_1_ref = glwe_backend_ref_from_mut::<BE>(&tmp_glwe_rank_1);
        let min_size: usize = res_backend.size().min(tmp_glwe_rank_1_ref.size());
        let n: usize = res_backend.n().into();

        self.vec_znx_zero_backend(&mut res_backend.data, 0);
        for i in 0..min_size {
            self.vec_znx_copy_range_backend(&mut res_backend.data, 0, i, 0, &tmp_glwe_rank_1_ref.data, 0, i, 0, 1);
            self.vec_znx_copy_range_backend(&mut res_backend.data, 0, i, 1, &tmp_glwe_rank_1_ref.data, 1, i, 0, n);
        }
    }
}

impl<BE: Backend> LWEFromGLWEDefault<BE> for Module<BE>
where
    Self: ModuleN + GLWEKeyswitchDefault<BE> + GLWERotate<BE> + VecZnxCopyRangeBackend<BE> + VecZnxZeroBackend<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
}
