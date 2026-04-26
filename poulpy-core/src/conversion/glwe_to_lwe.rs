use poulpy_hal::{
    api::{ModuleN, VecZnxCopyRangeBackend, VecZnxZeroBackend},
    layouts::{Backend, Module, ScratchArena},
};

pub use crate::api::{LWEFromGLWE, LWESampleExtract};
use crate::{
    GLWEKeyswitch, GLWERotate, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWEToBackendRef, LWEInfos, LWEToBackendMut, Rank, glwe_backend_mut_from_mut,
        glwe_backend_ref_from_mut, prepared::GGLWEPreparedToBackendRef,
    },
};

impl<BE: Backend> LWESampleExtract<BE> for Module<BE> where Self: ModuleN + VecZnxCopyRangeBackend<BE> + VecZnxZeroBackend<BE> {}

#[doc(hidden)]
pub trait LWEFromGLWEDefault<BE: Backend>
where
    Self: GLWEKeyswitch<BE> + LWESampleExtract<BE> + GLWERotate<BE>,
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
        let lvl_1: usize = self.glwe_keyswitch_tmp_bytes(&res_infos, glwe_infos, key_infos);
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

        self.glwe_keyswitch(&mut tmp_glwe_rank_1, &a_backend, key, &mut scratch_1);
        if a_idx != 0 {
            let mut tmp_glwe_rank_1_backend = glwe_backend_mut_from_mut::<BE>(&mut tmp_glwe_rank_1);
            self.glwe_rotate_inplace(-(a_idx as i64), &mut tmp_glwe_rank_1_backend, &mut scratch_1);
        }

        let tmp_glwe_rank_1_ref = glwe_backend_ref_from_mut::<BE>(&tmp_glwe_rank_1);
        self.lwe_sample_extract(res, &tmp_glwe_rank_1_ref);
    }
}

impl<BE: Backend> LWEFromGLWEDefault<BE> for Module<BE>
where
    Self: GLWEKeyswitch<BE> + LWESampleExtract<BE> + GLWERotate<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
}
