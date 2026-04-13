use poulpy_hal::{
    api::{ModuleN, ScratchAvailable},
    layouts::{Backend, DataMut, Module, Scratch},
};

pub use crate::api::{LWEFromGLWE, LWESampleExtract};
use crate::{
    GLWEKeyswitch, GLWERotate, ScratchTakeCore,
    layouts::{GGLWEInfos, GGLWEPreparedToRef, GLWE, GLWEInfos, GLWELayout, GLWEToRef, LWE, LWEInfos, LWEToMut, Rank},
};

impl<BE: Backend> LWESampleExtract for Module<BE> where Self: ModuleN {}

#[doc(hidden)]
pub trait LWEFromGLWEDefault<BE: Backend>
where
    Self: GLWEKeyswitch<BE> + LWESampleExtract + GLWERotate<BE>,
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

        let lvl_0: usize = GLWE::<Vec<u8>, ()>::bytes_of(self.n().into(), lwe_infos.base2k(), lwe_infos.max_k(), 1u32.into());
        let lvl_1: usize = self.glwe_keyswitch_tmp_bytes(&res_infos, glwe_infos, key_infos);
        let lvl_2: usize = GLWE::<Vec<u8>, ()>::bytes_of_from_infos(glwe_infos);

        lvl_0 + lvl_1 + lvl_2
    }

    fn lwe_from_glwe_default<R, A, K>(&self, res: &mut R, a: &A, a_idx: usize, key: &K, scratch: &mut Scratch<BE>)
    where
        R: LWEToMut,
        A: GLWEToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut LWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

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

        let (mut tmp_glwe_rank_1, scratch_1) = scratch.take_glwe(&glwe_layout);

        match a_idx {
            0 => {
                self.glwe_keyswitch(&mut tmp_glwe_rank_1, a, key, scratch_1);
            }
            _ => {
                let (mut tmp_glwe_in, scratch_2) = scratch_1.take_glwe(a);
                self.glwe_rotate(-(a_idx as i64), &mut tmp_glwe_in, a);
                self.glwe_keyswitch(&mut tmp_glwe_rank_1, &tmp_glwe_in, key, scratch_2);
            }
        }

        self.lwe_sample_extract(res, &tmp_glwe_rank_1);
    }
}

impl<BE: Backend> LWEFromGLWEDefault<BE> for Module<BE> where Self: GLWEKeyswitch<BE> + LWESampleExtract + GLWERotate<BE> {}

impl LWE<Vec<u8>> {
    pub fn from_glwe_tmp_bytes<R, A, K, M, BE: Backend>(module: &M, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
        M: LWEFromGLWE<BE>,
    {
        module.lwe_from_glwe_tmp_bytes(lwe_infos, glwe_infos, key_infos)
    }
}

impl<D: DataMut> LWE<D> {
    pub fn sample_extract<A, M>(&mut self, module: &M, a: &A)
    where
        A: GLWEToRef,
        M: LWESampleExtract,
    {
        module.lwe_sample_extract(self, a);
    }

    pub fn from_glwe<A, K, M, BE: Backend>(&mut self, module: &M, a: &A, a_idx: usize, key: &K, scratch: &mut Scratch<BE>)
    where
        A: GLWEToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        M: LWEFromGLWE<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.lwe_from_glwe(self, a, a_idx, key, scratch);
    }
}
