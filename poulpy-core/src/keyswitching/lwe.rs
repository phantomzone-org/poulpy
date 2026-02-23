use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, DataMut, Module, Scratch, ZnxView, ZnxViewMut, ZnxZero},
};

use crate::{
    LWESampleExtract, ScratchTakeCore,
    keyswitching::GLWEKeyswitch,
    layouts::{GGLWEInfos, GGLWEPreparedToRef, GLWE, GLWELayout, LWE, LWEInfos, LWEToMut, LWEToRef, Rank, TorusPrecision},
};

impl LWE<Vec<u8>> {
    pub fn keyswitch_tmp_bytes<M, R, A, K, BE: Backend>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
        M: LWEKeySwitch<BE>,
    {
        module.lwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos)
    }
}

impl<D: DataMut> LWE<D> {
    pub fn keyswitch<M, A, K, BE: Backend>(&mut self, module: &M, a: &A, ksk: &K, scratch: &mut Scratch<BE>)
    where
        A: LWEToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
        M: LWEKeySwitch<BE>,
    {
        module.lwe_keyswitch(self, a, ksk, scratch);
    }
}

impl<BE: Backend> LWEKeySwitch<BE> for Module<BE> where Self: GLWEKeyswitch<BE> + LWESampleExtract {}

pub trait LWEKeySwitch<BE: Backend>
where
    Self: GLWEKeyswitch<BE> + LWESampleExtract,
{
    fn lwe_keyswitch_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, key_infos.n());

        let max_k: TorusPrecision = a_infos.k().max(res_infos.k());

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

        let lvl_0: usize = GLWE::bytes_of_from_infos(&glwe_a_infos);
        let lvl_1: usize = GLWE::bytes_of_from_infos(&glwe_res_infos);
        let lvl_2: usize = self.glwe_keyswitch_tmp_bytes(&glwe_res_infos, &glwe_a_infos, key_infos);

        lvl_0 + lvl_1 + lvl_2
    }

    fn lwe_keyswitch<R, A, K>(&self, res: &mut R, a: &A, ksk: &K, scratch: &mut Scratch<BE>)
    where
        R: LWEToMut,
        A: LWEToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut LWE<&mut [u8]> = &mut res.to_mut();
        let a: &LWE<&[u8]> = &a.to_ref();

        assert!(res.n().as_usize() <= self.n());
        assert!(a.n().as_usize() <= self.n());
        assert_eq!(ksk.n(), self.n() as u32);
        assert!(
            scratch.available() >= self.lwe_keyswitch_tmp_bytes(res, a, ksk),
            "scratch.available(): {} < LWEKeySwitch::lwe_keyswitch_tmp_bytes: {}",
            scratch.available(),
            self.lwe_keyswitch_tmp_bytes(res, a, ksk)
        );

        let (mut glwe_in, scratch_1) = scratch.take_glwe(&GLWELayout {
            n: ksk.n(),
            base2k: a.base2k(),
            k: a.k(),
            rank: Rank(1),
        });
        glwe_in.data.zero();

        let n_lwe: usize = a.n().into();

        for i in 0..a.size() {
            let data_lwe: &[i64] = a.data.at(0, i);
            glwe_in.data.at_mut(0, i)[0] = data_lwe[0];
            glwe_in.data.at_mut(1, i)[..n_lwe].copy_from_slice(&data_lwe[1..]);
        }

        let (mut glwe_out, scratch_2) = scratch_1.take_glwe(&GLWELayout {
            n: ksk.n(),
            base2k: res.base2k(),
            k: res.k(),
            rank: Rank(1),
        });

        self.glwe_keyswitch(&mut glwe_out, &glwe_in, ksk, scratch_2);
        self.lwe_sample_extract(res, &glwe_out);
    }
}
