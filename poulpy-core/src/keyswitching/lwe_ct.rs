use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, DataMut, Module, Scratch, ZnxView, ZnxViewMut, ZnxZero},
};

use crate::{
    LWESampleExtract, ScratchTakeCore,
    keyswitching::glwe_ct::GLWEKeyswitch,
    layouts::{
        GGLWEInfos, GLWE, GLWEAlloc, GLWELayout, LWE, LWEInfos, LWEToMut, LWEToRef, Rank, TorusPrecision,
        prepared::{LWESwitchingKeyPrepared, LWESwitchingKeyPreparedToRef},
    },
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
        K: LWESwitchingKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
        M: LWEKeySwitch<BE>,
    {
        module.lwe_keyswitch(self, a, ksk, scratch);
    }
}

impl<BE: Backend> LWEKeySwitch<BE> for Module<BE> where Self: LWEKeySwitch<BE> {}

pub trait LWEKeySwitch<BE: Backend>
where
    Self: GLWEKeyswitch<BE> + GLWEAlloc + LWESampleExtract,
{
    fn lwe_keyswitch_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        let max_k: TorusPrecision = a_infos.k().max(res_infos.k());

        let glwe_a_infos: GLWELayout = GLWELayout {
            n: self.ring_degree(),
            base2k: a_infos.base2k(),
            k: max_k,
            rank: Rank(1),
        };

        let glwe_res_infos: GLWELayout = GLWELayout {
            n: self.ring_degree(),
            base2k: res_infos.base2k(),
            k: max_k,
            rank: Rank(1),
        };

        let glwe_in: usize = GLWE::bytes_of_from_infos(self, &glwe_a_infos);
        let glwe_out: usize = GLWE::bytes_of_from_infos(self, &glwe_res_infos);
        let ks: usize = self.glwe_keyswitch_tmp_bytes(&glwe_res_infos, &glwe_a_infos, key_infos);

        glwe_in + glwe_out + ks
    }

    fn lwe_keyswitch<R, A, K>(&self, res: &mut R, a: &A, ksk: &K, scratch: &mut Scratch<BE>)
    where
        R: LWEToMut,
        A: LWEToRef,
        K: LWESwitchingKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut LWE<&mut [u8]> = &mut res.to_mut();
        let a: &LWE<&[u8]> = &a.to_ref();
        let ksk: &LWESwitchingKeyPrepared<&[u8], BE> = &ksk.to_ref();

        assert!(res.n().as_usize() <= self.n());
        assert!(a.n().as_usize() <= self.n());
        assert_eq!(ksk.n(), self.n() as u32);
        assert!(scratch.available() >= self.lwe_keyswitch_tmp_bytes(res, a, ksk));

        let max_k: TorusPrecision = res.k().max(a.k());

        let a_size: usize = a.k().div_ceil(ksk.base2k()) as usize;

        let (mut glwe_in, scratch_1) = scratch.take_glwe_ct(
            self,
            &GLWELayout {
                n: ksk.n(),
                base2k: a.base2k(),
                k: max_k,
                rank: Rank(1),
            },
        );
        glwe_in.data.zero();

        let (mut glwe_out, scratch_1) = scratch_1.take_glwe_ct(
            self,
            &GLWELayout {
                n: ksk.n(),
                base2k: res.base2k(),
                k: max_k,
                rank: Rank(1),
            },
        );

        let n_lwe: usize = a.n().into();

        for i in 0..a_size {
            let data_lwe: &[i64] = a.data.at(0, i);
            glwe_in.data.at_mut(0, i)[0] = data_lwe[0];
            glwe_in.data.at_mut(1, i)[..n_lwe].copy_from_slice(&data_lwe[1..]);
        }

        self.glwe_keyswitch(&mut glwe_out, &glwe_in, &ksk.0, scratch_1);
        self.lwe_sample_extract(res, &glwe_out);
    }
}
