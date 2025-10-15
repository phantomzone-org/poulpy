use poulpy_hal::{
    api::{
        ScratchAvailable, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftApply,
        VecZnxDftBytesOf, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxView, ZnxViewMut, ZnxZero},
};

use crate::{
    keyswitching::glwe_ct::GLWEKeySwitch,
    layouts::{prepared::LWESwitchingKeyPrepared, GGLWEInfos, GLWEAlloc, GLWELayout, GetDegree, LWEToRef, LWEInfos, Rank, TorusPrecision, GLWE, LWE},
};

pub trait LWEKeySwitch<BE: Backend>
where
    Self: GLWEKeySwitch<BE> + GLWEAlloc,
{
    fn keyswitch_tmp_bytes<B: Backend, R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        let max_k: TorusPrecision = a_infos.k().max(res_infos.k());

        let glwe_a_infos: GLWELayout = GLWELayout {
            n: GetDegree::n(self),
            base2k: a_infos.base2k(),
            k: max_k,
            rank: Rank(1),
        };

        let glwe_res_infos: GLWELayout = GLWELayout {
            n: GetDegree::n(self),
            base2k: res_infos.base2k(),
            k: max_k,
            rank: Rank(1),
        };

        let glwe_in: usize = GLWE::bytes_of_from_infos(self, &glwe_a_infos);
        let glwe_out: usize = GLWE::bytes_of_from_infos(self, &glwe_res_infos);
        let ks: usize = self.glwe_keyswitch_tmp_bytes(&glwe_res_infos, &glwe_a_infos, key_infos);

        glwe_in + glwe_out + ks
    }

    fn keyswitch<A, DKs, B: Backend>(
        &mut self,
        module: &Module<B>,
        a: &A,
        ksk: &K,
        scratch: &mut Scratch<B>,
    ) where
        A: LWEToRef,
        DKs: DataRef,
        Scratch<B>: ScratchAvailable,
    {
        assert!(self.n() <= module.n() as u32);
            assert!(a.n() <= module.n() as u32);
            assert!(scratch.available() >= LWE::keyswitch_tmp_bytes(module, self, a, ksk));

        let max_k: TorusPrecision = self.k().max(a.k());

        let a_size: usize = a.k().div_ceil(ksk.base2k()) as usize;

        let (mut glwe_in, scratch_1) = scratch.take_glwe_ct(&GLWELayout {
            n: ksk.n(),
            base2k: a.base2k(),
            k: max_k,
            rank: Rank(1),
        });
        glwe_in.data.zero();

        let (mut glwe_out, scratch_1) = scratch_1.take_glwe_ct(&GLWELayout {
            n: ksk.n(),
            base2k: self.base2k(),
            k: max_k,
            rank: Rank(1),
        });

        let n_lwe: usize = a.n().into();

        for i in 0..a_size {
            let data_lwe: &[i64] = a.data.at(0, i);
            glwe_in.data.at_mut(0, i)[0] = data_lwe[0];
            glwe_in.data.at_mut(1, i)[..n_lwe].copy_from_slice(&data_lwe[1..]);
        }

        glwe_out.keyswitch(module, &glwe_in, &ksk.0, scratch_1);
        self.sample_extract(&glwe_out);
    }
}

impl LWE<Vec<u8>> {}

impl<DLwe: DataMut> LWE<DLwe> {
 
}
