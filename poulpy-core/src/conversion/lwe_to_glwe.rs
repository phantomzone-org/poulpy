use poulpy_hal::{
    api::ScratchTakeBasic,
    layouts::{Backend, DataMut, Module, Scratch, VecZnx, ZnxView, ZnxViewMut, ZnxZero},
};

use crate::{
    GLWEKeyswitch, ScratchTakeCore,
    layouts::{GGLWEInfos, GGLWEPreparedToRef, GLWE, GLWEInfos, GLWELayout, GLWEToMut, LWE, LWEInfos, LWEToRef},
};

impl<BE: Backend> GLWEFromLWE<BE> for Module<BE> where Self: GLWEKeyswitch<BE> {}

pub trait GLWEFromLWE<BE: Backend>
where
    Self: GLWEKeyswitch<BE>,
{
    fn glwe_from_lwe_tmp_bytes<R, A, K>(&self, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        let ct: usize = GLWE::bytes_of(
            self.n().into(),
            key_infos.base2k(),
            lwe_infos.k().max(glwe_infos.k()),
            1u32.into(),
        );

        let ks: usize = self.glwe_keyswitch_tmp_bytes(glwe_infos, glwe_infos, key_infos);
        if lwe_infos.base2k() == key_infos.base2k() {
            ct + ks
        } else {
            let a_conv = VecZnx::bytes_of(self.n(), 1, lwe_infos.size()) + self.vec_znx_normalize_tmp_bytes();
            ct + a_conv + ks
        }
    }

    fn glwe_from_lwe<R, A, K>(&self, res: &mut R, lwe: &A, ksk: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: LWEToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let lwe: &LWE<&[u8]> = &lwe.to_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(ksk.n(), self.n() as u32);
        assert!(lwe.n() <= self.n() as u32);

        let (mut glwe, scratch_1) = scratch.take_glwe(&GLWELayout {
            n: ksk.n(),
            base2k: ksk.base2k(),
            k: lwe.k(),
            rank: 1u32.into(),
        });
        glwe.data.zero();

        let n_lwe: usize = lwe.n().into();

        if lwe.base2k() == ksk.base2k() {
            for i in 0..lwe.size() {
                let data_lwe: &[i64] = lwe.data.at(0, i);
                glwe.data.at_mut(0, i)[0] = data_lwe[0];
                glwe.data.at_mut(1, i)[..n_lwe].copy_from_slice(&data_lwe[1..]);
            }
        } else {
            let (mut a_conv, scratch_2) = scratch_1.take_vec_znx(self.n(), 1, lwe.size());
            a_conv.zero();
            for j in 0..lwe.size() {
                let data_lwe: &[i64] = lwe.data.at(0, j);
                a_conv.at_mut(0, j)[0] = data_lwe[0]
            }

            self.vec_znx_normalize(
                ksk.base2k().into(),
                &mut glwe.data,
                0,
                lwe.base2k().into(),
                &a_conv,
                0,
                scratch_2,
            );

            a_conv.zero();
            for j in 0..lwe.size() {
                let data_lwe: &[i64] = lwe.data.at(0, j);
                a_conv.at_mut(0, j)[..n_lwe].copy_from_slice(&data_lwe[1..]);
            }

            self.vec_znx_normalize(
                ksk.base2k().into(),
                &mut glwe.data,
                1,
                lwe.base2k().into(),
                &a_conv,
                0,
                scratch_2,
            );
        }

        self.glwe_keyswitch(res, &glwe, ksk, scratch_1);
    }
}

impl GLWE<Vec<u8>> {
    pub fn from_lwe_tmp_bytes<R, A, K, M, BE: Backend>(module: &M, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
        M: GLWEFromLWE<BE>,
    {
        module.glwe_from_lwe_tmp_bytes(glwe_infos, lwe_infos, key_infos)
    }
}

impl<D: DataMut> GLWE<D> {
    pub fn from_lwe<A, K, M, BE: Backend>(&mut self, module: &M, lwe: &A, ksk: &K, scratch: &mut Scratch<BE>)
    where
        M: GLWEFromLWE<BE>,
        A: LWEToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_from_lwe(self, lwe, ksk, scratch);
    }
}
