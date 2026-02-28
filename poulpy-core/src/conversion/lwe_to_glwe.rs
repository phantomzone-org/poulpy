use poulpy_hal::{
    api::{ScratchAvailable, ScratchTakeBasic, VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, DataMut, Module, Scratch, VecZnx, ZnxView, ZnxViewMut, ZnxZero},
};

use crate::{
    GLWEKeyswitch, ScratchTakeCore,
    layouts::{GGLWEInfos, GGLWEPreparedToRef, GLWE, GLWEInfos, GLWELayout, GLWEToMut, LWE, LWEInfos, LWEToRef},
};

impl<BE: Backend> GLWEFromLWE<BE> for Module<BE>
where
    Self: GLWEKeyswitch<BE> + VecZnxNormalizeTmpBytes + VecZnxNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_from_lwe_tmp_bytes<R, A, K>(&self, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, glwe_infos.n());
        assert_eq!(self.n() as u32, key_infos.n());

        let lvl_0: usize = GLWE::bytes_of(
            self.n().into(),
            key_infos.base2k(),
            lwe_infos.k().max(glwe_infos.k()),
            1u32.into(),
        );

        let lvl_1_ks: usize = self.glwe_keyswitch_tmp_bytes(glwe_infos, glwe_infos, key_infos);
        let lvl_1_a_conv: usize = if lwe_infos.base2k() == key_infos.base2k() {
            0
        } else {
            VecZnx::bytes_of(self.n(), 1, lwe_infos.size()) + self.vec_znx_normalize_tmp_bytes()
        };

        let lvl_1: usize = lvl_1_ks.max(lvl_1_a_conv);

        lvl_0 + lvl_1
    }

    fn glwe_from_lwe<R, A, K>(&self, res: &mut R, lwe: &A, ksk: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: LWEToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let lwe: &LWE<&[u8]> = &lwe.to_ref();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(ksk.n(), self.n() as u32);
        assert!(lwe.n() <= self.n() as u32);
        assert!(
            scratch.available() >= self.glwe_from_lwe_tmp_bytes(res, lwe, ksk),
            "scratch.available(): {} < GLWEFromLWE::glwe_from_lwe_tmp_bytes: {}",
            scratch.available(),
            self.glwe_from_lwe_tmp_bytes(res, lwe, ksk)
        );

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
                &mut glwe.data,
                ksk.base2k().into(),
                0,
                0,
                &a_conv,
                lwe.base2k().into(),
                0,
                scratch_2,
            );

            a_conv.zero();
            for j in 0..lwe.size() {
                let data_lwe: &[i64] = lwe.data.at(0, j);
                a_conv.at_mut(0, j)[..n_lwe].copy_from_slice(&data_lwe[1..]);
            }

            self.vec_znx_normalize(
                &mut glwe.data,
                ksk.base2k().into(),
                0,
                1,
                &a_conv,
                lwe.base2k().into(),
                0,
                scratch_2,
            );
        }

        self.glwe_keyswitch(res, &glwe, ksk, scratch_1);
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

    fn glwe_from_lwe<R, A, K>(&self, res: &mut R, lwe: &A, ksk: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: LWEToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos;
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
