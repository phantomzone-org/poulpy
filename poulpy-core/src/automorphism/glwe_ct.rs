use poulpy_hal::{
    api::{
        ScratchTakeBasic, VecZnxAutomorphismInplace, VecZnxBigAutomorphismInplace, VecZnxBigSubSmallInplace,
        VecZnxBigSubSmallNegateInplace,
    },
    layouts::{Backend, DataMut, Module, Scratch, VecZnxBig},
};

use crate::{
    GLWEKeyswitch, ScratchTakeCore, keyswitch_internal,
    layouts::{
        GGLWEInfos, GGLWEPreparedToRef, GLWE, GLWEInfos, GLWEToMut, GLWEToRef, LWEInfos, prepared::GetAutomorphismGaloisElement,
    },
};

impl GLWE<Vec<u8>> {
    pub fn automorphism_tmp_bytes<M, R, A, K, BE: Backend>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
        M: GLWEAutomorphism<BE>,
    {
        module.glwe_automorphism_tmp_bytes(res_infos, a_infos, key_infos)
    }
}

impl<DataSelf: DataMut> GLWE<DataSelf> {
    pub fn automorphism<M, A, K, BE: Backend>(&mut self, module: &M, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        M: GLWEAutomorphism<BE>,
        A: GLWEToRef,
        K: GetAutomorphismGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_automorphism(self, a, key, scratch);
    }

    pub fn automorphism_add<M, A, K, BE: Backend>(&mut self, module: &M, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        M: GLWEAutomorphism<BE>,
        A: GLWEToRef,
        K: GetAutomorphismGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_automorphism_add(self, a, key, scratch);
    }

    pub fn automorphism_sub<M, A, K, BE: Backend>(&mut self, module: &M, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        M: GLWEAutomorphism<BE>,
        A: GLWEToRef,
        K: GetAutomorphismGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_automorphism_sub(self, a, key, scratch);
    }

    pub fn glwe_automorphism_sub_negate<M, A, K, BE: Backend>(&mut self, module: &M, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        M: GLWEAutomorphism<BE>,
        A: GLWEToRef,
        K: GetAutomorphismGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_automorphism_sub_negate(self, a, key, scratch);
    }

    pub fn automorphism_inplace<M, K, BE: Backend>(&mut self, module: &M, key: &K, scratch: &mut Scratch<BE>)
    where
        M: GLWEAutomorphism<BE>,
        K: GetAutomorphismGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_automorphism_inplace(self, key, scratch);
    }

    pub fn automorphism_add_inplace<M, K, BE: Backend>(&mut self, module: &M, key: &K, scratch: &mut Scratch<BE>)
    where
        M: GLWEAutomorphism<BE>,
        K: GetAutomorphismGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_automorphism_add_inplace(self, key, scratch);
    }

    pub fn automorphism_sub_inplace<M, K, BE: Backend>(&mut self, module: &M, key: &K, scratch: &mut Scratch<BE>)
    where
        M: GLWEAutomorphism<BE>,
        K: GetAutomorphismGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_automorphism_sub_inplace(self, key, scratch);
    }

    pub fn automorphism_sub_negate_inplace<M, K, BE: Backend>(&mut self, module: &M, key: &K, scratch: &mut Scratch<BE>)
    where
        M: GLWEAutomorphism<BE>,
        K: GetAutomorphismGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_automorphism_sub_negate_inplace(self, key, scratch);
    }
}

pub trait GLWEAutomorphism<BE: Backend>
where
    Self: GLWEKeyswitch<BE>
        + VecZnxAutomorphismInplace<BE>
        + VecZnxBigAutomorphismInplace<BE>
        + VecZnxBigSubSmallInplace<BE>
        + VecZnxBigSubSmallNegateInplace<BE>,
{
    fn glwe_automorphism_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        self.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos)
    }

    fn glwe_automorphism<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        K: GetAutomorphismGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.glwe_keyswitch(res, a, key, scratch);

        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_automorphism_inplace(key.p(), res.data_mut(), i, scratch);
        }
    }

    fn glwe_automorphism_inplace<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GetAutomorphismGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.glwe_keyswitch_inplace(res, key, scratch);

        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_automorphism_inplace(key.p(), res.data_mut(), i, scratch);
        }
    }

    fn glwe_automorphism_add<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        K: GetAutomorphismGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // TODO: optimise size
        let mut res_big: VecZnxBig<_, BE> = keyswitch_internal(self, res_dft, a, key, scratch_1);

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch_1);
            self.vec_znx_big_add_small_inplace(&mut res_big, i, a.data(), i);
            self.vec_znx_big_normalize(
                res.base2k().into(),
                res.data_mut(),
                i,
                key.base2k().into(),
                &res_big,
                i,
                scratch_1,
            );
        }
    }

    fn glwe_automorphism_add_inplace<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GetAutomorphismGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // TODO: optimise size
        let mut res_big: VecZnxBig<_, BE> = keyswitch_internal(self, res_dft, res, key, scratch_1);

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch_1);
            self.vec_znx_big_add_small_inplace(&mut res_big, i, res.data(), i);
            self.vec_znx_big_normalize(
                res.base2k().into(),
                res.data_mut(),
                i,
                key.base2k().into(),
                &res_big,
                i,
                scratch_1,
            );
        }
    }

    fn glwe_automorphism_sub<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        K: GetAutomorphismGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // TODO: optimise size
        let mut res_big: VecZnxBig<_, BE> = keyswitch_internal(self, res_dft, a, key, scratch_1);

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch_1);
            self.vec_znx_big_sub_small_inplace(&mut res_big, i, a.data(), i);
            self.vec_znx_big_normalize(
                res.base2k().into(),
                res.data_mut(),
                i,
                key.base2k().into(),
                &res_big,
                i,
                scratch_1,
            );
        }
    }

    fn glwe_automorphism_sub_negate<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        K: GetAutomorphismGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // TODO: optimise size
        let mut res_big: VecZnxBig<_, BE> = keyswitch_internal(self, res_dft, a, key, scratch_1);

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch_1);
            self.vec_znx_big_sub_small_negate_inplace(&mut res_big, i, a.data(), i);
            self.vec_znx_big_normalize(
                res.base2k().into(),
                res.data_mut(),
                i,
                key.base2k().into(),
                &res_big,
                i,
                scratch_1,
            );
        }
    }

    fn glwe_automorphism_sub_inplace<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GetAutomorphismGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // TODO: optimise size
        let mut res_big: VecZnxBig<_, BE> = keyswitch_internal(self, res_dft, res, key, scratch_1);

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch_1);
            self.vec_znx_big_sub_small_inplace(&mut res_big, i, res.data(), i);
            self.vec_znx_big_normalize(
                res.base2k().into(),
                res.data_mut(),
                i,
                key.base2k().into(),
                &res_big,
                i,
                scratch_1,
            );
        }
    }

    fn glwe_automorphism_sub_negate_inplace<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GetAutomorphismGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // TODO: optimise size
        let mut res_big: VecZnxBig<_, BE> = keyswitch_internal(self, res_dft, res, key, scratch_1);

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch_1);
            self.vec_znx_big_sub_small_negate_inplace(&mut res_big, i, res.data(), i);
            self.vec_znx_big_normalize(
                res.base2k().into(),
                res.data_mut(),
                i,
                key.base2k().into(),
                &res_big,
                i,
                scratch_1,
            );
        }
    }
}

impl<BE: Backend> GLWEAutomorphism<BE> for Module<BE> where
    Self: GLWEKeyswitch<BE>
        + VecZnxAutomorphismInplace<BE>
        + VecZnxBigAutomorphismInplace<BE>
        + VecZnxBigSubSmallInplace<BE>
        + VecZnxBigSubSmallNegateInplace<BE>
{
}
