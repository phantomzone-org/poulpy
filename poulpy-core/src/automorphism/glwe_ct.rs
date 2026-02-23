use poulpy_hal::{
    api::{
        ScratchAvailable, ScratchTakeBasic, VecZnxAutomorphismInplace, VecZnxAutomorphismInplaceTmpBytes,
        VecZnxBigAddSmallInplace, VecZnxBigAutomorphismInplace, VecZnxBigAutomorphismInplaceTmpBytes, VecZnxBigNormalize,
        VecZnxBigSubSmallInplace, VecZnxBigSubSmallNegateInplace, VecZnxNormalize,
    },
    layouts::{Backend, DataMut, Module, Scratch, VecZnxBig},
};

use crate::{
    GLWEKeySwitchInternal, GLWEKeyswitch, GLWENormalize, ScratchTakeCore,
    layouts::{GGLWEInfos, GGLWEPreparedToRef, GLWE, GLWEInfos, GLWELayout, GLWEToMut, GLWEToRef, GetGaloisElement, LWEInfos},
};

impl GLWE<Vec<u8>> {
    /// Returns the scratch buffer size in bytes required by the GLWE automorphism operation.
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
    /// Applies the Galois automorphism X -> X^k to a GLWE ciphertext `a` using the
    /// automorphism key `key`, writing the result into `self`.
    ///
    /// Internally performs a key-switch followed by the polynomial automorphism on each
    /// component of the resulting GLWE ciphertext.
    pub fn automorphism<M, A, K, BE: Backend>(&mut self, module: &M, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        M: GLWEAutomorphism<BE>,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_automorphism(self, a, key, scratch);
    }

    /// Computes `self = self + Automorphism(a, key)`.
    ///
    /// Applies the automorphism to `a` and adds the result to the current value of `self`,
    /// fusing the key-switch, automorphism, and addition into a single pass over the
    /// extended-precision accumulator to reduce normalization overhead.
    pub fn automorphism_add<M, A, K, BE: Backend>(&mut self, module: &M, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        M: GLWEAutomorphism<BE>,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_automorphism_add(self, a, key, scratch);
    }

    /// Computes `self = Automorphism(a, key) - a`.
    ///
    /// Applies the automorphism to `a` and subtracts `a` from the result,
    /// fusing the key-switch, automorphism, and subtraction into a single pass over the
    /// extended-precision accumulator to reduce normalization overhead.
    pub fn automorphism_sub<M, A, K, BE: Backend>(&mut self, module: &M, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        M: GLWEAutomorphism<BE>,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_automorphism_sub(self, a, key, scratch);
    }

    /// Computes `self = -(Automorphism(a, key) - a)` i.e. `self = a - Automorphism(a, key)`.
    ///
    /// Applies the automorphism to `a`, subtracts `a`, and negates the result,
    /// fusing all operations into a single pass over the extended-precision accumulator.
    pub fn automorphism_sub_negate<M, A, K, BE: Backend>(&mut self, module: &M, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        M: GLWEAutomorphism<BE>,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_automorphism_sub_negate(self, a, key, scratch);
    }

    /// Applies the Galois automorphism X -> X^k to `self` in place using the
    /// automorphism key `key`.
    ///
    /// Internally performs a key-switch followed by the polynomial automorphism on each
    /// component of the GLWE ciphertext.
    pub fn automorphism_inplace<M, K, BE: Backend>(&mut self, module: &M, key: &K, scratch: &mut Scratch<BE>)
    where
        M: GLWEAutomorphism<BE>,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_automorphism_inplace(self, key, scratch);
    }

    /// Computes `self = self + Automorphism(self, key)` in place.
    ///
    /// Applies the automorphism to the current value of `self` and adds the result back,
    /// fusing the key-switch, automorphism, and addition into a single pass.
    pub fn automorphism_add_inplace<M, K, BE: Backend>(&mut self, module: &M, key: &K, scratch: &mut Scratch<BE>)
    where
        M: GLWEAutomorphism<BE>,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_automorphism_add_inplace(self, key, scratch);
    }

    /// Computes `self = Automorphism(self, key) - self` in place.
    ///
    /// Applies the automorphism to the current value of `self` and subtracts the original,
    /// fusing the key-switch, automorphism, and subtraction into a single pass.
    pub fn automorphism_sub_inplace<M, K, BE: Backend>(&mut self, module: &M, key: &K, scratch: &mut Scratch<BE>)
    where
        M: GLWEAutomorphism<BE>,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_automorphism_sub_inplace(self, key, scratch);
    }

    /// Computes `self = -(Automorphism(self, key) - self)` i.e. `self = self - Automorphism(self, key)` in place.
    ///
    /// Applies the automorphism to the current value of `self`, subtracts the original,
    /// and negates, fusing all operations into a single pass.
    pub fn automorphism_sub_negate_inplace<M, K, BE: Backend>(&mut self, module: &M, key: &K, scratch: &mut Scratch<BE>)
    where
        M: GLWEAutomorphism<BE>,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_automorphism_sub_negate_inplace(self, key, scratch);
    }
}

pub trait GLWEAutomorphism<BE: Backend> {
    fn glwe_automorphism_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_automorphism<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_inplace<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_add<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_add_inplace<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_sub<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_sub_negate<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_sub_inplace<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_sub_negate_inplace<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;
}

impl<BE: Backend> GLWEAutomorphism<BE> for Module<BE>
where
    Self: Sized
        + GLWEKeyswitch<BE>
        + GLWEKeySwitchInternal<BE>
        + VecZnxNormalize<BE>
        + VecZnxAutomorphismInplace<BE>
        + VecZnxAutomorphismInplaceTmpBytes
        + VecZnxBigAutomorphismInplace<BE>
        + VecZnxBigAutomorphismInplaceTmpBytes
        + VecZnxBigSubSmallInplace<BE>
        + VecZnxBigSubSmallNegateInplace<BE>
        + VecZnxBigAddSmallInplace<BE>
        + VecZnxBigNormalize<BE>
        + GLWENormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_automorphism_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, res_infos.n());
        assert_eq!(self.n() as u32, a_infos.n());
        assert_eq!(self.n() as u32, key_infos.n());

        let lvl_0: usize = self.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos);
        let lvl_1: usize = self
            .vec_znx_automorphism_inplace_tmp_bytes()
            .max(self.vec_znx_big_automorphism_inplace_tmp_bytes());

        lvl_0.max(lvl_1)
    }

    fn glwe_automorphism<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes(res, a, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes(res, a, key)
        );

        self.glwe_keyswitch(res, a, key, scratch);

        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_automorphism_inplace(key.p(), res.data_mut(), i, scratch);
        }
    }

    fn glwe_automorphism_inplace<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes(res, res, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes(res, res, key)
        );

        self.glwe_keyswitch_inplace(res, key, scratch);

        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_automorphism_inplace(key.p(), res.data_mut(), i, scratch);
        }
    }

    fn glwe_automorphism_add<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes(res, a, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes(res, a, key)
        );

        let a_base2k: usize = a.base2k().into();
        let key_base2k: usize = key.base2k().into();
        let res_base2k: usize = res.base2k().into();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // TODO: optimise size

        if a_base2k != key_base2k {
            let (mut a_conv, scratch_2) = scratch_1.take_glwe(&GLWELayout {
                n: a.n(),
                base2k: key.base2k(),
                k: a.k(),
                rank: a.rank(),
            });
            self.glwe_normalize(&mut a_conv, a, scratch_2);
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, &a_conv, key, scratch_2);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch_2);
                self.vec_znx_big_add_small_inplace(&mut res_big, i, a_conv.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_2);
            }
        } else {
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, a, key, scratch_1);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch_1);
                self.vec_znx_big_add_small_inplace(&mut res_big, i, a.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_1);
            }
        };
    }

    fn glwe_automorphism_add_inplace<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes(res, res, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes(res, res, key)
        );

        let key_base2k: usize = key.base2k().into();
        let res_base2k: usize = res.base2k().into();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // TODO: optimise size

        if res_base2k != key_base2k {
            let (mut res_conv, scratch_2) = scratch_1.take_glwe(&GLWELayout {
                n: res.n(),
                base2k: key.base2k(),
                k: res.k(),
                rank: res.rank(),
            });
            self.glwe_normalize(&mut res_conv, res, scratch_2);
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, &res_conv, key, scratch_2);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch_2);
                self.vec_znx_big_add_small_inplace(&mut res_big, i, res_conv.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_2);
            }
        } else {
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, res, key, scratch_1);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch_1);
                self.vec_znx_big_add_small_inplace(&mut res_big, i, res.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_1);
            }
        };
    }

    fn glwe_automorphism_sub<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes(res, a, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes(res, a, key)
        );

        let a_base2k: usize = a.base2k().into();
        let key_base2k: usize = key.base2k().into();
        let res_base2k: usize = res.base2k().into();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // TODO: optimise size

        if a_base2k != key_base2k {
            let (mut a_conv, scratch_2) = scratch_1.take_glwe(&GLWELayout {
                n: a.n(),
                base2k: key.base2k(),
                k: a.k(),
                rank: a.rank(),
            });
            self.glwe_normalize(&mut a_conv, a, scratch_2);
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, &a_conv, key, scratch_2);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch_2);
                self.vec_znx_big_sub_small_inplace(&mut res_big, i, a_conv.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_2);
            }
        } else {
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, a, key, scratch_1);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch_1);
                self.vec_znx_big_sub_small_inplace(&mut res_big, i, a.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_1);
            }
        };
    }

    fn glwe_automorphism_sub_negate<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes(res, a, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes(res, a, key)
        );

        let a_base2k: usize = a.base2k().into();
        let key_base2k: usize = key.base2k().into();
        let res_base2k: usize = res.base2k().into();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // TODO: optimise size

        if a_base2k != key_base2k {
            let (mut a_conv, scratch_2) = scratch_1.take_glwe(&GLWELayout {
                n: a.n(),
                base2k: key.base2k(),
                k: a.k(),
                rank: a.rank(),
            });
            self.glwe_normalize(&mut a_conv, a, scratch_2);
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, &a_conv, key, scratch_2);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch_2);
                self.vec_znx_big_sub_small_negate_inplace(&mut res_big, i, a_conv.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_2);
            }
        } else {
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, a, key, scratch_1);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch_1);
                self.vec_znx_big_sub_small_negate_inplace(&mut res_big, i, a.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_1);
            }
        };
    }

    fn glwe_automorphism_sub_inplace<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes(res, res, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes(res, res, key)
        );

        let key_base2k: usize = key.base2k().into();
        let res_base2k: usize = res.base2k().into();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // TODO: optimise size

        if res_base2k != key_base2k {
            let (mut res_conv, scratch_2) = scratch_1.take_glwe(&GLWELayout {
                n: res.n(),
                base2k: key.base2k(),
                k: res.k(),
                rank: res.rank(),
            });
            self.glwe_normalize(&mut res_conv, res, scratch_2);
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, &res_conv, key, scratch_2);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch_2);
                self.vec_znx_big_sub_small_inplace(&mut res_big, i, res_conv.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_2);
            }
        } else {
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, res, key, scratch_1);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch_1);
                self.vec_znx_big_sub_small_inplace(&mut res_big, i, res.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_1);
            }
        };
    }

    fn glwe_automorphism_sub_negate_inplace<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes(res, res, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes(res, res, key)
        );

        let key_base2k: usize = key.base2k().into();
        let res_base2k: usize = res.base2k().into();

        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // TODO: optimise size

        if res_base2k != key_base2k {
            let (mut res_conv, scratch_2) = scratch_1.take_glwe(&GLWELayout {
                n: res.n(),
                base2k: key.base2k(),
                k: res.k(),
                rank: res.rank(),
            });
            self.glwe_normalize(&mut res_conv, res, scratch_2);
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, &res_conv, key, scratch_2);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch_2);
                self.vec_znx_big_sub_small_negate_inplace(&mut res_big, i, res_conv.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_2);
            }
        } else {
            let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_keyswitch_internal(res_dft, res, key, scratch_1);
            for i in 0..res.rank().as_usize() + 1 {
                self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch_1);
                self.vec_znx_big_sub_small_negate_inplace(&mut res_big, i, res.data(), i);
                self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_1);
            }
        };
    }
}
