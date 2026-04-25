use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxAutomorphismInplace, VecZnxAutomorphismInplaceTmpBytes, VecZnxBigAddSmallAssign,
        VecZnxBigAutomorphismInplace, VecZnxBigAutomorphismInplaceTmpBytes, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallInplace, VecZnxBigSubSmallNegateInplace, VecZnxDftBytesOf, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes, VecZnxNormalize,
    },
    layouts::{Backend, Module, ScratchArena, VecZnxBigReborrowBackendRef, VecZnxDftReborrowBackendRef},
};

pub use crate::api::GLWEAutomorphism;
use crate::{
    GLWEKeySwitchInternal, GLWEKeyswitch, GLWENormalize, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GLWE, GLWEBackendMut, GLWEBackendRef, GLWEInfos, GetGaloisElement, LWEInfos, glwe_backend_ref_from_mut,
        prepared::GGLWEPreparedToBackendRef,
    },
};

pub(crate) trait GLWEAutomorphismDefault<BE: Backend>:
    Sized
    + GLWEKeyswitch<BE>
    + GLWEKeySwitchInternal<BE>
    + VecZnxNormalize<BE>
    + VecZnxAutomorphismInplace<BE>
    + VecZnxAutomorphismInplaceTmpBytes
    + VecZnxBigAutomorphismInplace<BE>
    + VecZnxBigAutomorphismInplaceTmpBytes
    + VecZnxBigBytesOf
    + VecZnxBigSubSmallInplace<BE>
    + VecZnxBigSubSmallNegateInplace<BE>
    + VecZnxBigAddSmallAssign<BE>
    + VecZnxBigNormalize<BE>
    + VecZnxBigNormalizeTmpBytes
    + VecZnxDftBytesOf
    + VecZnxIdftApply<BE>
    + VecZnxIdftApplyTmpBytes
    + GLWENormalize<BE>
where
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn glwe_automorphism_tmp_bytes_default<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, res_infos.n());
        assert_eq!(self.n() as u32, a_infos.n());
        assert_eq!(self.n() as u32, key_infos.n());

        let lvl_conv: usize = if res_infos.max_k() > a_infos.max_k() {
            GLWE::<Vec<u8>>::bytes_of_from_infos(res_infos)
        } else {
            GLWE::<Vec<u8>>::bytes_of_from_infos(a_infos)
        };
        let lvl_0: usize = self.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos);
        let cols: usize = res_infos.rank().as_usize() + 1;
        let lvl_1: usize = self.vec_znx_automorphism_inplace_tmp_bytes();
        let lvl_2: usize = lvl_conv
            + self.bytes_of_vec_znx_dft(cols, key_infos.size())
            + self.bytes_of_vec_znx_big(cols, key_infos.size())
            + self
                .vec_znx_idft_apply_tmp_bytes()
                .max(self.vec_znx_big_automorphism_inplace_tmp_bytes())
                .max(self.vec_znx_big_normalize_tmp_bytes());

        lvl_0.max(lvl_1).max(lvl_2)
    }

    fn glwe_automorphism_default<'s, 'r, 'a, K>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &GLWEBackendRef<'a, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes_default(res, a, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes_default(res, a, key)
        );

        self.glwe_keyswitch(res, a, key, scratch);

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_automorphism_inplace(key.p(), &mut res.data, i, &mut scratch.borrow());
        }
    }

    fn glwe_automorphism_inplace_default<'s, 'r, K>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes_default(res, res, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes_default(res, res, key)
        );

        self.glwe_keyswitch_assign(res, key, scratch);

        for i in 0..res.rank().as_usize() + 1 {
            self.vec_znx_automorphism_inplace(key.p(), &mut res.data, i, &mut scratch.borrow());
        }
    }

    fn glwe_automorphism_add_default<'s, 'r, 'a, K>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &GLWEBackendRef<'a, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes_default(res, a, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes_default(res, a, key)
        );

        let key_base2k: usize = key.base2k().into();
        let res_base2k: usize = res.base2k().into();
        let cols: usize = (res.rank() + 1).into();
        let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft(self, cols, key.size());
        let mut a_layout = a.glwe_layout();
        a_layout.base2k = key.base2k();
        let (mut a_conv, mut scratch_2) = scratch_1.take_glwe(&a_layout);
        self.glwe_normalize(&mut a_conv, a, &mut scratch_2);
        let a_norm = glwe_backend_ref_from_mut::<BE>(&a_conv);

        {
            let mut scratch = scratch_2;
            self.glwe_keyswitch_internal(&mut res_dft, &a_norm, key, &mut scratch);
            let (mut res_big, mut scratch) = scratch.borrow().take_vec_znx_big(self, cols, key.size());
            let res_dft_ref = res_dft.reborrow_backend_ref();
            for i in 0..cols {
                scratch = scratch.apply_mut(|scratch| self.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, scratch));
            }
            self.vec_znx_big_add_small_assign(&mut res_big, 0, &a_norm.data, 0);

            for i in 0..cols {
                scratch = scratch.apply_mut(|scratch| self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch));
                self.vec_znx_big_add_small_assign(&mut res_big, i, &a_norm.data, i);
            }

            let res_big_ref = res_big.reborrow_backend_ref();
            for i in 0..cols {
                self.vec_znx_big_normalize(
                    &mut res.data,
                    res_base2k,
                    0,
                    i,
                    &res_big_ref,
                    key_base2k,
                    i,
                    &mut scratch.borrow(),
                );
            }
        }
    }

    fn glwe_automorphism_add_inplace_default<'s, 'r, K>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes_default(res, res, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes_default(res, res, key)
        );

        let key_base2k: usize = key.base2k().into();
        let res_base2k: usize = res.base2k().into();
        let cols: usize = (res.rank() + 1).into();
        let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft(self, cols, key.size());
        let mut res_layout = res.glwe_layout();
        res_layout.base2k = key.base2k();
        let (mut res_conv, mut scratch_2) = scratch_1.take_glwe(&res_layout);
        let scratch = {
            let res_ref = glwe_backend_ref_from_mut::<BE>(&*res);
            self.glwe_normalize(&mut res_conv, &res_ref, &mut scratch_2);
            scratch_2
        };
        let res_norm = glwe_backend_ref_from_mut::<BE>(&res_conv);

        {
            let mut scratch = scratch;
            self.glwe_keyswitch_internal(&mut res_dft, &res_norm, key, &mut scratch);
            let (mut res_big, mut scratch) = scratch.borrow().take_vec_znx_big(self, cols, key.size());
            let res_dft_ref = res_dft.reborrow_backend_ref();
            for i in 0..cols {
                scratch = scratch.apply_mut(|scratch| self.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, scratch));
            }
            self.vec_znx_big_add_small_assign(&mut res_big, 0, &res_norm.data, 0);

            for i in 0..cols {
                scratch = scratch.apply_mut(|scratch| self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch));
                self.vec_znx_big_add_small_assign(&mut res_big, i, &res_norm.data, i);
            }

            let res_big_ref = res_big.reborrow_backend_ref();
            for i in 0..cols {
                self.vec_znx_big_normalize(
                    &mut res.data,
                    res_base2k,
                    0,
                    i,
                    &res_big_ref,
                    key_base2k,
                    i,
                    &mut scratch.borrow(),
                );
            }
        }
    }

    fn glwe_automorphism_sub_default<'s, 'r, 'a, K>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &GLWEBackendRef<'a, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes_default(res, a, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes_default(res, a, key)
        );

        let key_base2k: usize = key.base2k().into();
        let res_base2k: usize = res.base2k().into();
        let cols: usize = (res.rank() + 1).into();
        let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft(self, cols, key.size());
        let mut a_layout = a.glwe_layout();
        a_layout.base2k = key.base2k();
        let (mut a_conv, mut scratch_2) = scratch_1.take_glwe(&a_layout);
        self.glwe_normalize(&mut a_conv, a, &mut scratch_2);
        let a_norm = glwe_backend_ref_from_mut::<BE>(&a_conv);

        {
            let mut scratch = scratch_2;
            self.glwe_keyswitch_internal(&mut res_dft, &a_norm, key, &mut scratch);
            let (mut res_big, mut scratch) = scratch.borrow().take_vec_znx_big(self, cols, key.size());
            let res_dft_ref = res_dft.reborrow_backend_ref();
            for i in 0..cols {
                scratch = scratch.apply_mut(|scratch| self.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, scratch));
            }

            for i in 0..cols {
                scratch = scratch.apply_mut(|scratch| self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch));
                self.vec_znx_big_sub_small_inplace(&mut res_big, i, &a_norm.data, i);
            }

            let res_big_ref = res_big.reborrow_backend_ref();
            for i in 0..cols {
                self.vec_znx_big_normalize(
                    &mut res.data,
                    res_base2k,
                    0,
                    i,
                    &res_big_ref,
                    key_base2k,
                    i,
                    &mut scratch.borrow(),
                );
            }
        }
    }

    fn glwe_automorphism_sub_negate_default<'s, 'r, 'a, K>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &GLWEBackendRef<'a, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes_default(res, a, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes_default(res, a, key)
        );

        let key_base2k: usize = key.base2k().into();
        let res_base2k: usize = res.base2k().into();
        let cols: usize = (res.rank() + 1).into();
        let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft(self, cols, key.size());
        let mut a_layout = a.glwe_layout();
        a_layout.base2k = key.base2k();
        let (mut a_conv, mut scratch_2) = scratch_1.take_glwe(&a_layout);
        self.glwe_normalize(&mut a_conv, a, &mut scratch_2);
        let a_norm = glwe_backend_ref_from_mut::<BE>(&a_conv);

        {
            let mut scratch = scratch_2;
            self.glwe_keyswitch_internal(&mut res_dft, &a_norm, key, &mut scratch);
            let (mut res_big, mut scratch) = scratch.borrow().take_vec_znx_big(self, cols, key.size());
            let res_dft_ref = res_dft.reborrow_backend_ref();
            for i in 0..cols {
                scratch = scratch.apply_mut(|scratch| self.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, scratch));
            }

            for i in 0..cols {
                scratch = scratch.apply_mut(|scratch| self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch));
                self.vec_znx_big_sub_small_negate_inplace(&mut res_big, i, &a_norm.data, i);
            }

            let res_big_ref = res_big.reborrow_backend_ref();
            for i in 0..cols {
                self.vec_znx_big_normalize(
                    &mut res.data,
                    res_base2k,
                    0,
                    i,
                    &res_big_ref,
                    key_base2k,
                    i,
                    &mut scratch.borrow(),
                );
            }
        }
    }

    fn glwe_automorphism_sub_inplace_default<'s, 'r, K>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes_default(res, res, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes_default(res, res, key)
        );

        let key_base2k: usize = key.base2k().into();
        let res_base2k: usize = res.base2k().into();
        let cols: usize = (res.rank() + 1).into();
        let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft(self, cols, key.size());
        let mut res_layout = res.glwe_layout();
        res_layout.base2k = key.base2k();
        let (mut res_conv, mut scratch_2) = scratch_1.take_glwe(&res_layout);
        let scratch = {
            let res_ref = glwe_backend_ref_from_mut::<BE>(&*res);
            self.glwe_normalize(&mut res_conv, &res_ref, &mut scratch_2);
            scratch_2
        };
        let res_norm = glwe_backend_ref_from_mut::<BE>(&res_conv);

        {
            let mut scratch = scratch;
            self.glwe_keyswitch_internal(&mut res_dft, &res_norm, key, &mut scratch);
            let (mut res_big, mut scratch) = scratch.borrow().take_vec_znx_big(self, cols, key.size());
            let res_dft_ref = res_dft.reborrow_backend_ref();
            for i in 0..cols {
                scratch = scratch.apply_mut(|scratch| self.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, scratch));
            }

            for i in 0..cols {
                scratch = scratch.apply_mut(|scratch| self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch));
                self.vec_znx_big_sub_small_inplace(&mut res_big, i, &res_norm.data, i);
            }

            let res_big_ref = res_big.reborrow_backend_ref();
            for i in 0..cols {
                self.vec_znx_big_normalize(
                    &mut res.data,
                    res_base2k,
                    0,
                    i,
                    &res_big_ref,
                    key_base2k,
                    i,
                    &mut scratch.borrow(),
                );
            }
        }
    }

    fn glwe_automorphism_sub_negate_inplace_default<'s, 'r, K>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        assert!(
            scratch.available() >= self.glwe_automorphism_tmp_bytes_default(res, res, key),
            "scratch.available(): {} < GLWEAutomorphism::glwe_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_tmp_bytes_default(res, res, key)
        );

        let key_base2k: usize = key.base2k().into();
        let res_base2k: usize = res.base2k().into();
        let cols: usize = (res.rank() + 1).into();
        let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft(self, cols, key.size());
        let mut res_layout = res.glwe_layout();
        res_layout.base2k = key.base2k();
        let (mut res_conv, mut scratch_2) = scratch_1.take_glwe(&res_layout);
        let scratch = {
            let res_ref = glwe_backend_ref_from_mut::<BE>(&*res);
            self.glwe_normalize(&mut res_conv, &res_ref, &mut scratch_2);
            scratch_2
        };
        let res_norm = glwe_backend_ref_from_mut::<BE>(&res_conv);

        {
            let mut scratch = scratch;
            self.glwe_keyswitch_internal(&mut res_dft, &res_norm, key, &mut scratch);
            let (mut res_big, mut scratch) = scratch.borrow().take_vec_znx_big(self, cols, key.size());
            let res_dft_ref = res_dft.reborrow_backend_ref();
            for i in 0..cols {
                scratch = scratch.apply_mut(|scratch| self.vec_znx_idft_apply(&mut res_big, i, &res_dft_ref, i, scratch));
            }

            for i in 0..cols {
                scratch = scratch.apply_mut(|scratch| self.vec_znx_big_automorphism_inplace(key.p(), &mut res_big, i, scratch));
                self.vec_znx_big_sub_small_negate_inplace(&mut res_big, i, &res_norm.data, i);
            }

            let res_big_ref = res_big.reborrow_backend_ref();
            for i in 0..cols {
                self.vec_znx_big_normalize(
                    &mut res.data,
                    res_base2k,
                    0,
                    i,
                    &res_big_ref,
                    key_base2k,
                    i,
                    &mut scratch.borrow(),
                );
            }
        }
    }
}

impl<BE: Backend> GLWEAutomorphismDefault<BE> for Module<BE>
where
    Self: Sized
        + GLWEKeyswitch<BE>
        + GLWEKeySwitchInternal<BE>
        + VecZnxNormalize<BE>
        + VecZnxAutomorphismInplace<BE>
        + VecZnxAutomorphismInplaceTmpBytes
        + VecZnxBigAutomorphismInplace<BE>
        + VecZnxBigAutomorphismInplaceTmpBytes
        + VecZnxBigBytesOf
        + VecZnxBigSubSmallInplace<BE>
        + VecZnxBigSubSmallNegateInplace<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + GLWENormalize<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
}
