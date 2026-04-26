use poulpy_hal::{
    api::{VecZnxAutomorphismBackend, VecZnxAutomorphismInplace, VecZnxAutomorphismInplaceTmpBytes},
    layouts::{Backend, CyclotomicOrder, GaloisElement, HostDataMut, Module, ScratchArena},
};

use crate::{
    GLWEKeyswitch, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GLWE, GetGaloisElement, SetGaloisElement,
        gglwe_at_backend_mut_from_mut, gglwe_at_backend_ref_from_ref, glwe_backend_ref_from_mut,
        prepared::GGLWEPreparedToBackendRef,
    },
};

#[doc(hidden)]
pub trait GLWEAutomorphismKeyAutomorphismDefault<BE: Backend>:
    Sized
    + GaloisElement
    + GLWEKeyswitch<BE>
    + VecZnxAutomorphismBackend<BE>
    + VecZnxAutomorphismInplace<BE>
    + VecZnxAutomorphismInplaceTmpBytes
    + CyclotomicOrder
where
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    for<'s> BE::BufMut<'s>: HostDataMut,
{
    fn glwe_automorphism_key_automorphism_tmp_bytes_default<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, res_infos.n());
        assert_eq!(self.n() as u32, a_infos.n());
        assert_eq!(self.n() as u32, key_infos.n());

        let lvl_0: usize = if res_infos.glwe_layout() == a_infos.glwe_layout() {
            self.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos)
        } else {
            self.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos) + GLWE::<Vec<u8>>::bytes_of_from_infos(a_infos)
        };
        let lvl_1: usize = self.vec_znx_automorphism_assign_tmp_bytes();

        lvl_0.max(lvl_1)
    }

    fn glwe_automorphism_key_automorphism_default<'s, R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + SetGaloisElement + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    {
        assert!(
            res.dnum().as_u32() <= a.dnum().as_u32(),
            "res dnum: {} > a dnum: {}",
            res.dnum(),
            a.dnum()
        );

        assert_eq!(res.dsize(), a.dsize(), "res dnum: {} != a dnum: {}", res.dsize(), a.dsize());
        assert_eq!(res.base2k(), a.base2k());
        assert!(
            scratch.available() >= self.glwe_automorphism_key_automorphism_tmp_bytes_default(res, a, key),
            "scratch.available(): {} < GLWEAutomorphismKeyAutomorphism::glwe_automorphism_key_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_key_automorphism_tmp_bytes_default(res, a, key)
        );

        let cols_out: usize = (key.rank_out() + 1).into();
        let cols_in: usize = key.rank_in().into();
        let p: i64 = a.p();
        let p_inv: i64 = self.galois_element_inv(p);
        let same_layout: bool = res.glwe_layout() == a.glwe_layout();

        {
            let res = &mut res.to_backend_mut();
            let a_backend = <A as GGLWEToBackendRef<BE>>::to_backend_ref(a);

            for row in 0..res.dnum().as_usize() {
                for col in 0..cols_in {
                    let mut res_tmp = gglwe_at_backend_mut_from_mut::<BE>(res, row, col);
                    let a_ct_backend = gglwe_at_backend_ref_from_ref::<BE>(&a_backend, row, col);

                    if same_layout {
                        for i in 0..cols_out {
                            self.vec_znx_automorphism_backend(p, &mut res_tmp.data, i, &a_ct_backend.data, i);
                        }

                        let mut scratch_iter = scratch.borrow();
                        self.glwe_keyswitch_inplace(&mut res_tmp, key, &mut scratch_iter);

                        for i in 0..cols_out {
                            self.vec_znx_automorphism_inplace(p_inv, &mut res_tmp.data, i, &mut scratch_iter);
                        }
                    } else {
                        let (mut tmp_glwe, mut scratch_iter) = scratch.borrow().take_glwe(&a_ct_backend);

                        for i in 0..cols_out {
                            self.vec_znx_automorphism_backend(p, &mut tmp_glwe.data, i, &a_ct_backend.data, i);
                        }

                        let tmp_glwe_ref = glwe_backend_ref_from_mut::<BE>(&tmp_glwe);
                        self.glwe_keyswitch(&mut res_tmp, &tmp_glwe_ref, key, &mut scratch_iter);

                        for i in 0..cols_out {
                            self.vec_znx_automorphism_inplace(p_inv, &mut res_tmp.data, i, &mut scratch_iter);
                        }
                    }
                }
            }
        }

        res.set_p((p * key.p()) % self.cyclotomic_order());
    }

    fn glwe_automorphism_key_automorphism_inplace_default<'s, R, K>(
        &self,
        res: &mut R,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + SetGaloisElement + GetGaloisElement + GGLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    {
        assert_eq!(res.rank(), key.rank(), "key rank: {} != key rank: {}", res.rank(), key.rank());
        assert!(
            scratch.available() >= self.glwe_automorphism_key_automorphism_tmp_bytes_default(res, res, key),
            "scratch.available(): {} < GLWEAutomorphismKeyAutomorphism::glwe_automorphism_key_automorphism_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_key_automorphism_tmp_bytes_default(res, res, key)
        );

        let cols_out: usize = (key.rank_out() + 1).into();
        let cols_in: usize = key.rank_in().into();
        let p: i64 = res.p();
        let p_inv: i64 = self.galois_element_inv(p);

        {
            let res = &mut res.to_backend_mut();
            for row in 0..res.dnum().as_usize() {
                for col in 0..cols_in {
                    let mut res_tmp = gglwe_at_backend_mut_from_mut::<BE>(res, row, col);

                    let mut scratch_iter = scratch.borrow();
                    for i in 0..cols_out {
                        self.vec_znx_automorphism_inplace(p, &mut res_tmp.data, i, &mut scratch_iter);
                    }

                    self.glwe_keyswitch_inplace(&mut res_tmp, key, &mut scratch_iter);

                    for i in 0..cols_out {
                        self.vec_znx_automorphism_inplace(p_inv, &mut res_tmp.data, i, &mut scratch_iter);
                    }
                }
            }
        }

        res.set_p((res.p() * key.p()) % self.cyclotomic_order());
    }
}

impl<BE: Backend> GLWEAutomorphismKeyAutomorphismDefault<BE> for Module<BE>
where
    Self: GaloisElement
        + GLWEKeyswitch<BE>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxAutomorphismInplace<BE>
        + VecZnxAutomorphismInplaceTmpBytes
        + CyclotomicOrder,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    for<'s> BE::BufMut<'s>: HostDataMut,
{
}

pub use crate::api::GLWEAutomorphismKeyAutomorphism;
