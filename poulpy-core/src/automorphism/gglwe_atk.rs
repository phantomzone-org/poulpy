use poulpy_hal::{
    api::VecZnxAutomorphism,
    layouts::{Backend, DataMut, GaloisElement, Module, Scratch},
};

use crate::{
    ScratchTakeCore,
    automorphism::glwe_ct::GLWEAutomorphism,
    layouts::{
        GGLWE, GGLWEInfos, GGLWEPreparedToRef, GGLWEToMut, GGLWEToRef, GLWE, GLWEAutomorphismKey, GetGaloisElement,
        SetGaloisElement,
    },
};

impl GLWEAutomorphismKey<Vec<u8>> {
    pub fn automorphism_tmp_bytes<R, A, K, M, BE: Backend>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
        M: GLWEAutomorphismKeyAutomorphism<BE>,
    {
        module.glwe_automorphism_key_automorphism_tmp_bytes(res_infos, a_infos, key_infos)
    }
}

impl<DataSelf: DataMut> GLWEAutomorphismKey<DataSelf> {
    pub fn automorphism<A, K, M, BE: Backend>(&mut self, module: &M, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        A: GGLWEToRef + GetGaloisElement + GGLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
        M: GLWEAutomorphismKeyAutomorphism<BE>,
    {
        module.glwe_automorphism_key_automorphism(self, a, key, scratch);
    }

    pub fn automorphism_inplace<K, M, BE: Backend>(&mut self, module: &M, key: &K, scratch: &mut Scratch<BE>)
    where
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
        M: GLWEAutomorphismKeyAutomorphism<BE>,
    {
        module.glwe_automorphism_key_automorphism_inplace(self, key, scratch);
    }
}

impl<BE: Backend> GLWEAutomorphismKeyAutomorphism<BE> for Module<BE> where
    Self: GaloisElement + GLWEAutomorphism<BE> + VecZnxAutomorphism
{
}

pub trait GLWEAutomorphismKeyAutomorphism<BE: Backend>
where
    Self: GaloisElement + GLWEAutomorphism<BE> + VecZnxAutomorphism,
{
    fn glwe_automorphism_key_automorphism_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
    {
        self.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos)
    }

    fn glwe_automorphism_key_automorphism<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut + SetGaloisElement + GGLWEInfos,
        A: GGLWEToRef + GetGaloisElement + GGLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        assert!(
            res.dnum().as_u32() <= a.dnum().as_u32(),
            "res dnum: {} > a dnum: {}",
            res.dnum(),
            a.dnum()
        );

        assert_eq!(
            res.dsize(),
            a.dsize(),
            "res dnum: {} != a dnum: {}",
            res.dsize(),
            a.dsize()
        );

        let cols_out: usize = (key.rank_out() + 1).into();
        let cols_in: usize = key.rank_in().into();

        let p: i64 = a.p();
        let p_inv: i64 = self.galois_element_inv(p);

        {
            let res: &mut GGLWE<&mut [u8]> = &mut res.to_mut();
            let a: &GGLWE<&[u8]> = &a.to_ref();

            for row in 0..res.dnum().as_usize() {
                for col in 0..cols_in {
                    let mut res_tmp: GLWE<&mut [u8]> = res.at_mut(row, col);
                    let a_ct: GLWE<&[u8]> = a.at(row, col);

                    // Reverts the automorphism X^{-k}: (-pi^{-1}_{k}(s)a + s, a) to (-sa + pi_{k}(s), a)
                    for i in 0..cols_out {
                        self.vec_znx_automorphism(p, res_tmp.data_mut(), i, &a_ct.data, i);
                    }

                    // Key-switch (-sa + pi_{k}(s), a) to (-pi^{-1}_{k'}(s)a + pi_{k}(s), a)
                    self.glwe_keyswitch_inplace(&mut res_tmp, key, scratch);

                    // Applies back the automorphism X^{-k}: (-pi^{-1}_{k'}(s)a + pi_{k}(s), a) to (-pi^{-1}_{k'+k}(s)a + s, a)
                    (0..cols_out).for_each(|i| {
                        self.vec_znx_automorphism_inplace(p_inv, res_tmp.data_mut(), i, scratch);
                    });
                }
            }
        }

        res.set_p((p * key.p()) % (self.cyclotomic_order() as i64));
    }

    fn glwe_automorphism_key_automorphism_inplace<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut + SetGaloisElement + GetGaloisElement + GGLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        assert_eq!(
            res.rank(),
            key.rank(),
            "key rank: {} != key rank: {}",
            res.rank(),
            key.rank()
        );

        let cols_out: usize = (key.rank_out() + 1).into();
        let cols_in: usize = key.rank_in().into();
        let p: i64 = res.p();
        let p_inv: i64 = self.galois_element_inv(p);

        {
            let res: &mut GGLWE<&mut [u8]> = &mut res.to_mut();
            for row in 0..res.dnum().as_usize() {
                for col in 0..cols_in {
                    let mut res_tmp: GLWE<&mut [u8]> = res.at_mut(row, col);

                    // Reverts the automorphism X^{-k}: (-pi^{-1}_{k}(s)a + s, a) to (-sa + pi_{k}(s), a)
                    for i in 0..cols_out {
                        self.vec_znx_automorphism_inplace(p, res_tmp.data_mut(), i, scratch);
                    }

                    // Key-switch (-sa + pi_{k}(s), a) to (-pi^{-1}_{k'}(s)a + pi_{k}(s), a)
                    self.glwe_keyswitch_inplace(&mut res_tmp, key, scratch);

                    // Applies back the automorphism X^{-k}: (-pi^{-1}_{k'}(s)a + pi_{k}(s), a) to (-pi^{-1}_{k'+k}(s)a + s, a)
                    for i in 0..cols_out {
                        self.vec_znx_automorphism_inplace(p_inv, res_tmp.data_mut(), i, scratch);
                    }
                }
            }
        }

        res.set_p((res.p() * key.p()) % (self.cyclotomic_order() as i64));
    }
}
