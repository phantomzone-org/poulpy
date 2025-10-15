use poulpy_hal::{
    api::{
        ScratchAvailable, VecZnxAutomorphism, VecZnxAutomorphismInplace, VecZnxBigAddSmallInplace, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyConsume, VecZnxNormalize,
        VecZnxNormalizeTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxZero},
};

use crate::layouts::{AutomorphismKey, GGLWEInfos, GLWE, prepared::AutomorphismKeyPrepared};

impl AutomorphismKey<Vec<u8>> {
    pub fn automorphism_tmp_bytes<B: Backend, OUT, IN, KEY>(
        module: &Module<B>,
        out_infos: &OUT,
        in_infos: &IN,
        key_infos: &KEY,
    ) -> usize
    where
        OUT: GGLWEInfos,
        IN: GGLWEInfos,
        KEY: GGLWEInfos,
        Module<B>: VecZnxDftBytesOf + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
    {
        GLWE::keyswitch_tmp_bytes(
            module,
            &out_infos.glwe_layout(),
            &in_infos.glwe_layout(),
            key_infos,
        )
    }

    pub fn automorphism_inplace_tmp_bytes<B: Backend, OUT, KEY>(module: &Module<B>, out_infos: &OUT, key_infos: &KEY) -> usize
    where
        OUT: GGLWEInfos,
        KEY: GGLWEInfos,
        Module<B>: VecZnxDftBytesOf + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
    {
        AutomorphismKey::automorphism_tmp_bytes(module, out_infos, out_infos, key_infos)
    }
}

impl<DataSelf: DataMut> AutomorphismKey<DataSelf> {
    pub fn automorphism<DataLhs: DataRef, DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &AutomorphismKey<DataLhs>,
        rhs: &AutomorphismKeyPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftBytesOf
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxAutomorphism
            + VecZnxAutomorphismInplace<B>
            + VecZnxNormalize<B>
            + VecZnxNormalizeTmpBytes,
        Scratch<B>: ScratchAvailable,
    {
        #[cfg(debug_assertions)]
        {
            use crate::layouts::LWEInfos;

            assert_eq!(
                self.rank_in(),
                lhs.rank_in(),
                "ksk_out input rank: {} != ksk_in input rank: {}",
                self.rank_in(),
                lhs.rank_in()
            );
            assert_eq!(
                self.rank_out(),
                rhs.rank_in(),
                "ksk_in output rank: {} != ksk_apply input rank: {}",
                self.rank_out(),
                rhs.rank_in()
            );
            assert_eq!(
                self.rank_out(),
                rhs.rank_out(),
                "ksk_out output rank: {} != ksk_apply output rank: {}",
                self.rank_out(),
                rhs.rank_out()
            );
            assert!(
                self.k() <= lhs.k(),
                "output k={} cannot be greater than input k={}",
                self.k(),
                lhs.k()
            )
        }

        let cols_out: usize = (rhs.rank_out() + 1).into();

        let p: i64 = lhs.p();
        let p_inv: i64 = module.galois_element_inv(p);

        (0..self.rank_in().into()).for_each(|col_i| {
            (0..self.dnum().into()).for_each(|row_j| {
                let mut res_ct: GLWE<&mut [u8]> = self.at_mut(row_j, col_i);
                let lhs_ct: GLWE<&[u8]> = lhs.at(row_j, col_i);

                // Reverts the automorphism X^{-k}: (-pi^{-1}_{k}(s)a + s, a) to (-sa + pi_{k}(s), a)
                (0..cols_out).for_each(|i| {
                    module.vec_znx_automorphism(lhs.p(), &mut res_ct.data, i, &lhs_ct.data, i);
                });

                // Key-switch (-sa + pi_{k}(s), a) to (-pi^{-1}_{k'}(s)a + pi_{k}(s), a)
                res_ct.keyswitch_inplace(module, &rhs.key, scratch);

                // Applies back the automorphism X^{-k}: (-pi^{-1}_{k'}(s)a + pi_{k}(s), a) to (-pi^{-1}_{k'+k}(s)a + s, a)
                (0..cols_out).for_each(|i| {
                    module.vec_znx_automorphism_inplace(p_inv, &mut res_ct.data, i, scratch);
                });
            });
        });

        (self.dnum().min(lhs.dnum()).into()..self.dnum().into()).for_each(|row_i| {
            (0..self.rank_in().into()).for_each(|col_j| {
                self.at_mut(row_i, col_j).data.zero();
            });
        });

        self.p = (lhs.p * rhs.p) % (module.cyclotomic_order() as i64);
    }

    pub fn automorphism_inplace<DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &AutomorphismKeyPrepared<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftBytesOf
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxAutomorphism
            + VecZnxAutomorphismInplace<B>
            + VecZnxNormalize<B>
            + VecZnxNormalizeTmpBytes,
        Scratch<B>: ScratchAvailable,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.rank_out(),
                rhs.rank_in(),
                "ksk_in output rank: {} != ksk_apply input rank: {}",
                self.rank_out(),
                rhs.rank_in()
            );
            assert_eq!(
                self.rank_out(),
                rhs.rank_out(),
                "ksk_out output rank: {} != ksk_apply output rank: {}",
                self.rank_out(),
                rhs.rank_out()
            );
        }

        let cols_out: usize = (rhs.rank_out() + 1).into();

        let p: i64 = self.p();
        let p_inv = module.galois_element_inv(p);

        (0..self.rank_in().into()).for_each(|col_i| {
            (0..self.dnum().into()).for_each(|row_j| {
                let mut res_ct: GLWE<&mut [u8]> = self.at_mut(row_j, col_i);

                // Reverts the automorphism X^{-k}: (-pi^{-1}_{k}(s)a + s, a) to (-sa + pi_{k}(s), a)
                (0..cols_out).for_each(|i| {
                    module.vec_znx_automorphism_inplace(p_inv, &mut res_ct.data, i, scratch);
                });

                // Key-switch (-sa + pi_{k}(s), a) to (-pi^{-1}_{k'}(s)a + pi_{k}(s), a)
                res_ct.keyswitch_inplace(module, &rhs.key, scratch);

                // Applies back the automorphism X^{-k}: (-pi^{-1}_{k'}(s)a + pi_{k}(s), a) to (-pi^{-1}_{k'+k}(s)a + s, a)
                (0..cols_out).for_each(|i| {
                    module.vec_znx_automorphism_inplace(p_inv, &mut res_ct.data, i, scratch);
                });
            });
        });

        self.p = (self.p * rhs.p) % (module.cyclotomic_order() as i64);
    }
}
