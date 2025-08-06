use backend::hal::{
    api::{ScratchAvailable, TakeVecZnxDft, VecZnxAutomorphism, VecZnxAutomorphismInplace, ZnxZero},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{AutomorphismKey, AutomorphismKeyExec, GLWECiphertext, GLWEKeyswitchFamily, Infos};

impl AutomorphismKey<Vec<u8>> {
    pub fn automorphism_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_in: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: GLWEKeyswitchFamily<B>,
    {
        GLWECiphertext::keyswitch_scratch_space(module, basek, k_out, k_in, k_ksk, digits, rank, rank)
    }

    pub fn automorphism_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: GLWEKeyswitchFamily<B>,
    {
        AutomorphismKey::automorphism_scratch_space(module, basek, k_out, k_out, k_ksk, digits, rank)
    }
}

impl<DataSelf: DataMut> AutomorphismKey<DataSelf> {
    pub fn automorphism<DataLhs: DataRef, DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &AutomorphismKey<DataLhs>,
        rhs: &AutomorphismKeyExec<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEKeyswitchFamily<B> + VecZnxAutomorphism + VecZnxAutomorphismInplace,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.rank_in(),
                lhs.rank_in(),
                "ksk_out input rank: {} != ksk_in input rank: {}",
                self.rank_in(),
                lhs.rank_in()
            );
            assert_eq!(
                lhs.rank_out(),
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

        let cols_out: usize = rhs.rank_out() + 1;

        let p: i64 = lhs.p();
        let p_inv = module.galois_element_inv(p);

        (0..self.rank_in()).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                let mut res_ct: GLWECiphertext<&mut [u8]> = self.at_mut(row_j, col_i);
                let lhs_ct: GLWECiphertext<&[u8]> = lhs.at(row_j, col_i);

                // Reverts the automorphism X^{-k}: (-pi^{-1}_{k}(s)a + s, a) to (-sa + pi_{k}(s), a)
                (0..cols_out).for_each(|i| {
                    module.vec_znx_automorphism(lhs.p(), &mut res_ct.data, i, &lhs_ct.data, i);
                });

                // Key-switch (-sa + pi_{k}(s), a) to (-pi^{-1}_{k'}(s)a + pi_{k}(s), a)
                res_ct.keyswitch_inplace(module, &rhs.key, scratch);

                // Applies back the automorphism X^{-k}: (-pi^{-1}_{k'}(s)a + pi_{k}(s), a) to (-pi^{-1}_{k'+k}(s)a + s, a)
                (0..cols_out).for_each(|i| {
                    module.vec_znx_automorphism_inplace(p_inv, &mut res_ct.data, i);
                });
            });
        });

        (self.rows().min(lhs.rows())..self.rows()).for_each(|row_i| {
            (0..self.rank_in()).for_each(|col_j| {
                self.at_mut(row_i, col_j).data.zero();
            });
        });

        self.p = (lhs.p * rhs.p) % (module.cyclotomic_order() as i64);
    }

    pub fn automorphism_inplace<DataRhs: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &AutomorphismKeyExec<DataRhs, B>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEKeyswitchFamily<B> + VecZnxAutomorphism + VecZnxAutomorphismInplace,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable,
    {
        unsafe {
            let self_ptr: *mut AutomorphismKey<DataSelf> = self as *mut AutomorphismKey<DataSelf>;
            self.automorphism(&module, &*self_ptr, rhs, scratch);
        }
    }
}
