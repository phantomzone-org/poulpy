use backend::{Backend, Module, Scratch, VecZnxAlloc, VecZnxDftAllocBytes, VecZnxDftFromVecZnx};

use crate::{
    AutomorphismExecFamily, AutomorphismKeyExec, GGSWCiphertext, GGSWKeySwitchFamily, GLWECiphertext, GLWETensorKeyExec, Infos,
};

impl GGSWCiphertext<Vec<u8>> {
    pub fn automorphism_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_in: usize,
        k_ksk: usize,
        digits_ksk: usize,
        k_tsk: usize,
        digits_tsk: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: AutomorphismExecFamily<B> + GGSWKeySwitchFamily<B>,
    {
        let cols: usize = rank + 1;
        let out_size: usize = k_out.div_ceil(basek);
        let res: usize = module.bytes_of_vec_znx(cols, out_size);
        let res_dft: usize = module.vec_znx_dft_alloc_bytes(cols, out_size);
        let ci_dft: usize = module.vec_znx_dft_alloc_bytes(cols, out_size);
        let ks_internal: usize =
            GLWECiphertext::keyswitch_scratch_space(module, basek, k_out, k_in, k_ksk, digits_ksk, rank, rank);
        let expand: usize = GGSWCiphertext::expand_row_scratch_space(module, basek, k_out, k_tsk, digits_tsk, rank);
        res + ci_dft + (ks_internal | expand | res_dft)
    }

    pub fn automorphism_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_ksk: usize,
        digits_ksk: usize,
        k_tsk: usize,
        digits_tsk: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: AutomorphismExecFamily<B> + GGSWKeySwitchFamily<B>,
    {
        GGSWCiphertext::automorphism_scratch_space(
            module, basek, k_out, k_out, k_ksk, digits_ksk, k_tsk, digits_tsk, rank,
        )
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GGSWCiphertext<DataSelf> {
    pub fn automorphism<DataLhs: AsRef<[u8]>, DataAk: AsRef<[u8]>, DataTsk: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GGSWCiphertext<DataLhs>,
        auto_key: &AutomorphismKeyExec<DataAk, B>,
        tensor_key: &GLWETensorKeyExec<DataTsk, B>,
        scratch: &mut Scratch,
    ) where
        Module<B>: AutomorphismExecFamily<B> + GGSWKeySwitchFamily<B>,
    {
        #[cfg(debug_assertions)]
        {
            use crate::Infos;

            assert_eq!(
                self.rank(),
                lhs.rank(),
                "ggsw_out rank: {} != ggsw_in rank: {}",
                self.rank(),
                lhs.rank()
            );
            assert_eq!(
                self.rank(),
                auto_key.rank(),
                "ggsw_in rank: {} != auto_key rank: {}",
                self.rank(),
                auto_key.rank()
            );
            assert_eq!(
                self.rank(),
                tensor_key.rank(),
                "ggsw_in rank: {} != tensor_key rank: {}",
                self.rank(),
                tensor_key.rank()
            );
            assert!(
                scratch.available()
                    >= GGSWCiphertext::automorphism_scratch_space(
                        module,
                        self.basek(),
                        self.k(),
                        lhs.k(),
                        auto_key.k(),
                        auto_key.digits(),
                        tensor_key.k(),
                        tensor_key.digits(),
                        self.rank(),
                    )
            )
        };

        let rank: usize = self.rank();
        let cols: usize = rank + 1;

        // Keyswitch the j-th row of the col 0
        (0..lhs.rows()).for_each(|row_i| {
            // Key-switch column 0, i.e.
            // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0, a1, a2) -> (-(a0pi^-1(s0) + a1pi^-1(s1) + a2pi^-1(s2)) + M[i], a0, a1, a2)
            self.at_mut(row_i, 0)
                .automorphism(module, &lhs.at(row_i, 0), auto_key, scratch);

            // Isolates DFT(AUTO(a[i]))
            let (mut ci_dft, scratch1) = scratch.tmp_vec_znx_dft(module, cols - 1, self.size());
            (1..cols).for_each(|i| {
                module.vec_znx_dft_from_vec_znx(1, 0, &mut ci_dft, i - 1, &self.at(row_i, 0).data, i);
            });

            // Generates
            //
            // col 1: (-(b0s0 + b1s1 + b2s2)    , b0 + pi(M[i]), b1           , b2           )
            // col 2: (-(c0s0 + c1s1 + c2s2)    , c0           , c1 + pi(M[i]), c2           )
            // col 3: (-(d0s0 + d1s1 + d2s2)    , d0           , d1           , d2 + pi(M[i]))
            (1..cols).for_each(|col_j| {
                self.expand_row(module, row_i, col_j, &ci_dft, tensor_key, scratch1);
            });
        })
    }

    pub fn automorphism_inplace<DataKsk: AsRef<[u8]>, DataTsk: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        auto_key: &AutomorphismKeyExec<DataKsk, B>,
        tensor_key: &GLWETensorKeyExec<DataTsk, B>,
        scratch: &mut Scratch,
    ) where
        Module<B>: AutomorphismExecFamily<B> + GGSWKeySwitchFamily<B>,
    {
        unsafe {
            let self_ptr: *mut GGSWCiphertext<DataSelf> = self as *mut GGSWCiphertext<DataSelf>;
            self.automorphism(module, &*self_ptr, auto_key, tensor_key, scratch);
        }
    }
}
