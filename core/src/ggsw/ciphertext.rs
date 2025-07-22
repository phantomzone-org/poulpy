use backend::{
    Backend, MatZnx, MatZnxAlloc, Module, Scratch, VecZnxAlloc, VecZnxBigAddSmallInplace, VecZnxBigAllocBytes, VecZnxBigAutomorphismInplace, VecZnxBigNormalize, VecZnxBigSubSmallAInplace, VecZnxBigSubSmallBInplace, VecZnxDft, VecZnxDftAddInplace, VecZnxDftAllocBytes, VecZnxDftCopy, VecZnxDftFromVecZnx, VecZnxDftToVecZnxBigConsume, VecZnxDftToVecZnxBigTmpA, VecZnxScratch, VmpApply, VmpPMat, VmpPMatAlloc, VmpPMatAllocBytes, ZnxInfos, ZnxZero
};

use crate::{
    GLWEAutomorphismKeyExec, GLWECiphertext, GLWESwitchingKeyExec, GLWETensorKeyPrep, Infos
};

pub struct GGSWCiphertext<D> {
    pub(crate) data: MatZnx<D>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) digits: usize,
}

impl<D: AsRef<[u8]>> GGSWCiphertext<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWECiphertext<&[u8]> {
        GLWECiphertext {
            data: self.data.at(row, col),
            basek: self.basek,
            k: self.k,
        }
    }
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> GGSWCiphertext<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertext<&mut [u8]> {
        GLWECiphertext {
            data: self.data.at_mut(row, col),
            basek: self.basek,
            k: self.k,
        }
    }
}

impl GGSWCiphertext<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self {
        let size: usize = k.div_ceil(basek);
        debug_assert!(digits > 0, "invalid ggsw: `digits` == 0");

        debug_assert!(
            size > digits,
            "invalid ggsw: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid ggsw: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
        );

        Self {
            data: module.new_mat_znx(rows, rank + 1, rank + 1, k.div_ceil(basek)),
            basek,
            k: k,
            digits,
        }
    }

    pub fn bytes_of<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize {
        let size: usize = k.div_ceil(basek);
        debug_assert!(
            size > digits,
            "invalid ggsw: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid ggsw: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
        );

        module.bytes_of_mat_znx(rows, rank + 1, rank + 1, size)
    }
}

impl<D> Infos for GGSWCiphertext<D> {
    type Inner = MatZnx<D>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.basek
    }

    fn k(&self) -> usize {
        self.k
    }
}

impl<D> GGSWCiphertext<D> {
    pub fn rank(&self) -> usize {
        self.data.cols_out() - 1
    }

    pub fn digits(&self) -> usize {
        self.digits
    }
}

impl GGSWCiphertext<Vec<u8>> {

    pub(crate) fn expand_row_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        self_k: usize,
        k_tsk: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VecZnxBigAllocBytes + VecZnxBigNormalize<B> + VmpApply<B>,
    {
        let tsk_size: usize = k_tsk.div_ceil(basek);
        let self_size_out: usize = self_k.div_ceil(basek);
        let self_size_in: usize = self_size_out.div_ceil(digits);
        let tmp_dft_i: usize = module.vec_znx_dft_alloc_bytes(rank + 1, tsk_size);
        let tmp_a: usize = module.vec_znx_dft_alloc_bytes(1, self_size_in);
        let vmp: usize = module.vmp_apply_tmp_bytes(
            self_size_out,
            self_size_in,
            self_size_in,
            rank,
            rank,
            tsk_size,
        );
        let tmp_idft: usize = module.vec_znx_big_alloc_bytes(1, tsk_size);
        let norm: usize = module.vec_znx_normalize_tmp_bytes();
        tmp_dft_i + ((tmp_a + vmp) | (tmp_idft + norm))
    }

    pub fn keyswitch_scratch_space<B: Backend>(
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
        Module<B>: VecZnxDftAllocBytes + VecZnxBigAllocBytes + VecZnxBigNormalize<B> + VmpApply<B>,
    {
        let out_size: usize = k_out.div_ceil(basek);
        let res_znx: usize = module.bytes_of_vec_znx(rank + 1, out_size);
        let ci_dft: usize = module.vec_znx_dft_alloc_bytes(rank + 1, out_size);
        let ks: usize = GLWECiphertext::keyswitch_scratch_space(module, basek, k_out, k_in, k_ksk, digits_ksk, rank, rank);
        let expand_rows: usize = GGSWCiphertext::expand_row_scratch_space(module, basek, k_out, k_tsk, digits_tsk, rank);
        let res_dft: usize = module.vec_znx_dft_alloc_bytes(rank + 1, out_size);
        res_znx + ci_dft + (ks | expand_rows | res_dft)
    }

    pub fn keyswitch_inplace_scratch_space<B: Backend>(
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
        Module<B>: VecZnxDftAllocBytes + VecZnxBigAllocBytes + VecZnxBigNormalize<B> + VmpApply<B>,
    {
        GGSWCiphertext::keyswitch_scratch_space(
            module, basek, k_out, k_out, k_ksk, digits_ksk, k_tsk, digits_tsk, rank,
        )
    }

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
        Module<B>: VecZnxDftAllocBytes + VecZnxBigAllocBytes + VecZnxBigNormalize<B> + VmpApply<B>,
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
        Module<B>: VecZnxDftAllocBytes + VecZnxBigAllocBytes + VecZnxBigNormalize<B> + VmpApply<B>,
    {
        GGSWCiphertext::automorphism_scratch_space(
            module, basek, k_out, k_out, k_ksk, digits_ksk, k_tsk, digits_tsk, rank,
        )
    }

    pub fn external_product_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_in: usize,
        k_ggsw: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApply<B>,
    {
        GLWECiphertext::external_product_scratch_space(module, basek, k_out, k_in, k_ggsw, digits, rank)
    }

    pub fn external_product_inplace_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_out: usize,
        k_ggsw: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApply<B>,
    {
        GLWECiphertext::external_product_inplace_scratch_space(module, basek, k_out, k_ggsw, digits, rank)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GGSWCiphertext<DataSelf> {
    pub(crate) fn expand_row<DataCi: AsRef<[u8]>, DataTsk: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        row_i: usize,
        col_j: usize,
        ci_dft: &VecZnxDft<DataCi, B>,
        tsk: &GLWETensorKeyPrep<DataTsk, B>,
        scratch: &mut Scratch,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VecZnxBigAllocBytes
            + VecZnxBigNormalize<B>
            + VmpApply<B>
            + VecZnxDftCopy<B>
            + VecZnxDftAddInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxDftToVecZnxBigTmpA<B>,
    {
        let cols: usize = self.rank() + 1;

        assert!(
            scratch.available()
                >= GGSWCiphertext::expand_row_scratch_space(
                    module,
                    self.basek(),
                    self.k(),
                    tsk.k(),
                    tsk.digits(),
                    tsk.rank()
                )
        );

        // Example for rank 3:
        //
        // Note: M is a vector (m, Bm, B^2m, B^3m, ...), so each column is
        // actually composed of that many rows and we focus on a specific row here
        // implicitely given ci_dft.
        //
        // # Input
        //
        // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0    , a1    , a2    )
        // col 1: (0, 0, 0, 0)
        // col 2: (0, 0, 0, 0)
        // col 3: (0, 0, 0, 0)
        //
        // # Output
        //
        // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0       , a1       , a2       )
        // col 1: (-(b0s0 + b1s1 + b2s2)       , b0 + M[i], b1       , b2       )
        // col 2: (-(c0s0 + c1s1 + c2s2)       , c0       , c1 + M[i], c2       )
        // col 3: (-(d0s0 + d1s1 + d2s2)       , d0       , d1       , d2 + M[i])

        let digits: usize = tsk.digits();

        let (mut tmp_dft_i, scratch1) = scratch.tmp_vec_znx_dft(module, cols, tsk.size());
        let (mut tmp_a, scratch2) = scratch1.tmp_vec_znx_dft(module, 1, (ci_dft.size() + digits - 1) / digits);

        {
            // Performs a key-switch for each combination of s[i]*s[j], i.e. for a0, a1, a2
            //
            // # Example for col=1
            //
            // a0 * (-(f0s0 + f1s1 + f1s2) + s0^2, f0, f1, f2) = (-(a0f0s0 + a0f1s1 + a0f1s2) + a0s0^2, a0f0, a0f1, a0f2)
            // +
            // a1 * (-(g0s0 + g1s1 + g1s2) + s0s1, g0, g1, g2) = (-(a1g0s0 + a1g1s1 + a1g1s2) + a1s0s1, a1g0, a1g1, a1g2)
            // +
            // a2 * (-(h0s0 + h1s1 + h1s2) + s0s2, h0, h1, h2) = (-(a2h0s0 + a2h1s1 + a2h1s2) + a2s0s2, a2h0, a2h1, a2h2)
            // =
            // (-(x0s0 + x1s1 + x2s2) + s0(a0s0 + a1s1 + a2s2), x0, x1, x2)
            (1..cols).for_each(|col_i| {
                let pmat: &VmpPMat<DataTsk, B> = &tsk.at(col_i - 1, col_j - 1).key.data; // Selects Enc(s[i]s[j])

                // Extracts a[i] and multipies with Enc(s[i]s[j])
                (0..digits).for_each(|di| {
                    tmp_a.set_size((ci_dft.size() + di) / digits);

                    // Small optimization for digits > 2
                    // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
                    // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(digits-1) * B}.
                    // As such we can ignore the last digits-2 limbs safely of the sum of vmp products.
                    // It is possible to further ignore the last digits-1 limbs, but this introduce
                    // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
                    // noise is kept with respect to the ideal functionality.
                    tmp_dft_i.set_size(tsk.size() - ((digits - di) as isize - 2).max(0) as usize);

                    module.vec_znx_dft_copy(digits, digits - 1 - di, &mut tmp_a, 0, ci_dft, col_i);
                    if di == 0 && col_i == 1 {
                        module.vmp_apply(&mut tmp_dft_i, &tmp_a, pmat, scratch2);
                    } else {
                        module.vmp_apply_add(&mut tmp_dft_i, &tmp_a, pmat, di, scratch2);
                    }
                });
            });
        }

        // Adds -(sum a[i] * s[i]) + m)  on the i-th column of tmp_idft_i
        //
        // (-(x0s0 + x1s1 + x2s2) + a0s0s0 + a1s0s1 + a2s0s2, x0, x1, x2)
        // +
        // (0, -(a0s0 + a1s1 + a2s2) + M[i], 0, 0)
        // =
        // (-(x0s0 + x1s1 + x2s2) + s0(a0s0 + a1s1 + a2s2), x0 -(a0s0 + a1s1 + a2s2) + M[i], x1, x2)
        // =
        // (-(x0s0 + x1s1 + x2s2), x0 + M[i], x1, x2)
        module.vec_znx_dft_add_inplace(&mut tmp_dft_i, col_j, ci_dft, 0);
        let (mut tmp_idft, scratch2) = scratch1.tmp_vec_znx_big(module, 1, tsk.size());
        (0..cols).for_each(|i| {
            module.vec_znx_dft_to_vec_znx_big_tmp_a(&mut tmp_idft, 0, &mut tmp_dft_i, i);
            module.vec_znx_big_normalize(
                self.basek(),
                &mut self.at_mut(row_i, col_j).data,
                i,
                &tmp_idft,
                0,
                scratch2,
            );
        });
    }

    pub fn keyswitch<DataLhs: AsRef<[u8]>, DataKsk: AsRef<[u8]>, DataTsk: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GGSWCiphertext<DataLhs>,
        ksk: &GLWESwitchingKeyExec<DataKsk, B>,
        tsk: &GLWETensorKeyPrep<DataTsk, B>,
        scratch: &mut Scratch,
    ) where
        Module<B>: VecZnxDftFromVecZnx<B>
            + VecZnxDftAllocBytes
            + VecZnxBigAllocBytes
            + VecZnxBigNormalize<B>
            + VmpApply<B>
            + VecZnxDftCopy<B>
            + VecZnxDftAddInplace<B>
            + VecZnxDftToVecZnxBigTmpA<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxBigSubSmallAInplace<B>
            + VecZnxBigSubSmallBInplace<B>
            + VecZnxBigNormalize<B>,
    {
        let rank: usize = self.rank();
        let cols: usize = rank + 1;

        // Keyswitch the j-th row of the col 0
        (0..lhs.rows()).for_each(|row_i| {
            // Key-switch column 0, i.e.
            // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0, a1, a2) -> (-(a0s0' + a1s1' + a2s2') + M[i], a0, a1, a2)
            self.at_mut(row_i, 0)
                .keyswitch(module, &lhs.at(row_i, 0), ksk, scratch);

            // Pre-compute DFT of (a0, a1, a2)
            let (mut ci_dft, scratch1) = scratch.tmp_vec_znx_dft(module, cols - 1, self.size());
            (1..cols).for_each(|i| {
                module.vec_znx_dft_from_vec_znx(1, 0, &mut ci_dft, i - 1, &self.at(row_i, 0).data, i);
            });
            // Generates
            //
            // col 1: (-(b0s0' + b1s1' + b2s2')    , b0 + M[i], b1       , b2       )
            // col 2: (-(c0s0' + c1s1' + c2s2')    , c0       , c1 + M[i], c2       )
            // col 3: (-(d0s0' + d1s1' + d2s2')    , d0       , d1       , d2 + M[i])
            (1..cols).for_each(|col_j| {
                self.expand_row(module, row_i, col_j, &ci_dft, tsk, scratch1);
            });
        })
    }

    pub fn keyswitch_inplace<DataKsk: AsRef<[u8]>, DataTsk: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        ksk: &GLWESwitchingKeyExec<DataKsk, B>,
        tsk: &GLWETensorKeyPrep<DataTsk, B>,
        scratch: &mut Scratch,
    ) where
        Module<B>: VecZnxDftFromVecZnx<B>
            + VecZnxDftAllocBytes
            + VecZnxBigAllocBytes
            + VecZnxBigNormalize<B>
            + VmpApply<B>
            + VecZnxDftCopy<B>
            + VecZnxDftAddInplace<B>
            + VecZnxDftToVecZnxBigTmpA<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxBigSubSmallAInplace<B>
            + VecZnxBigSubSmallBInplace<B>
            + VecZnxBigNormalize<B>,
    {
        unsafe {
            let self_ptr: *mut GGSWCiphertext<DataSelf> = self as *mut GGSWCiphertext<DataSelf>;
            self.keyswitch(module, &*self_ptr, ksk, tsk, scratch);
        }
    }

    pub fn automorphism<DataLhs: AsRef<[u8]>, DataAk: AsRef<[u8]>, DataTsk: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GGSWCiphertext<DataLhs>,
        auto_key: &GLWEAutomorphismKeyExec<DataAk, B>,
        tensor_key: &GLWETensorKeyPrep<DataTsk, B>,
        scratch: &mut Scratch,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VecZnxBigAllocBytes
            + VecZnxDftFromVecZnx<B>
            + VecZnxDftCopy<B>
            + VecZnxDftToVecZnxBigTmpA<B>
            + VmpApply<B>
            + VecZnxDftAddInplace<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxBigSubSmallAInplace<B>
            + VecZnxBigSubSmallBInplace<B>
            + VecZnxBigNormalize<B>,
    {
        #[cfg(debug_assertions)]
        {
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
        auto_key: &GLWEAutomorphismKeyExec<DataKsk, B>,
        tensor_key: &GLWETensorKeyPrep<DataTsk, B>,
        scratch: &mut Scratch,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VecZnxBigAllocBytes
            + VecZnxDftFromVecZnx<B>
            + VecZnxDftCopy<B>
            + VecZnxDftToVecZnxBigTmpA<B>
            + VmpApply<B>
            + VecZnxDftAddInplace<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxBigSubSmallAInplace<B>
            + VecZnxBigSubSmallBInplace<B>
            + VecZnxBigNormalize<B>,
    {
        unsafe {
            let self_ptr: *mut GGSWCiphertext<DataSelf> = self as *mut GGSWCiphertext<DataSelf>;
            self.automorphism(module, &*self_ptr, auto_key, tensor_key, scratch);
        }
    }

    pub fn external_product<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GGSWCiphertext<DataLhs>,
        rhs: &GGSWCiphertextExec<DataRhs, B>,
        scratch: &mut Scratch,
    ) where
        Module<B>:
            VecZnxDftAllocBytes + VmpApply<B> + VecZnxDftFromVecZnx<B> + VecZnxDftToVecZnxBigConsume<B> + VecZnxBigNormalize<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.rank(),
                lhs.rank(),
                "ggsw_out rank: {} != ggsw_in rank: {}",
                self.rank(),
                lhs.rank()
            );
            assert_eq!(
                self.rank(),
                rhs.rank(),
                "ggsw_in rank: {} != ggsw_apply rank: {}",
                self.rank(),
                rhs.rank()
            );

            assert!(
                scratch.available()
                    >= GGSWCiphertext::external_product_scratch_space(
                        module,
                        self.basek(),
                        self.k(),
                        lhs.k(),
                        rhs.k(),
                        rhs.digits(),
                        rhs.rank()
                    )
            )
        }

        (0..self.rank() + 1).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                self.at_mut(row_j, col_i)
                    .external_product(module, &lhs.at(row_j, col_i), rhs, scratch);
            });
        });

        (self.rows().min(lhs.rows())..self.rows()).for_each(|row_i| {
            (0..self.rank() + 1).for_each(|col_j| {
                self.at_mut(row_i, col_j).data.zero();
            });
        });
    }

    pub fn external_product_inplace<DataRhs: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        rhs: &GGSWCiphertextExec<DataRhs, B>,
        scratch: &mut Scratch,
    ) where
        Module<B>:
            VecZnxDftAllocBytes + VmpApply<B> + VecZnxDftFromVecZnx<B> + VecZnxDftToVecZnxBigConsume<B> + VecZnxBigNormalize<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.rank(),
                rhs.rank(),
                "ggsw_out rank: {} != ggsw_apply: {}",
                self.rank(),
                rhs.rank()
            );
        }

        (0..self.rank() + 1).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                self.at_mut(row_j, col_i)
                    .external_product_inplace(module, rhs, scratch);
            });
        });
    }
}

pub struct GGSWCiphertextExec<C, B: Backend> {
    pub(crate) data: VmpPMat<C, B>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) digits: usize,
}

impl<B: Backend> GGSWCiphertextExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        let size: usize = k.div_ceil(basek);
        debug_assert!(digits > 0, "invalid ggsw: `digits` == 0");

        debug_assert!(
            size > digits,
            "invalid ggsw: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid ggsw: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
        );

        Self {
            data: module.vmp_pmat_alloc(rows, rank + 1, rank + 1, k.div_ceil(basek)),
            basek,
            k: k,
            digits,
        }
    }

    pub fn bytes_of(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        let size: usize = k.div_ceil(basek);
        debug_assert!(
            size > digits,
            "invalid ggsw: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid ggsw: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
        );

        module.vmp_pmat_alloc_bytes(rows, rank + 1, rank + 1, size)
    }
}

impl<T, B: Backend> Infos for GGSWCiphertextExec<T, B> {
    type Inner = VmpPMat<T, B>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.basek
    }

    fn k(&self) -> usize {
        self.k
    }
}

impl<T, B: Backend> GGSWCiphertextExec<T, B> {
    pub fn rank(&self) -> usize {
        self.data.cols_out() - 1
    }

    pub fn digits(&self) -> usize {
        self.digits
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>, B: Backend> GGSWCiphertextExec<DataSelf, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &GGLWECiphertext<DataOther>, scratch: &mut Scratch)
    where
        DataOther: AsRef<[u8]>,
        Module<B>: VmpPMatPrepare<B>,
    {
        module.vmp_prepare(&mut self.data, &other.data, scratch);
        self.k = other.k;
        self.basek = other.basek;
        self.digits = other.digits;
    }
}