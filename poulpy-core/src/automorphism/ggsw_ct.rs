use poulpy_hal::{
    api::{
        ScratchAvailable, VecZnxAutomorphismInplace, VecZnxBigAddSmallInplace, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftAddInplace, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxIdftApplyConsume,
        VecZnxIdftApplyTmpA, VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftAdd,
        VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::layouts::{
    GGLWEInfos, GGSW, GGSWInfos, GLWE,
    prepared::{AutomorphismKeyPrepared, TensorKeyPrepared},
};

impl GGSW<Vec<u8>> {
    pub fn automorphism_scratch_space<B: Backend, OUT, IN, KEY, TSK>(
        module: &Module<B>,
        out_infos: &OUT,
        in_infos: &IN,
        key_infos: &KEY,
        tsk_infos: &TSK,
    ) -> usize
    where
        OUT: GGSWInfos,
        IN: GGSWInfos,
        KEY: GGLWEInfos,
        TSK: GGLWEInfos,
        Module<B>:
            VecZnxDftBytesOf + VmpApplyDftToDftTmpBytes + VecZnxBigBytesOf + VecZnxNormalizeTmpBytes + VecZnxBigNormalizeTmpBytes,
    {
        let out_size: usize = out_infos.size();
        let ci_dft: usize = module.bytes_of_vec_znx_dft((key_infos.rank_out() + 1).into(), out_size);
        let ks_internal: usize = GLWE::keyswitch_scratch_space(
            module,
            &out_infos.glwe_layout(),
            &in_infos.glwe_layout(),
            key_infos,
        );
        let expand: usize = GGSW::expand_row_scratch_space(module, out_infos, tsk_infos);
        ci_dft + (ks_internal | expand)
    }

    pub fn automorphism_inplace_scratch_space<B: Backend, OUT, KEY, TSK>(
        module: &Module<B>,
        out_infos: &OUT,
        key_infos: &KEY,
        tsk_infos: &TSK,
    ) -> usize
    where
        OUT: GGSWInfos,
        KEY: GGLWEInfos,
        TSK: GGLWEInfos,
        Module<B>:
            VecZnxDftBytesOf + VmpApplyDftToDftTmpBytes + VecZnxBigBytesOf + VecZnxNormalizeTmpBytes + VecZnxBigNormalizeTmpBytes,
    {
        GGSW::automorphism_scratch_space(module, out_infos, out_infos, key_infos, tsk_infos)
    }
}

impl<DataSelf: DataMut> GGSW<DataSelf> {
    pub fn automorphism<DataLhs: DataRef, DataAk: DataRef, DataTsk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        lhs: &GGSW<DataLhs>,
        auto_key: &AutomorphismKeyPrepared<DataAk, B>,
        tensor_key: &TensorKeyPrepared<DataTsk, B>,
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
            + VecZnxAutomorphismInplace<B>
            + VecZnxBigBytesOf
            + VecZnxNormalizeTmpBytes
            + VecZnxDftCopy<B>
            + VecZnxDftAddInplace<B>
            + VecZnxIdftApplyTmpA<B>
            + VecZnxNormalize<B>,
        Scratch<B>: ScratchAvailable,
    {
        #[cfg(debug_assertions)]
        {
            use crate::layouts::{GLWEInfos, LWEInfos};

            assert_eq!(self.n(), module.n() as u32);
            assert_eq!(lhs.n(), module.n() as u32);
            assert_eq!(auto_key.n(), module.n() as u32);
            assert_eq!(tensor_key.n(), module.n() as u32);

            assert_eq!(
                self.rank(),
                lhs.rank(),
                "ggsw_out rank: {} != ggsw_in rank: {}",
                self.rank(),
                lhs.rank()
            );
            assert_eq!(
                self.rank(),
                auto_key.rank_out(),
                "ggsw_in rank: {} != auto_key rank: {}",
                self.rank(),
                auto_key.rank_out()
            );
            assert_eq!(
                self.rank(),
                tensor_key.rank_out(),
                "ggsw_in rank: {} != tensor_key rank: {}",
                self.rank(),
                tensor_key.rank_out()
            );
            assert!(scratch.available() >= GGSW::automorphism_scratch_space(module, self, lhs, auto_key, tensor_key))
        };

        // Keyswitch the j-th row of the col 0
        (0..lhs.dnum().into()).for_each(|row_i| {
            // Key-switch column 0, i.e.
            // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0, a1, a2) -> (-(a0pi^-1(s0) + a1pi^-1(s1) + a2pi^-1(s2)) + M[i], a0, a1, a2)
            self.at_mut(row_i, 0)
                .automorphism(module, &lhs.at(row_i, 0), auto_key, scratch);
        });
        self.expand_row(module, tensor_key, scratch);
    }

    pub fn automorphism_inplace<DataKsk: DataRef, DataTsk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        auto_key: &AutomorphismKeyPrepared<DataKsk, B>,
        tensor_key: &TensorKeyPrepared<DataTsk, B>,
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
            + VecZnxAutomorphismInplace<B>
            + VecZnxBigBytesOf
            + VecZnxNormalizeTmpBytes
            + VecZnxDftCopy<B>
            + VecZnxDftAddInplace<B>
            + VecZnxIdftApplyTmpA<B>
            + VecZnxNormalize<B>,
        Scratch<B>: ScratchAvailable,
    {
        // Keyswitch the j-th row of the col 0
        (0..self.dnum().into()).for_each(|row_i| {
            // Key-switch column 0, i.e.
            // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0, a1, a2) -> (-(a0pi^-1(s0) + a1pi^-1(s1) + a2pi^-1(s2)) + M[i], a0, a1, a2)
            self.at_mut(row_i, 0)
                .automorphism_inplace(module, auto_key, scratch);
        });
        self.expand_row(module, tensor_key, scratch);
    }
}
