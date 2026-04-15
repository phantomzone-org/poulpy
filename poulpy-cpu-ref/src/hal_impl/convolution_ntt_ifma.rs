macro_rules! hal_impl_convolution_ntt_ifma {
    () => {
        fn cnv_prepare_left_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
            <Self as NTTIfmaConvolutionDefaults<Self>>::cnv_prepare_left_tmp_bytes_default(module, res_size, a_size)
        }

        fn cnv_prepare_left<R, A>(module: &Module<Self>, res: &mut R, a: &A, mask: i64, scratch: &mut Scratch<Self>)
        where
            R: CnvPVecLToMut<Self>,
            A: VecZnxToRef,
        {
            <Self as NTTIfmaConvolutionDefaults<Self>>::cnv_prepare_left_default(module, res, a, mask, scratch)
        }

        fn cnv_prepare_right_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
            <Self as NTTIfmaConvolutionDefaults<Self>>::cnv_prepare_right_tmp_bytes_default(module, res_size, a_size)
        }

        fn cnv_prepare_right<R, A>(module: &Module<Self>, res: &mut R, a: &A, mask: i64, scratch: &mut Scratch<Self>)
        where
            R: CnvPVecRToMut<Self>,
            A: VecZnxToRef + ZnxInfos,
        {
            <Self as NTTIfmaConvolutionDefaults<Self>>::cnv_prepare_right_default(module, res, a, mask, scratch)
        }

        fn cnv_apply_dft_tmp_bytes(
            module: &Module<Self>,
            cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            <Self as NTTIfmaConvolutionDefaults<Self>>::cnv_apply_dft_tmp_bytes_default(
                module, cnv_offset, res_size, a_size, b_size,
            )
        }

        fn cnv_by_const_apply_tmp_bytes(
            module: &Module<Self>,
            cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            <Self as NTTIfmaConvolutionDefaults<Self>>::cnv_by_const_apply_tmp_bytes_default(
                module, cnv_offset, res_size, a_size, b_size,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_by_const_apply<R, A>(
            module: &Module<Self>,
            cnv_offset: usize,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
            b: &[i64],
            scratch: &mut Scratch<Self>,
        ) where
            R: VecZnxBigToMut<Self>,
            A: VecZnxToRef,
        {
            <Self as NTTIfmaConvolutionDefaults<Self>>::cnv_by_const_apply_default(
                module, cnv_offset, res, res_col, a, a_col, b, scratch,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_apply_dft<R, A, B>(
            module: &Module<Self>,
            cnv_offset: usize,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
            b: &B,
            b_col: usize,
            scratch: &mut Scratch<Self>,
        ) where
            R: VecZnxDftToMut<Self>,
            A: CnvPVecLToRef<Self>,
            B: CnvPVecRToRef<Self>,
        {
            <Self as NTTIfmaConvolutionDefaults<Self>>::cnv_apply_dft_default(
                module, cnv_offset, res, res_col, a, a_col, b, b_col, scratch,
            )
        }

        fn cnv_pairwise_apply_dft_tmp_bytes(
            module: &Module<Self>,
            cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            <Self as NTTIfmaConvolutionDefaults<Self>>::cnv_pairwise_apply_dft_tmp_bytes_default(
                module, cnv_offset, res_size, a_size, b_size,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_pairwise_apply_dft<R, A, B>(
            module: &Module<Self>,
            cnv_offset: usize,
            res: &mut R,
            res_col: usize,
            a: &A,
            b: &B,
            i: usize,
            j: usize,
            scratch: &mut Scratch<Self>,
        ) where
            R: VecZnxDftToMut<Self>,
            A: CnvPVecLToRef<Self>,
            B: CnvPVecRToRef<Self>,
        {
            <Self as NTTIfmaConvolutionDefaults<Self>>::cnv_pairwise_apply_dft_default(
                module, cnv_offset, res, res_col, a, b, i, j, scratch,
            )
        }

        fn cnv_prepare_self_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
            <Self as NTTIfmaConvolutionDefaults<Self>>::cnv_prepare_self_tmp_bytes_default(module, res_size, a_size)
        }

        fn cnv_prepare_self<L, R, A>(
            module: &Module<Self>,
            left: &mut L,
            right: &mut R,
            a: &A,
            mask: i64,
            scratch: &mut Scratch<Self>,
        ) where
            L: CnvPVecLToMut<Self>,
            R: CnvPVecRToMut<Self>,
            A: VecZnxToRef + ZnxInfos,
        {
            <Self as NTTIfmaConvolutionDefaults<Self>>::cnv_prepare_self_default(module, left, right, a, mask, scratch)
        }
    };
}
