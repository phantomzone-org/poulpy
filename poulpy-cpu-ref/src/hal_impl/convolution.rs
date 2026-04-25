#[macro_export]
macro_rules! hal_impl_convolution {
    ($defaults:ident) => {
        fn cnv_prepare_left_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
            <Self as $defaults<Self>>::cnv_prepare_left_tmp_bytes_default(module, res_size, a_size)
        }

        fn cnv_prepare_left<'s, R>(
            module: &Module<Self>,
            res: &mut R,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            mask: i64,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) where
            R: CnvPVecLToMut<Self>,
        {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::cnv_prepare_left_default(module, res, a, mask, &mut scratch);
        }

        fn cnv_prepare_right_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
            <Self as $defaults<Self>>::cnv_prepare_right_tmp_bytes_default(module, res_size, a_size)
        }

        fn cnv_prepare_right<'s, R>(
            module: &Module<Self>,
            res: &mut R,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            mask: i64,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) where
            R: CnvPVecRToMut<Self>,
        {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::cnv_prepare_right_default(module, res, a, mask, &mut scratch);
        }

        fn cnv_apply_dft_tmp_bytes(
            module: &Module<Self>,
            cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            <Self as $defaults<Self>>::cnv_apply_dft_tmp_bytes_default(module, cnv_offset, res_size, a_size, b_size)
        }

        fn cnv_by_const_apply_tmp_bytes(
            module: &Module<Self>,
            cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            <Self as $defaults<Self>>::cnv_by_const_apply_tmp_bytes_default(module, cnv_offset, res_size, a_size, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_by_const_apply<'s, R>(
            module: &Module<Self>,
            cnv_offset: usize,
            res: &mut R,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            a_col: usize,
            b: &[i64],
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) where
            R: VecZnxBigToMut<Self>,
        {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::cnv_by_const_apply_default(module, cnv_offset, res, res_col, a, a_col, b, &mut scratch);
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_apply_dft<'s, R>(
            module: &Module<Self>,
            cnv_offset: usize,
            res: &mut R,
            res_col: usize,
            a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
            b_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) where
            R: VecZnxDftToMut<Self>,
        {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::cnv_apply_dft_default(module, cnv_offset, res, res_col, a, a_col, b, b_col, &mut scratch);
        }

        fn cnv_pairwise_apply_dft_tmp_bytes(
            module: &Module<Self>,
            cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            <Self as $defaults<Self>>::cnv_pairwise_apply_dft_tmp_bytes_default(module, cnv_offset, res_size, a_size, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_pairwise_apply_dft<'s, R>(
            module: &Module<Self>,
            cnv_offset: usize,
            res: &mut R,
            res_col: usize,
            a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
            i: usize,
            j: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) where
            R: VecZnxDftToMut<Self>,
        {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::cnv_pairwise_apply_dft_default(module, cnv_offset, res, res_col, a, b, i, j, &mut scratch);
        }

        fn cnv_prepare_self_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
            <Self as $defaults<Self>>::cnv_prepare_self_tmp_bytes_default(module, res_size, a_size)
        }

        fn cnv_prepare_self<'s, L, R>(
            module: &Module<Self>,
            left: &mut L,
            right: &mut R,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            mask: i64,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) where
            L: CnvPVecLToMut<Self>,
            R: CnvPVecRToMut<Self>,
        {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::cnv_prepare_self_default(module, left, right, a, mask, &mut scratch);
        }
    };
}
