macro_rules! hal_impl_vmp_ntt120 {
    () => {
        fn vmp_prepare_tmp_bytes(module: &Module<Self>, _rows: usize, _cols_in: usize, _cols_out: usize, _size: usize) -> usize {
            crate::ntt120::vmp::vmp_prepare_tmp_bytes_avx(module.n())
        }

        fn vmp_prepare<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut Scratch<Self>)
        where
            R: VmpPMatToMut<Self>,
            A: MatZnxToRef,
        {
            use poulpy_hal::api::TakeSlice;
            let bytes = crate::ntt120::vmp::vmp_prepare_tmp_bytes_avx(module.n());
            let (tmp, _) = scratch.take_slice::<u64>(bytes / std::mem::size_of::<u64>());
            crate::ntt120::vmp::vmp_prepare_avx_pm(module, res, a, tmp)
        }

        fn vmp_apply_dft_to_dft_tmp_bytes(
            _module: &Module<Self>,
            _res_size: usize,
            a_size: usize,
            b_rows: usize,
            b_cols_in: usize,
            _b_cols_out: usize,
            _b_size: usize,
        ) -> usize {
            crate::ntt120::vmp::vmp_apply_tmp_bytes_avx(a_size, b_rows, b_cols_in)
        }

        fn vmp_apply_dft_to_dft_accumulate_tmp_bytes(
            _module: &Module<Self>,
            _res_size: usize,
            a_size: usize,
            b_rows: usize,
            b_cols_in: usize,
            _b_cols_out: usize,
            _b_size: usize,
        ) -> usize {
            crate::ntt120::vmp::vmp_apply_tmp_bytes_avx(a_size, b_rows, b_cols_in)
        }

        fn vmp_apply_dft_to_dft<R, A, C>(
            module: &Module<Self>,
            res: &mut R,
            a: &A,
            b: &C,
            limb_offset: usize,
            scratch: &mut Scratch<Self>,
        ) where
            R: VecZnxDftToMut<Self>,
            A: VecZnxDftToRef<Self>,
            C: VmpPMatToRef<Self>,
        {
            use poulpy_hal::api::TakeSlice;
            let a_ref: VecZnxDft<&[u8], Self> = a.to_ref();
            let b_ref: VmpPMat<&[u8], Self> = b.to_ref();
            let bytes = crate::ntt120::vmp::vmp_apply_tmp_bytes_avx(a_ref.size(), b_ref.rows(), b_ref.cols_in());
            let (tmp, _) = scratch.take_slice::<u64>(bytes / std::mem::size_of::<u64>());
            crate::ntt120::vmp::vmp_apply_dft_to_dft_avx(module, res, a, b, limb_offset, tmp)
        }

        fn vmp_apply_dft_to_dft_accumulate<R, A, C>(
            module: &Module<Self>,
            res: &mut R,
            a: &A,
            b: &C,
            limb_offset: usize,
            scratch: &mut Scratch<Self>,
        ) where
            R: VecZnxDftToMut<Self>,
            A: VecZnxDftToRef<Self>,
            C: VmpPMatToRef<Self>,
        {
            use poulpy_hal::api::TakeSlice;
            let a_ref: VecZnxDft<&[u8], Self> = a.to_ref();
            let b_ref: VmpPMat<&[u8], Self> = b.to_ref();
            let bytes = crate::ntt120::vmp::vmp_apply_tmp_bytes_avx(a_ref.size(), b_ref.rows(), b_ref.cols_in());
            let (tmp, _) = scratch.take_slice::<u64>(bytes / std::mem::size_of::<u64>());
            crate::ntt120::vmp::vmp_apply_dft_to_dft_accumulate_avx(module, res, a, b, limb_offset, tmp)
        }

        fn vmp_zero<R>(module: &Module<Self>, res: &mut R)
        where
            R: VmpPMatToMut<Self>,
        {
            <Self as NTT120VmpDefaults<Self>>::vmp_zero_default(module, res)
        }
    };
}
