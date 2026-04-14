macro_rules! hal_impl_vmp_ntt_ifma {
    () => {
        fn vmp_prepare_tmp_bytes(module: &Module<Self>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
            <Self as NTTIfmaVmpDefaults<Self>>::vmp_prepare_tmp_bytes_default(module, rows, cols_in, cols_out, size)
        }

        fn vmp_prepare<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut Scratch<Self>)
        where
            R: VmpPMatToMut<Self>,
            A: MatZnxToRef,
        {
            <Self as NTTIfmaVmpDefaults<Self>>::vmp_prepare_default(module, res, a, scratch)
        }

        fn vmp_apply_dft_to_dft_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_size: usize,
            b_rows: usize,
            b_cols_in: usize,
            b_cols_out: usize,
            b_size: usize,
        ) -> usize {
            <Self as NTTIfmaVmpDefaults<Self>>::vmp_apply_dft_to_dft_tmp_bytes_default(
                module, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size,
            )
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
            use poulpy_cpu_ref::reference::ntt_ifma::vmp::ntt_ifma_vmp_apply_dft_to_dft_tmp_bytes;
            use poulpy_hal::api::TakeSlice;
            let a_ref: VecZnxDft<&[u8], Self> = a.to_ref();
            let b_ref: VmpPMat<&[u8], Self> = b.to_ref();
            let bytes = ntt_ifma_vmp_apply_dft_to_dft_tmp_bytes(a_ref.size(), b_ref.rows(), b_ref.cols_in());
            let (tmp, _) = scratch.take_slice::<u64>(bytes / std::mem::size_of::<u64>());
            crate::ntt_ifma::vmp::vmp_apply_dft_to_dft_ifma(module, res, a, b, limb_offset, tmp)
        }

        fn vmp_zero<R>(module: &Module<Self>, res: &mut R)
        where
            R: VmpPMatToMut<Self>,
        {
            <Self as NTTIfmaVmpDefaults<Self>>::vmp_zero_default(module, res)
        }
    };
}
