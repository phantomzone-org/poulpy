macro_rules! hal_impl_vmp_ntt120 {
    () => {
        fn vmp_prepare_tmp_bytes(module: &Module<Self>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
            <Self as NTT120VmpDefaults<Self>>::vmp_prepare_tmp_bytes_default(module, rows, cols_in, cols_out, size)
        }

        fn vmp_prepare<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut Scratch<Self>)
        where
            R: VmpPMatToMut<Self>,
            A: MatZnxToRef,
        {
            <Self as NTT120VmpDefaults<Self>>::vmp_prepare_default(module, res, a, scratch)
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
            <Self as NTT120VmpDefaults<Self>>::vmp_apply_dft_to_dft_tmp_bytes_default(
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
            <Self as NTT120VmpDefaults<Self>>::vmp_apply_dft_to_dft_default(module, res, a, b, limb_offset, scratch)
        }

        fn vmp_zero<R>(module: &Module<Self>, res: &mut R)
        where
            R: VmpPMatToMut<Self>,
        {
            <Self as NTT120VmpDefaults<Self>>::vmp_zero_default(module, res)
        }
    };
}
