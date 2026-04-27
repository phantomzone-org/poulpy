macro_rules! hal_impl_family_common {
    () => {
        fn vmp_apply_dft_tmp_bytes(
            module: &Module<Self>,
            res_size: usize,
            a_size: usize,
            b_rows: usize,
            b_cols_in: usize,
            b_cols_out: usize,
            b_size: usize,
        ) -> usize {
            let a_dft_size = a_size.min(b_rows);
            <Self as Backend>::bytes_of_vec_znx_dft(module.n(), b_cols_in, a_dft_size)
                + Self::vmp_apply_dft_to_dft_tmp_bytes(module, res_size, a_dft_size, b_rows, b_cols_in, b_cols_out, b_size)
        }

        fn vmp_apply_dft<R, A, C>(module: &Module<Self>, res: &mut R, a: &A, b: &C, scratch: &mut Scratch<Self>)
        where
            R: VecZnxDftToMut<Self>,
            A: VecZnxToRef,
            C: VmpPMatToRef<Self>,
        {
            let a = a.to_ref();
            let b = b.to_ref();

            let a_cols = <VecZnx<&[u8]> as ZnxInfos>::cols(&a);
            let a_size = <VecZnx<&[u8]> as ZnxInfos>::size(&a);
            let b_rows = <VmpPMat<&[u8], Self> as ZnxInfos>::rows(&b);
            let cols_to_copy = a_cols.min(b.cols_in());
            let a_start_col = a_cols - cols_to_copy;
            let a_dft_size = a_size.min(b_rows);
            let offset = b.cols_in() - cols_to_copy;

            let (mut a_dft, scratch) =
                <Scratch<Self> as ScratchTakeBasic>::take_vec_znx_dft(scratch, module, b.cols_in(), a_dft_size);

            for j in 0..offset {
                <Module<Self> as VecZnxDftZero<Self>>::vec_znx_dft_zero(module, &mut a_dft, j);
            }

            for j in 0..cols_to_copy {
                <Module<Self> as VecZnxDftApply<Self>>::vec_znx_dft_apply(
                    module,
                    1,
                    0,
                    &mut a_dft,
                    offset + j,
                    &a,
                    a_start_col + j,
                );
            }

            <Module<Self> as VmpApplyDftToDft<Self>>::vmp_apply_dft_to_dft(module, res, &a_dft, &b, 0, scratch)
        }
    };
}
