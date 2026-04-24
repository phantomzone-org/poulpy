#[macro_export]
macro_rules! hal_impl_svp {
    ($defaults:ident) => {
        fn svp_prepare<A>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::SvpPPolBackendMut<'_, Self>,
            res_col: usize,
            a: &A,
            a_col: usize,
        ) where
            A: ScalarZnxToRef,
        {
            <Self as $defaults<Self>>::svp_prepare_default(module, res, res_col, a, a_col)
        }

        fn svp_apply_dft<R, C>(
            module: &Module<Self>,
            res: &mut R,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpPPolBackendRef<'_, Self>,
            a_col: usize,
            b: &C,
            b_col: usize,
        ) where
            R: VecZnxDftToMut<Self>,
            C: VecZnxToRef,
        {
            <Self as $defaults<Self>>::svp_apply_dft_default(module, res, res_col, a, a_col, b, b_col)
        }

        fn svp_apply_dft_to_dft<R, C>(
            module: &Module<Self>,
            res: &mut R,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpPPolBackendRef<'_, Self>,
            a_col: usize,
            b: &C,
            b_col: usize,
        ) where
            R: VecZnxDftToMut<Self>,
            C: VecZnxDftToRef<Self>,
        {
            <Self as $defaults<Self>>::svp_apply_dft_to_dft_default(module, res, res_col, a, a_col, b, b_col)
        }

        fn svp_apply_dft_to_dft_inplace<R>(
            module: &Module<Self>,
            res: &mut R,
            res_col: usize,
            a: &poulpy_hal::layouts::SvpPPolBackendRef<'_, Self>,
            a_col: usize,
        ) where
            R: VecZnxDftToMut<Self>,
        {
            <Self as $defaults<Self>>::svp_apply_dft_to_dft_inplace_default(module, res, res_col, a, a_col)
        }
    };
}
