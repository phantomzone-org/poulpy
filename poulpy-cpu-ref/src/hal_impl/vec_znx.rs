#[macro_export]
macro_rules! hal_impl_vec_znx {
    () => {
        fn vec_znx_zero<R>(module: &Module<Self>, res: &mut R, res_col: usize)
        where
            R: VecZnxToMut,
        {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_zero_default(module, res, res_col)
        }

        fn vec_znx_zero_backend<'r>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_col: usize,
        ) {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_zero_backend_default(module, res, res_col)
        }

        fn vec_znx_normalize_tmp_bytes(module: &Module<Self>) -> usize {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_normalize_tmp_bytes_default(module)
        }

        fn vec_znx_normalize<'s, 'r, 'a>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_base2k: usize,
            res_offset: i64,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'a, Self>,
            a_base2k: usize,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefaults<Self>>::vec_znx_normalize_default(
                module,
                res,
                res_base2k,
                res_offset,
                res_col,
                a,
                a_base2k,
                a_col,
                &mut scratch,
            );
        }

        fn vec_znx_normalize_inplace<'s, A>(
            module: &Module<Self>,
            base2k: usize,
            a: &mut A,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) where
            A: VecZnxToMut,
        {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefaults<Self>>::vec_znx_normalize_inplace_default(module, base2k, a, a_col, &mut scratch);
        }

        fn vec_znx_normalize_inplace_backend<'s, 'r>(
            module: &Module<Self>,
            base2k: usize,
            a: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefaults<Self>>::vec_znx_normalize_inplace_backend_default(module, base2k, a, a_col, &mut scratch);
        }

        fn vec_znx_add_into<R, A, C>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
        where
            R: VecZnxToMut,
            A: VecZnxToRef,
            C: VecZnxToRef,
        {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_add_into_default(module, res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_add_assign<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxToMut,
            A: VecZnxToRef,
        {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_add_assign_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_add_scalar_into<R, A, B>(
            module: &Module<Self>,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
            b: &B,
            b_col: usize,
            b_limb: usize,
        ) where
            R: VecZnxToMut,
            A: ScalarZnxToRef,
            B: VecZnxToRef,
        {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_add_scalar_into_default(module, res, res_col, a, a_col, b, b_col, b_limb)
        }

        fn vec_znx_add_scalar_assign<R, A>(
            module: &Module<Self>,
            res: &mut R,
            res_col: usize,
            res_limb: usize,
            a: &A,
            a_col: usize,
        ) where
            R: VecZnxToMut,
            A: ScalarZnxToRef,
        {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_add_scalar_assign_default(module, res, res_col, res_limb, a, a_col)
        }

        fn vec_znx_sub<R, A, C>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
        where
            R: VecZnxToMut,
            A: VecZnxToRef,
            C: VecZnxToRef,
        {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_sub_default(module, res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_sub_assign<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxToMut,
            A: VecZnxToRef,
        {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_sub_assign_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_sub_negate_assign<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxToMut,
            A: VecZnxToRef,
        {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_sub_negate_assign_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_sub_scalar<R, A, B>(
            module: &Module<Self>,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
            b: &B,
            b_col: usize,
            b_limb: usize,
        ) where
            R: VecZnxToMut,
            A: ScalarZnxToRef,
            B: VecZnxToRef,
        {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_sub_scalar_default(module, res, res_col, a, a_col, b, b_col, b_limb)
        }

        fn vec_znx_sub_scalar_assign<R, A>(
            module: &Module<Self>,
            res: &mut R,
            res_col: usize,
            res_limb: usize,
            a: &A,
            a_col: usize,
        ) where
            R: VecZnxToMut,
            A: ScalarZnxToRef,
        {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_sub_scalar_assign_default(module, res, res_col, res_limb, a, a_col)
        }

        fn vec_znx_negate<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxToMut,
            A: VecZnxToRef,
        {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_negate_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_negate_assign<A>(module: &Module<Self>, a: &mut A, a_col: usize)
        where
            A: VecZnxToMut,
        {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_negate_assign_default(module, a, a_col)
        }

        fn vec_znx_rsh_tmp_bytes(module: &Module<Self>) -> usize {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_rsh_tmp_bytes_default(module)
        }

        fn vec_znx_rsh<'s, R, A>(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) where
            R: VecZnxToMut,
            A: VecZnxToRef,
        {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefaults<Self>>::vec_znx_rsh_default(module, base2k, k, res, res_col, a, a_col, &mut scratch);
        }

        fn vec_znx_rsh_add_into<'s, R, A>(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) where
            R: VecZnxToMut,
            A: VecZnxToRef,
        {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefaults<Self>>::vec_znx_rsh_add_into_default(
                module,
                base2k,
                k,
                res,
                res_col,
                a,
                a_col,
                &mut scratch,
            );
        }

        fn vec_znx_lsh_tmp_bytes(module: &Module<Self>) -> usize {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_lsh_tmp_bytes_default(module)
        }

        fn vec_znx_lsh<'s, R, A>(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) where
            R: VecZnxToMut,
            A: VecZnxToRef,
        {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefaults<Self>>::vec_znx_lsh_default(module, base2k, k, res, res_col, a, a_col, &mut scratch);
        }

        fn vec_znx_lsh_add_into<'s, R, A>(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) where
            R: VecZnxToMut,
            A: VecZnxToRef,
        {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefaults<Self>>::vec_znx_lsh_add_into_default(
                module,
                base2k,
                k,
                res,
                res_col,
                a,
                a_col,
                &mut scratch,
            );
        }

        fn vec_znx_lsh_sub<'s, R, A>(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) where
            R: VecZnxToMut,
            A: VecZnxToRef,
        {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefaults<Self>>::vec_znx_lsh_sub_default(module, base2k, k, res, res_col, a, a_col, &mut scratch);
        }

        fn vec_znx_rsh_sub<'s, R, A>(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) where
            R: VecZnxToMut,
            A: VecZnxToRef,
        {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefaults<Self>>::vec_znx_rsh_sub_default(module, base2k, k, res, res_col, a, a_col, &mut scratch);
        }

        fn vec_znx_rsh_inplace<'s, R>(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            a: &mut R,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) where
            R: VecZnxToMut,
        {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefaults<Self>>::vec_znx_rsh_inplace_default(module, base2k, k, a, a_col, &mut scratch);
        }

        fn vec_znx_lsh_inplace<'s, R>(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            a: &mut R,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) where
            R: VecZnxToMut,
        {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefaults<Self>>::vec_znx_lsh_inplace_default(module, base2k, k, a, a_col, &mut scratch);
        }

        fn vec_znx_rotate<'r, 'a>(
            module: &Module<Self>,
            k: i64,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'a, Self>,
            a_col: usize,
        ) {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_rotate_default(module, k, res, res_col, a, a_col)
        }

        fn vec_znx_rotate_assign_tmp_bytes(module: &Module<Self>) -> usize {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_rotate_assign_tmp_bytes_default(module)
        }

        fn vec_znx_rotate_inplace<'s, 'r>(
            module: &Module<Self>,
            k: i64,
            a: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefaults<Self>>::vec_znx_rotate_inplace_default(module, k, a, a_col, &mut scratch);
        }

        fn vec_znx_automorphism<R, A>(module: &Module<Self>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxToMut,
            A: VecZnxToRef,
        {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_automorphism_default(module, k, res, res_col, a, a_col)
        }

        fn vec_znx_automorphism_assign_tmp_bytes(module: &Module<Self>) -> usize {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_automorphism_assign_tmp_bytes_default(module)
        }

        fn vec_znx_automorphism_inplace<'s, 'r>(
            module: &Module<Self>,
            k: i64,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefaults<Self>>::vec_znx_automorphism_inplace_default(module, k, res, res_col, &mut scratch);
        }

        fn vec_znx_mul_xp_minus_one<R, A>(module: &Module<Self>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxToMut,
            A: VecZnxToRef,
        {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_mul_xp_minus_one_default(module, k, res, res_col, a, a_col)
        }

        fn vec_znx_mul_xp_minus_one_assign_tmp_bytes(module: &Module<Self>) -> usize {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_mul_xp_minus_one_assign_tmp_bytes_default(module)
        }

        fn vec_znx_mul_xp_minus_one_inplace<'s, R>(
            module: &Module<Self>,
            k: i64,
            res: &mut R,
            res_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) where
            R: VecZnxToMut,
        {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefaults<Self>>::vec_znx_mul_xp_minus_one_inplace_default(module, k, res, res_col, &mut scratch);
        }

        fn vec_znx_split_ring_tmp_bytes(module: &Module<Self>) -> usize {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_split_ring_tmp_bytes_default(module)
        }

        fn vec_znx_split_ring<'s, R, A>(
            module: &Module<Self>,
            res: &mut [R],
            res_col: usize,
            a: &A,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) where
            R: VecZnxToMut,
            A: VecZnxToRef,
        {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefaults<Self>>::vec_znx_split_ring_default(module, res, res_col, a, a_col, &mut scratch);
        }

        fn vec_znx_merge_rings_tmp_bytes(module: &Module<Self>) -> usize {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_merge_rings_tmp_bytes_default(module)
        }

        fn vec_znx_merge_rings<'s, R, A>(
            module: &Module<Self>,
            res: &mut R,
            res_col: usize,
            a: &[A],
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) where
            R: VecZnxToMut,
            A: VecZnxToRef,
        {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefaults<Self>>::vec_znx_merge_rings_default(module, res, res_col, a, a_col, &mut scratch);
        }

        fn vec_znx_switch_ring<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxToMut,
            A: VecZnxToRef,
        {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_switch_ring_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_copy<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxToMut,
            A: VecZnxToRef,
        {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_copy_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_fill_uniform<R>(module: &Module<Self>, base2k: usize, res: &mut R, res_col: usize, source: &mut Source)
        where
            R: VecZnxToMut,
        {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_fill_uniform_default(module, base2k, res, res_col, source)
        }

        fn vec_znx_fill_normal<R>(
            module: &Module<Self>,
            res_base2k: usize,
            res: &mut R,
            res_col: usize,
            noise_infos: NoiseInfos,
            source: &mut Source,
        ) where
            R: VecZnxToMut,
        {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_fill_normal_default(module, res_base2k, res, res_col, noise_infos, source)
        }

        fn vec_znx_add_normal<R>(
            module: &Module<Self>,
            res_base2k: usize,
            res: &mut R,
            res_col: usize,
            noise_infos: NoiseInfos,
            source: &mut Source,
        ) where
            R: VecZnxToMut,
        {
            <Self as HalVecZnxDefaults<Self>>::vec_znx_add_normal_default(module, res_base2k, res, res_col, noise_infos, source)
        }
    };
}
