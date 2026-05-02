#[macro_export]
macro_rules! impl_ckks_mul_default_methods {
    ($backend:ty) => {
        fn ckks_mul_tmp_bytes<R: poulpy_core::layouts::GLWEInfos, T: poulpy_core::layouts::GGLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            res: &R,
            tsk: &T,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWETensoring<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_tmp_bytes_default(module, res, tsk)
        }

        fn ckks_square_tmp_bytes<R: poulpy_core::layouts::GLWEInfos, T: poulpy_core::layouts::GGLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            res: &R,
            tsk: &T,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWETensoring<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_square_tmp_bytes_default(module, res, tsk)
        }

        fn ckks_mul_pt_vec_znx_tmp_bytes<R: poulpy_core::layouts::GLWEInfos, A: poulpy_core::layouts::GLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            res: &R,
            a: &A,
            b: &$crate::CKKSMeta,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEMulPlain<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_vec_znx_tmp_bytes_default(module, res, a, b)
        }

        fn ckks_mul_pt_const_tmp_bytes<R: poulpy_core::layouts::GLWEInfos, A: poulpy_core::layouts::GLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            res: &R,
            a: &A,
            b: &$crate::CKKSMeta,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEMulConst<$backend> + poulpy_core::GLWERotate<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_const_tmp_bytes_default(module, res, a, b)
        }

        fn ckks_mul_into<Dst, A, B, T>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            a: &A,
            b: &B,
            tsk: &T,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend>
                + $crate::CKKSInfos
                + $crate::SetCKKSInfos
                + poulpy_core::layouts::GLWEInfos,
            A: $crate::GLWEToBackendRef<$backend> + $crate::CKKSInfos + poulpy_core::layouts::GLWEInfos,
            B: $crate::GLWEToBackendRef<$backend> + $crate::CKKSInfos + poulpy_core::layouts::GLWEInfos,
            T: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<$backend> + poulpy_core::layouts::GGLWEInfos,
            poulpy_hal::layouts::Module<$backend>:
                poulpy_core::GLWECopy<$backend> + poulpy_core::GLWETensoring<$backend> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = <$backend as poulpy_hal::layouts::Backend>::OwnedBuf>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_into_default(module, dst, a, b, tsk, scratch)
        }

        fn ckks_mul_assign<Dst, A, T>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            a: &A,
            tsk: &T,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend>
                + $crate::GLWEToBackendRef<$backend>
                + $crate::CKKSInfos
                + $crate::SetCKKSInfos
                + poulpy_core::layouts::GLWEInfos,
            A: $crate::GLWEToBackendRef<$backend> + $crate::CKKSInfos + poulpy_core::layouts::GLWEInfos,
            T: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<$backend> + poulpy_core::layouts::GGLWEInfos,
            poulpy_hal::layouts::Module<$backend>:
                poulpy_core::GLWECopy<$backend> + poulpy_core::GLWETensoring<$backend> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = <$backend as poulpy_hal::layouts::Backend>::OwnedBuf>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_assign_default(module, dst, a, tsk, scratch)
        }

        fn ckks_square_into<Dst, A, T>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            a: &A,
            tsk: &T,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend>
                + $crate::CKKSInfos
                + $crate::SetCKKSInfos
                + poulpy_core::layouts::GLWEInfos,
            A: $crate::GLWEToBackendRef<$backend> + $crate::CKKSInfos + poulpy_core::layouts::GLWEInfos,
            T: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<$backend> + poulpy_core::layouts::GGLWEInfos,
            poulpy_hal::layouts::Module<$backend>:
                poulpy_core::GLWECopy<$backend> + poulpy_core::GLWETensoring<$backend> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = <$backend as poulpy_hal::layouts::Backend>::OwnedBuf>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_square_into_default(module, dst, a, tsk, scratch)
        }

        fn ckks_square_assign<Dst, T>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            tsk: &T,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend>
                + $crate::GLWEToBackendRef<$backend>
                + $crate::CKKSInfos
                + $crate::SetCKKSInfos
                + poulpy_core::layouts::GLWEInfos,
            T: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<$backend> + poulpy_core::layouts::GGLWEInfos,
            poulpy_hal::layouts::Module<$backend>:
                poulpy_core::GLWECopy<$backend> + poulpy_core::GLWETensoring<$backend> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = <$backend as poulpy_hal::layouts::Backend>::OwnedBuf>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_square_assign_default(module, dst, tsk, scratch)
        }

        fn ckks_mul_pt_vec_znx_into<Dst, A, P>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            a: &A,
            pt_znx: &P,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend>
                + $crate::CKKSInfos
                + $crate::SetCKKSInfos
                + poulpy_core::layouts::GLWEInfos,
            A: $crate::GLWEToBackendRef<$backend> + $crate::CKKSInfos + poulpy_core::layouts::GLWEInfos,
            P: $crate::GLWEToBackendRef<$backend> + poulpy_core::layouts::LWEInfos + poulpy_core::layouts::GLWEInfos + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>:
                poulpy_core::GLWECopy<$backend>
                + poulpy_core::GLWEMulPlain<$backend>
                + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = <$backend as poulpy_hal::layouts::Backend>::OwnedBuf>
                + poulpy_hal::api::VecZnxCopyBackend<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_vec_znx_into_default(module, dst, a, pt_znx, scratch)
        }

        fn ckks_mul_pt_vec_znx_assign<Dst, P>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            pt_znx: &P,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend>
                + $crate::CKKSInfos
                + $crate::SetCKKSInfos
                + poulpy_core::layouts::GLWEInfos,
            P: $crate::GLWEToBackendRef<$backend> + poulpy_core::layouts::LWEInfos + poulpy_core::layouts::GLWEInfos + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>:
                poulpy_core::GLWECopy<$backend>
                + poulpy_core::GLWEMulPlain<$backend>
                + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = <$backend as poulpy_hal::layouts::Backend>::OwnedBuf>
                + poulpy_hal::api::VecZnxCopyBackend<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_vec_znx_assign_default(module, dst, pt_znx, scratch)
        }

        fn ckks_mul_pt_const_znx_into<Dst, A, P>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            a: &A,
            pt_znx: &P,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend>
                + $crate::CKKSInfos
                + $crate::SetCKKSInfos
                + poulpy_core::layouts::GLWEInfos,
            A: $crate::GLWEToBackendRef<$backend> + $crate::CKKSInfos + poulpy_core::layouts::GLWEInfos,
            P: $crate::GLWEToBackendRef<$backend> + poulpy_core::layouts::LWEInfos + poulpy_core::layouts::GLWEInfos + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>:
                poulpy_core::GLWEAdd<$backend>
                + poulpy_core::GLWECopy<$backend>
                + poulpy_core::GLWEMulConst<$backend>
                + poulpy_core::GLWERotate<$backend>
                + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = <$backend as poulpy_hal::layouts::Backend>::OwnedBuf>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_const_znx_into_default(module, dst, a, pt_znx, scratch)
        }

        fn ckks_mul_pt_const_znx_assign<Dst, P>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            pt_znx: &P,
            pt_coeff: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend>
                + $crate::GLWEToBackendRef<$backend>
                + $crate::CKKSInfos
                + $crate::SetCKKSInfos
                + poulpy_core::layouts::GLWEInfos,
            P: $crate::GLWEToBackendRef<$backend> + poulpy_core::layouts::LWEInfos + poulpy_core::layouts::GLWEInfos + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>:
                poulpy_core::GLWEAdd<$backend>
                + poulpy_core::GLWECopy<$backend>
                + poulpy_core::GLWEMulConst<$backend>
                + poulpy_core::GLWERotate<$backend>
                + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = <$backend as poulpy_hal::layouts::Backend>::OwnedBuf>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_const_znx_assign_default(module, dst, pt_znx, pt_coeff, scratch)
        }

    };
}

pub use crate::impl_ckks_mul_default_methods;
