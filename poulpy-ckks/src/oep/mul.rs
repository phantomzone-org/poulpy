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

        fn ckks_mul_pt_vec_rnx_tmp_bytes<R: poulpy_core::layouts::GLWEInfos, A: poulpy_core::layouts::GLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            res: &R,
            a: &A,
            b: &$crate::CKKSMeta,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::ModuleN + poulpy_core::GLWEMulPlain<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_vec_rnx_tmp_bytes_default(module, res, a, b)
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

        fn ckks_mul_into<Dst: poulpy_hal::layouts::Data, A: poulpy_hal::layouts::Data, B: poulpy_hal::layouts::Data, T: poulpy_hal::layouts::Data>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<Dst>,
            a: &$crate::layouts::CKKSCiphertext<A>,
            b: &$crate::layouts::CKKSCiphertext<B>,
            tsk: &poulpy_core::layouts::GLWETensorKeyPrepared<T, $backend>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>:
                poulpy_core::GLWECopy<$backend> + poulpy_core::GLWETensoring<$backend> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = <$backend as poulpy_hal::layouts::Backend>::OwnedBuf>,
            $crate::layouts::CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<$backend>,
            $crate::layouts::CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<$backend> + poulpy_core::layouts::GLWEInfos,
            $crate::layouts::CKKSCiphertext<B>: poulpy_core::layouts::GLWEToBackendRef<$backend> + poulpy_core::layouts::GLWEInfos,
            poulpy_core::layouts::GLWETensorKeyPrepared<T, $backend>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_into_default(module, dst, a, b, tsk, scratch)
        }

        fn ckks_mul_assign<Dst: poulpy_hal::layouts::Data, A: poulpy_hal::layouts::Data, T: poulpy_hal::layouts::Data>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<Dst>,
            a: &$crate::layouts::CKKSCiphertext<A>,
            tsk: &poulpy_core::layouts::GLWETensorKeyPrepared<T, $backend>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>:
                poulpy_core::GLWECopy<$backend> + poulpy_core::GLWETensoring<$backend> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = <$backend as poulpy_hal::layouts::Backend>::OwnedBuf>,
            $crate::layouts::CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<$backend> + poulpy_core::layouts::GLWEToBackendRef<$backend> + poulpy_core::layouts::GLWEInfos,
            $crate::layouts::CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<$backend> + poulpy_core::layouts::GLWEInfos,
            poulpy_core::layouts::GLWETensorKeyPrepared<T, $backend>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_assign_default(module, dst, a, tsk, scratch)
        }

        fn ckks_square_into<Dst: poulpy_hal::layouts::Data, A: poulpy_hal::layouts::Data, T: poulpy_hal::layouts::Data>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<Dst>,
            a: &$crate::layouts::CKKSCiphertext<A>,
            tsk: &poulpy_core::layouts::GLWETensorKeyPrepared<T, $backend>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>:
                poulpy_core::GLWECopy<$backend> + poulpy_core::GLWETensoring<$backend> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = <$backend as poulpy_hal::layouts::Backend>::OwnedBuf>,
            $crate::layouts::CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<$backend>,
            $crate::layouts::CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<$backend> + poulpy_core::layouts::GLWEInfos,
            poulpy_core::layouts::GLWETensorKeyPrepared<T, $backend>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_square_into_default(module, dst, a, tsk, scratch)
        }

        fn ckks_square_assign<Dst: poulpy_hal::layouts::Data, T: poulpy_hal::layouts::Data>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<Dst>,
            tsk: &poulpy_core::layouts::GLWETensorKeyPrepared<T, $backend>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>:
                poulpy_core::GLWECopy<$backend> + poulpy_core::GLWETensoring<$backend> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = <$backend as poulpy_hal::layouts::Backend>::OwnedBuf>,
            $crate::layouts::CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<$backend> + poulpy_core::layouts::GLWEToBackendRef<$backend> + poulpy_core::layouts::GLWEInfos,
            poulpy_core::layouts::GLWETensorKeyPrepared<T, $backend>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_square_assign_default(module, dst, tsk, scratch)
        }

        fn ckks_mul_pt_vec_znx_into<Dst: poulpy_hal::layouts::Data, A: poulpy_hal::layouts::Data, P: poulpy_hal::layouts::Data>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<Dst>,
            a: &$crate::layouts::CKKSCiphertext<A>,
            pt_znx: &$crate::layouts::plaintext::CKKSPlaintextVecZnx<P>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>:
                poulpy_core::GLWECopy<$backend>
                + poulpy_core::GLWEMulPlain<$backend>
                + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = <$backend as poulpy_hal::layouts::Backend>::OwnedBuf>
                + poulpy_hal::api::VecZnxCopyBackend<$backend>,
            $crate::layouts::CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<$backend>,
            $crate::layouts::CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<$backend> + poulpy_core::layouts::GLWEInfos,
            $crate::layouts::plaintext::CKKSPlaintextVecZnx<P>: poulpy_core::layouts::GLWEPlaintextToBackendRef<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_vec_znx_into_default(module, dst, a, pt_znx, scratch)
        }

        fn ckks_mul_pt_vec_znx_assign<Dst: poulpy_hal::layouts::Data, P: poulpy_hal::layouts::Data>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<Dst>,
            pt_znx: &$crate::layouts::plaintext::CKKSPlaintextVecZnx<P>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>:
                poulpy_core::GLWECopy<$backend>
                + poulpy_core::GLWEMulPlain<$backend>
                + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = <$backend as poulpy_hal::layouts::Backend>::OwnedBuf>
                + poulpy_hal::api::VecZnxCopyBackend<$backend>,
            $crate::layouts::CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<$backend> + poulpy_core::layouts::GLWEToBackendRef<$backend> + poulpy_core::layouts::GLWEInfos,
            $crate::layouts::plaintext::CKKSPlaintextVecZnx<P>: poulpy_core::layouts::GLWEPlaintextToBackendRef<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_vec_znx_assign_default(module, dst, pt_znx, scratch)
        }

        fn ckks_mul_pt_vec_rnx_into<Dst: poulpy_hal::layouts::Data, A: poulpy_hal::layouts::Data, F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<Dst>,
            a: &$crate::layouts::CKKSCiphertext<A>,
            pt_rnx: &$crate::layouts::plaintext::CKKSPlaintextVecRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            $backend: poulpy_hal::layouts::HostBackend<OwnedBuf = Vec<u8>>,
            poulpy_hal::layouts::Module<$backend>:
                poulpy_core::GLWECopy<$backend>
                + poulpy_core::GLWEMulPlain<$backend>
                + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = <$backend as poulpy_hal::layouts::Backend>::OwnedBuf>
                + poulpy_hal::api::ModuleN
                + poulpy_hal::api::VecZnxCopyBackend<$backend>,
            $crate::layouts::CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<$backend>,
            $crate::layouts::CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<$backend> + poulpy_core::layouts::GLWEInfos,
            $crate::layouts::plaintext::CKKSPlaintextVecRnx<F>: $crate::layouts::plaintext::CKKSPlaintextConversion,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_vec_rnx_into_default(module, dst, a, pt_rnx, prec, scratch)
        }

        fn ckks_mul_pt_vec_rnx_assign<Dst: poulpy_hal::layouts::Data, F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<Dst>,
            pt_rnx: &$crate::layouts::plaintext::CKKSPlaintextVecRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            $backend: poulpy_hal::layouts::HostBackend<OwnedBuf = Vec<u8>>,
            poulpy_hal::layouts::Module<$backend>:
                poulpy_core::GLWECopy<$backend>
                + poulpy_core::GLWEMulPlain<$backend>
                + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = <$backend as poulpy_hal::layouts::Backend>::OwnedBuf>
                + poulpy_hal::api::ModuleN
                + poulpy_hal::api::VecZnxCopyBackend<$backend>,
            $crate::layouts::CKKSCiphertext<Dst>:
                poulpy_core::layouts::GLWEToBackendMut<$backend>
                + poulpy_core::layouts::GLWEToBackendRef<$backend>
                + poulpy_core::layouts::GLWEInfos,
            $crate::layouts::plaintext::CKKSPlaintextVecRnx<F>: $crate::layouts::plaintext::CKKSPlaintextConversion,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_vec_rnx_assign_default(module, dst, pt_rnx, prec, scratch)
        }

        fn ckks_mul_pt_const_znx_into<Dst: poulpy_hal::layouts::Data, A: poulpy_hal::layouts::Data>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<Dst>,
            a: &$crate::layouts::CKKSCiphertext<A>,
            cst_znx: &$crate::layouts::plaintext::CKKSPlaintextCstZnx,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>:
                poulpy_core::GLWEAdd<$backend>
                + poulpy_core::GLWECopy<$backend>
                + poulpy_core::GLWEMulConst<$backend>
                + poulpy_core::GLWERotate<$backend>
                + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = <$backend as poulpy_hal::layouts::Backend>::OwnedBuf>,
            $crate::layouts::CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<$backend>,
            $crate::layouts::CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<$backend> + poulpy_core::layouts::GLWEInfos,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_const_znx_into_default(module, dst, a, cst_znx, scratch)
        }

        fn ckks_mul_pt_const_znx_assign<Dst: poulpy_hal::layouts::Data>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<Dst>,
            cst_znx: &$crate::layouts::plaintext::CKKSPlaintextCstZnx,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>:
                poulpy_core::GLWEAdd<$backend>
                + poulpy_core::GLWECopy<$backend>
                + poulpy_core::GLWEMulConst<$backend>
                + poulpy_core::GLWERotate<$backend>
                + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = <$backend as poulpy_hal::layouts::Backend>::OwnedBuf>,
            $crate::layouts::CKKSCiphertext<Dst>:
                poulpy_core::layouts::GLWEToBackendMut<$backend>
                + poulpy_core::layouts::GLWEToBackendRef<$backend>
                + poulpy_core::layouts::GLWEInfos,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_const_znx_assign_default(module, dst, cst_znx, scratch)
        }

        fn ckks_mul_pt_const_rnx_into<Dst: poulpy_hal::layouts::Data, A: poulpy_hal::layouts::Data, F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<Dst>,
            a: &$crate::layouts::CKKSCiphertext<A>,
            cst_rnx: &$crate::layouts::plaintext::CKKSPlaintextCstRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>:
                poulpy_core::GLWEAdd<$backend>
                + poulpy_core::GLWECopy<$backend>
                + poulpy_core::GLWEMulConst<$backend>
                + poulpy_core::GLWERotate<$backend>
                + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = <$backend as poulpy_hal::layouts::Backend>::OwnedBuf>,
            $crate::layouts::CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<$backend>,
            $crate::layouts::CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<$backend> + poulpy_core::layouts::GLWEInfos,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
            $crate::layouts::plaintext::CKKSPlaintextCstRnx<F>: $crate::layouts::plaintext::CKKSConstPlaintextConversion,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_const_rnx_into_default(module, dst, a, cst_rnx, prec, scratch)
        }

        fn ckks_mul_pt_const_rnx_assign<Dst: poulpy_hal::layouts::Data, F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<Dst>,
            cst_rnx: &$crate::layouts::plaintext::CKKSPlaintextCstRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>:
                poulpy_core::GLWEAdd<$backend>
                + poulpy_core::GLWECopy<$backend>
                + poulpy_core::GLWEMulConst<$backend>
                + poulpy_core::GLWERotate<$backend>
                + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = <$backend as poulpy_hal::layouts::Backend>::OwnedBuf>,
            $crate::layouts::CKKSCiphertext<Dst>:
                poulpy_core::layouts::GLWEToBackendMut<$backend>
                + poulpy_core::layouts::GLWEToBackendRef<$backend>
                + poulpy_core::layouts::GLWEInfos,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
            $crate::layouts::plaintext::CKKSPlaintextCstRnx<F>: $crate::layouts::plaintext::CKKSConstPlaintextConversion,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_const_rnx_assign_default(module, dst, cst_rnx, prec, scratch)
        }
    };
}

pub use crate::impl_ckks_mul_default_methods;
