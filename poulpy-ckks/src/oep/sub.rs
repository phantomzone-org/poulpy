#[macro_export]
macro_rules! impl_ckks_sub_default_methods {
    ($backend:ty) => {
        fn ckks_sub_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend> + poulpy_core::GLWENormalize<$backend> + poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_tmp_bytes_default(module)
        }

        fn ckks_sub_pt_vec_znx_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend> + poulpy_core::GLWENormalize<$backend> + poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_vec_znx_tmp_bytes_default(module)
        }

        fn ckks_sub_into<Dst, A, B>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            a: &A,
            b: &B,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos + $crate::SetCKKSInfos,
            A: $crate::GLWEToBackendRef<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos,
            B: $crate::GLWEToBackendRef<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWESub<$backend> + poulpy_core::GLWEShift<$backend> + poulpy_core::GLWENormalize<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_into_default(module, dst, a, b, scratch)
        }

        fn ckks_sub_assign<Dst, A>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            a: &A,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend> + $crate::CKKSInfos + $crate::SetCKKSInfos,
            A: $crate::GLWEToBackendRef<$backend> + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWESub<$backend> + poulpy_core::GLWEShift<$backend> + poulpy_core::GLWENormalize<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_assign_default(module, dst, a, scratch)
        }

        fn ckks_sub_pt_vec_znx_into<Dst, A, P>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            a: &A,
            pt_znx: &P,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos + $crate::SetCKKSInfos,
            A: $crate::GLWEToBackendRef<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos,
            P: $crate::GLWEToBackendRef<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxRshSubBackend<$backend> + poulpy_core::GLWEShift<$backend> + poulpy_core::GLWENormalize<$backend> + $crate::leveled::default::pt_znx::CKKSPlaintextDefault<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_vec_znx_into_default(module, dst, a, pt_znx, scratch)
        }

        fn ckks_sub_pt_vec_znx_assign<Dst, P>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            pt_znx: &P,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos,
            P: $crate::GLWEToBackendRef<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxRshSubBackend<$backend> + poulpy_core::GLWENormalize<$backend> + $crate::leveled::default::pt_znx::CKKSPlaintextDefault<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_vec_znx_assign_default(module, dst, pt_znx, scratch)
        }

        fn ckks_sub_pt_const_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend> + poulpy_core::GLWENormalize<$backend> + poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_const_tmp_bytes_default(module)
        }

        fn ckks_sub_pt_const_znx_into<Dst, A, P>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            a: &A,
            dst_coeff: usize,
            pt_znx: &P,
            pt_coeff: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos + $crate::SetCKKSInfos,
            A: $crate::GLWEToBackendRef<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos,
            P: $crate::GLWEToBackendRef<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend> + poulpy_core::GLWENormalize<$backend> + poulpy_hal::api::VecZnxRshSubCoeffIntoBackend<$backend> + $crate::leveled::default::pt_znx::CKKSPlaintextDefault<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_const_znx_into_default(module, dst, a, dst_coeff, pt_znx, pt_coeff, scratch)
        }

        fn ckks_sub_pt_const_znx_assign<Dst, P>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            dst_coeff: usize,
            pt_znx: &P,
            pt_coeff: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos,
            P: $crate::GLWEToBackendRef<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxRshSubCoeffIntoBackend<$backend> + poulpy_core::GLWENormalize<$backend> + $crate::leveled::default::pt_znx::CKKSPlaintextDefault<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_const_znx_assign_default(module, dst, dst_coeff, pt_znx, pt_coeff, scratch)
        }
    };
}

pub use crate::impl_ckks_sub_default_methods;
