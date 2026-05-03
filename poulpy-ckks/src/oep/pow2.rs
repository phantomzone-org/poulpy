#[macro_export]
macro_rules! impl_ckks_pow2_default_methods {
    ($backend:ty) => {
        fn ckks_mul_pow2_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::pow2::CKKSPow2Default<$backend>>::ckks_mul_pow2_tmp_bytes_default(module)
        }

        fn ckks_mul_pow2_into<Dst, Src>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            src: &Src,
            bits: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos + $crate::SetCKKSInfos,
            Src: $crate::GLWEToBackendRef<$backend>
                + poulpy_core::layouts::GLWEInfos
                + poulpy_core::layouts::LWEInfos
                + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::pow2::CKKSPow2Default<$backend>>::ckks_mul_pow2_into_default(module, dst, src, bits, scratch)
        }

        fn ckks_mul_pow2_assign<Dst>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            bits: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos + $crate::SetCKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::pow2::CKKSPow2Default<$backend>>::ckks_mul_pow2_assign_default(module, dst, bits, scratch)
        }

        fn ckks_div_pow2_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::pow2::CKKSPow2Default<$backend>>::ckks_div_pow2_tmp_bytes_default(module)
        }

        fn ckks_div_pow2_into<Dst, Src>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            src: &Src,
            bits: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos + $crate::SetCKKSInfos,
            Src: $crate::GLWEToBackendRef<$backend>
                + poulpy_core::layouts::GLWEInfos
                + poulpy_core::layouts::LWEInfos
                + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend> + poulpy_core::GLWECopy<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::pow2::CKKSPow2Default<$backend>>::ckks_div_pow2_into_default(module, dst, src, bits, scratch)
        }

        fn ckks_div_pow2_assign<Dst>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            bits: usize,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos + $crate::SetCKKSInfos,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::pow2::CKKSPow2Default<$backend>>::ckks_div_pow2_assign_default(module, dst, bits)
        }
    };
}

pub use crate::impl_ckks_pow2_default_methods;
