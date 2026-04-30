#[macro_export]
macro_rules! impl_ckks_neg_default_methods {
    ($backend:ty) => {
        fn ckks_neg_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::neg::CKKSNegDefault<$backend>>::ckks_neg_tmp_bytes_default(module)
        }

        fn ckks_neg_into<Dst: poulpy_hal::layouts::Data, Src: poulpy_hal::layouts::Data>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<Dst>,
            src: &$crate::layouts::CKKSCiphertext<Src>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWENegate<$backend> + poulpy_core::GLWEShift<$backend>,
            $crate::layouts::CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<$backend>,
            $crate::layouts::CKKSCiphertext<Src>: poulpy_core::layouts::GLWEToBackendRef<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::neg::CKKSNegDefault<$backend>>::ckks_neg_into_default(module, dst, src, scratch)
        }

        fn ckks_neg_assign<Dst: poulpy_hal::layouts::Data>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<Dst>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWENegate<$backend>,
            $crate::layouts::CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::neg::CKKSNegDefault<$backend>>::ckks_neg_assign_default(module, dst)
        }
    };
}

pub use crate::impl_ckks_neg_default_methods;
