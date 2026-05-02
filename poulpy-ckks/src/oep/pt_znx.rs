#[macro_export]
macro_rules! impl_ckks_pt_znx_default_methods {
    ($backend:ty) => {
        fn ckks_extract_pt_znx_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxLshTmpBytes + poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::pt_znx::CKKSPlaintextZnxDefault<$backend>>::ckks_extract_pt_znx_tmp_bytes_default(module)
        }

        fn ckks_extract_pt_znx<Dst, Src: poulpy_hal::layouts::Data, S: $crate::CKKSInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            src: &poulpy_core::layouts::GLWEPlaintext<Src>,
            src_meta: &S,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_core::ScratchArenaTakeCore<'a, $backend>,
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxLshBackend<$backend> + poulpy_hal::api::VecZnxRshBackend<$backend>,
            Dst: poulpy_core::layouts::GLWEPlaintextToBackendMut<$backend>
                + poulpy_core::layouts::GLWEInfos
                + poulpy_core::layouts::LWEInfos
                + $crate::CKKSInfos
                + $crate::SetCKKSInfos,
            poulpy_core::layouts::GLWEPlaintext<Src>: poulpy_core::layouts::GLWEPlaintextToBackendRef<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::pt_znx::CKKSPlaintextZnxDefault<$backend>>::ckks_extract_pt_znx_default(module, dst, src, src_meta, scratch)
        }
    };
}

pub use crate::impl_ckks_pt_znx_default_methods;
