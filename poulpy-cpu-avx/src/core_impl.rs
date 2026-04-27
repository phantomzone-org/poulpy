use crate::{FFT64Avx, NTT120Avx};
use poulpy_core::{
    ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendRef, GGLWEToBackendMut, GGLWEToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef,
        GGSWInfos, GGSWPreparedToBackendRef, GLWEInfos, GLWEPlaintext, GLWESecretPrepared, GLWESecretTensorPrepared, GLWETensor,
        GLWEToBackendMut, GLWEToBackendRef, LWEInfos, LWEPlaintextToBackendMut, LWESecretToBackendRef, LWEToBackendRef,
        SetLWEInfos,
    },
    oep::{
        AutomorphismDefaults, AutomorphismImpl, ConversionImpl, DecryptionImpl, GGLWEExternalProductImpl, GGLWEKeyswitchImpl,
        GGSWExternalProductImpl, GGSWKeyswitchImpl, GGSWRotateImpl, GLWEExternalProductImpl, GLWEKeyswitchImpl, GLWEMulConstImpl,
        GLWEMulPlainImpl, GLWEMulXpMinusOneImpl, GLWENormalizeImpl, GLWEPackImpl, GLWERotateImpl, GLWEShiftImpl,
        GLWETensoringImpl, GLWETraceImpl, LWEKeyswitchImpl, OperationsDefaults,
    },
};
use poulpy_hal::layouts::{Backend, HostBackend, HostDataMut, Module, ScratchArena};

poulpy_cpu_ref::impl_decryption_via_helpers!(FFT64Avx, poulpy_cpu_ref::core_impl);
poulpy_cpu_ref::impl_conversion_via_helpers!(FFT64Avx, poulpy_cpu_ref::core_impl);
poulpy_cpu_ref::impl_external_product_via_helpers!(FFT64Avx, poulpy_cpu_ref::core_impl);
poulpy_cpu_ref::impl_keyswitching_via_helpers!(FFT64Avx, poulpy_cpu_ref::core_impl);

poulpy_cpu_ref::impl_decryption_via_helpers!(NTT120Avx, poulpy_cpu_ref::core_impl);
poulpy_cpu_ref::impl_conversion_via_helpers!(NTT120Avx, poulpy_cpu_ref::core_impl);
poulpy_cpu_ref::impl_external_product_via_helpers!(NTT120Avx, poulpy_cpu_ref::core_impl);
poulpy_cpu_ref::impl_keyswitching_via_helpers!(NTT120Avx, poulpy_cpu_ref::core_impl);

poulpy_cpu_ref::impl_automorphism_via_defaults!(FFT64Avx);
poulpy_cpu_ref::impl_operations_via_defaults!(FFT64Avx);

poulpy_cpu_ref::impl_automorphism_via_defaults!(NTT120Avx);
poulpy_cpu_ref::impl_operations_via_defaults!(NTT120Avx);
