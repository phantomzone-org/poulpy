use crate::{FFT64Avx, NTT120Avx};
use poulpy_core::{
    ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendRef, GGLWEToBackendMut, GGLWEToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef,
        GGSWInfos, GGSWPreparedToBackendRef, GGSWToBackendMut, GLWEInfos, GLWEPlaintext, GLWEScratchMut, GLWESecretPrepared,
        GLWESecretTensorPrepared, GLWETensor, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, LWEPlaintextToBackendMut,
        LWESecretToBackendRef, LWEToBackendMut, LWEToBackendRef, SetLWEInfos,
    },
    oep::{
        AutomorphismDefaults, AutomorphismImpl, ConversionImpl, DecryptionImpl, GGLWEExternalProductImpl, GGLWEKeyswitchImpl,
        GGSWExternalProductImpl, GGSWKeyswitchImpl, GGSWRotateImpl, GLWEAddImpl, GLWECopyImpl, GLWEExternalProductImpl,
        GLWEKeyswitchImpl, GLWEMulConstImpl, GLWEMulPlainImpl, GLWEMulXpMinusOneImpl, GLWENegateImpl, GLWENormalizeImpl,
        GLWEPackImpl, GLWERotateImpl, GLWEShiftImpl, GLWESubImpl, GLWETensoringImpl, GLWETraceImpl, LWEKeyswitchImpl,
        OperationsDefaults,
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
