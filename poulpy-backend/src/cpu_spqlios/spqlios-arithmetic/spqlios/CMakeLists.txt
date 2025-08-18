enable_language(ASM)

# C source files that are compiled for all targets (i.e. reference code)
set(SRCS_GENERIC
        commons.c 
        commons_private.c
        coeffs/coeffs_arithmetic.c
        arithmetic/vec_znx.c
        arithmetic/vec_znx_dft.c
        arithmetic/vector_matrix_product.c
        cplx/cplx_common.c
        cplx/cplx_conversions.c
        cplx/cplx_fft_asserts.c
        cplx/cplx_fft_ref.c
        cplx/cplx_fftvec_ref.c
        cplx/cplx_ifft_ref.c
        cplx/spqlios_cplx_fft.c
        reim4/reim4_arithmetic_ref.c
        reim4/reim4_fftvec_addmul_ref.c
        reim4/reim4_fftvec_conv_ref.c
        reim/reim_conversions.c
        reim/reim_fft_ifft.c
        reim/reim_fft_ref.c
        reim/reim_fftvec_ref.c
        reim/reim_ifft_ref.c
        reim/reim_ifft_ref.c
        reim/reim_to_tnx_ref.c
        q120/q120_ntt.c
        q120/q120_arithmetic_ref.c
        q120/q120_arithmetic_simple.c
        arithmetic/scalar_vector_product.c
        arithmetic/vec_znx_big.c
        arithmetic/znx_small.c
        arithmetic/module_api.c
        arithmetic/zn_vmp_int8_ref.c
        arithmetic/zn_vmp_int16_ref.c
        arithmetic/zn_vmp_int32_ref.c
        arithmetic/zn_vmp_ref.c
        arithmetic/zn_api.c
        arithmetic/zn_conversions_ref.c
        arithmetic/zn_approxdecomp_ref.c
        arithmetic/vec_rnx_api.c
        arithmetic/vec_rnx_conversions_ref.c
        arithmetic/vec_rnx_svp_ref.c
        reim/reim_execute.c
        cplx/cplx_execute.c
        reim4/reim4_execute.c
        arithmetic/vec_rnx_arithmetic.c
        arithmetic/vec_rnx_approxdecomp_ref.c
        arithmetic/vec_rnx_vmp_ref.c
)
# C or assembly source files compiled only on x86 targets
set(SRCS_X86
        )
# C or assembly source files compiled only on aarch64 targets
set(SRCS_AARCH64
        cplx/cplx_fallbacks_aarch64.c
        reim/reim_fallbacks_aarch64.c
        reim4/reim4_fallbacks_aarch64.c
        q120/q120_fallbacks_aarch64.c
        reim/reim_fft_neon.c
)

# C or assembly source files compiled only on x86: avx, avx2, fma targets
set(SRCS_FMA_C
        arithmetic/vector_matrix_product_avx.c
        cplx/cplx_conversions_avx2_fma.c
        cplx/cplx_fft_avx2_fma.c
        cplx/cplx_fft_sse.c
        cplx/cplx_fftvec_avx2_fma.c
        cplx/cplx_ifft_avx2_fma.c
        reim4/reim4_arithmetic_avx2.c
        reim4/reim4_fftvec_conv_fma.c
        reim4/reim4_fftvec_addmul_fma.c
        reim/reim_conversions_avx.c
        reim/reim_fft4_avx_fma.c
        reim/reim_fft8_avx_fma.c
        reim/reim_ifft4_avx_fma.c
        reim/reim_ifft8_avx_fma.c
        reim/reim_fft_avx2.c
        reim/reim_ifft_avx2.c
        reim/reim_to_tnx_avx.c
        reim/reim_fftvec_fma.c
)
set(SRCS_FMA_ASM
        cplx/cplx_fft16_avx_fma.s
        cplx/cplx_ifft16_avx_fma.s
        reim/reim_fft16_avx_fma.s
        reim/reim_ifft16_avx_fma.s
)
set(SRCS_FMA_WIN32_ASM
        cplx/cplx_fft16_avx_fma_win32.s
        cplx/cplx_ifft16_avx_fma_win32.s
        reim/reim_fft16_avx_fma_win32.s
        reim/reim_ifft16_avx_fma_win32.s
)
set_source_files_properties(${SRCS_FMA_C} PROPERTIES COMPILE_OPTIONS "-mfma;-mavx;-mavx2")
set_source_files_properties(${SRCS_FMA_ASM} PROPERTIES COMPILE_OPTIONS "-mfma;-mavx;-mavx2")

# C or assembly source files compiled only on x86: avx512f/vl/dq + fma targets
set(SRCS_AVX512
        cplx/cplx_fft_avx512.c
        )
set_source_files_properties(${SRCS_AVX512} PROPERTIES COMPILE_OPTIONS "-mfma;-mavx512f;-mavx512vl;-mavx512dq")

# C or assembly source files compiled only on x86: avx2 + bmi targets
set(SRCS_AVX2
        arithmetic/vec_znx_avx.c
        coeffs/coeffs_arithmetic_avx.c
        arithmetic/vec_znx_dft_avx2.c
        arithmetic/zn_vmp_int8_avx.c
        arithmetic/zn_vmp_int16_avx.c
        arithmetic/zn_vmp_int32_avx.c
        q120/q120_arithmetic_avx2.c
        q120/q120_ntt_avx2.c
        arithmetic/vec_rnx_arithmetic_avx.c
        arithmetic/vec_rnx_approxdecomp_avx.c
        arithmetic/vec_rnx_vmp_avx.c
        
)
set_source_files_properties(${SRCS_AVX2} PROPERTIES COMPILE_OPTIONS "-mbmi2;-mavx2")

# C source files on float128 via libquadmath on x86 targets targets
set(SRCS_F128
        cplx_f128/cplx_fft_f128.c
        cplx_f128/cplx_fft_f128.h
        )

# H header files containing the public API (these headers are installed)
set(HEADERSPUBLIC
        commons.h
        arithmetic/vec_znx_arithmetic.h
        arithmetic/vec_rnx_arithmetic.h
        arithmetic/zn_arithmetic.h
        cplx/cplx_fft.h
        reim/reim_fft.h
        q120/q120_common.h
        q120/q120_arithmetic.h
        q120/q120_ntt.h
        )

# H header files containing the private API (these headers are used internally)
set(HEADERSPRIVATE
        commons_private.h
        cplx/cplx_fft_internal.h
        cplx/cplx_fft_private.h
        reim4/reim4_arithmetic.h
        reim4/reim4_fftvec_internal.h
        reim4/reim4_fftvec_private.h
        reim4/reim4_fftvec_public.h
        reim/reim_fft_internal.h
        reim/reim_fft_private.h
        q120/q120_arithmetic_private.h
        q120/q120_ntt_private.h
        arithmetic/vec_znx_arithmetic.h
        arithmetic/vec_rnx_arithmetic_private.h
        arithmetic/vec_rnx_arithmetic_plugin.h
        arithmetic/zn_arithmetic_private.h
        arithmetic/zn_arithmetic_plugin.h
        coeffs/coeffs_arithmetic.h
        reim/reim_fft_core_template.h
)

set(SPQLIOSSOURCES
        ${SRCS_GENERIC}
        ${HEADERSPUBLIC}
        ${HEADERSPRIVATE}
        )
if (${X86})
    set(SPQLIOSSOURCES ${SPQLIOSSOURCES}
            ${SRCS_X86}
            ${SRCS_FMA_C}
            ${SRCS_FMA_ASM}
            ${SRCS_AVX2}
            ${SRCS_AVX512}
            )
elseif (${X86_WIN32})
    set(SPQLIOSSOURCES ${SPQLIOSSOURCES}
            #${SRCS_X86}
            ${SRCS_FMA_C}
            ${SRCS_FMA_WIN32_ASM}
            ${SRCS_AVX2}
            ${SRCS_AVX512}
    )
elseif (${AARCH64})
    set(SPQLIOSSOURCES ${SPQLIOSSOURCES}
            ${SRCS_AARCH64}
            )
endif ()


set(SPQLIOSLIBDEP
        m         # libmath depencency for cosinus/sinus functions
        )

if (ENABLE_SPQLIOS_F128)
    find_library(quadmath REQUIRED NAMES quadmath)
    set(SPQLIOSSOURCES ${SPQLIOSSOURCES} ${SRCS_F128})
    set(SPQLIOSLIBDEP ${SPQLIOSLIBDEP} quadmath)
endif (ENABLE_SPQLIOS_F128)

add_library(libspqlios-static STATIC ${SPQLIOSSOURCES})
add_library(libspqlios SHARED ${SPQLIOSSOURCES})
set_property(TARGET libspqlios-static PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET libspqlios PROPERTY OUTPUT_NAME spqlios)
set_property(TARGET libspqlios-static PROPERTY OUTPUT_NAME spqlios)
set_property(TARGET libspqlios PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET libspqlios PROPERTY SOVERSION ${SPQLIOS_VERSION_MAJOR})
set_property(TARGET libspqlios PROPERTY VERSION ${SPQLIOS_VERSION})
if (NOT APPLE)
target_link_options(libspqlios-static PUBLIC -Wl,--no-undefined)
target_link_options(libspqlios PUBLIC -Wl,--no-undefined)
endif()
target_link_libraries(libspqlios ${SPQLIOSLIBDEP})
target_link_libraries(libspqlios-static ${SPQLIOSLIBDEP})
install(TARGETS libspqlios-static)
install(TARGETS libspqlios)

# install the public headers only
foreach (file ${HEADERSPUBLIC})
    get_filename_component(dir ${file} DIRECTORY)
    install(FILES ${file} DESTINATION include/spqlios/${dir})
endforeach ()
