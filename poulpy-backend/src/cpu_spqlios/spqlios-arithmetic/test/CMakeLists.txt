set(CMAKE_CXX_STANDARD 17)

set(test_incs ..)
set(gtest_libs)
set(benchmark_libs)
# searching for libgtest
find_path(gtest_inc NAMES gtest/gtest.h)
find_library(gtest NAMES gtest)
find_library(gtest_main REQUIRED NAMES gtest_main)
if (gtest_inc AND gtest AND gtest_main)
    message(STATUS "Found gtest: I=${gtest_inc} L=${gtest},${gtest_main}")
    set(test_incs ${test_incs} ${gtest_inc})
    set(gtest_libs ${gtest_libs} ${gtest} ${gtest_main} pthread)
else()
    message(FATAL_ERROR "Libgtest not found (required if ENABLE_TESTING is on): I=${gtest_inc} L=${gtest},${gtest_main}")
endif()
# searching for libbenchmark
find_path(benchmark_inc NAMES benchmark/benchmark.h)
find_library(benchmark NAMES benchmark)
if (benchmark_inc AND benchmark)
    message(STATUS "Found benchmark: I=${benchmark_inc} L=${benchmark}")
    set(test_incs ${test_incs} ${benchmark_inc})
    set(benchmark_libs ${benchmark_libs} ${benchmark})
else()
    message(FATAL_ERROR "Libbenchmark not found (required if ENABLE_TESTING is on): I=${benchmark_inc} L=${benchmark}")
endif()
find_path(VALGRIND_DIR NAMES valgrind/valgrind.h)
if (VALGRIND_DIR)
    message(STATUS "Found valgrind header ${VALGRIND_DIR}")
else ()
    # for now, we will fail if we don't find valgrind for tests
    message(STATUS "CANNOT FIND valgrind header: ${VALGRIND_DIR}")
endif ()

add_library(spqlios-testlib SHARED
        testlib/random.cpp
        testlib/test_commons.h
        testlib/test_commons.cpp
        testlib/mod_q120.h
        testlib/mod_q120.cpp
        testlib/negacyclic_polynomial.cpp
        testlib/negacyclic_polynomial.h
        testlib/negacyclic_polynomial_impl.h
        testlib/reim4_elem.cpp
        testlib/reim4_elem.h
        testlib/fft64_dft.cpp
        testlib/fft64_dft.h
        testlib/fft64_layouts.h
        testlib/fft64_layouts.cpp
        testlib/ntt120_layouts.cpp
        testlib/ntt120_layouts.h
        testlib/ntt120_dft.cpp
        testlib/ntt120_dft.h
        testlib/test_hash.cpp
        testlib/sha3.h
        testlib/sha3.c
        testlib/polynomial_vector.h
        testlib/polynomial_vector.cpp
        testlib/vec_rnx_layout.h
        testlib/vec_rnx_layout.cpp
        testlib/zn_layouts.h
        testlib/zn_layouts.cpp
)
if (VALGRIND_DIR)
    target_include_directories(spqlios-testlib PRIVATE ${VALGRIND_DIR})
    target_compile_definitions(spqlios-testlib PRIVATE VALGRIND_MEM_TESTS)
endif ()
target_link_libraries(spqlios-testlib libspqlios)



# main unittest file
message(STATUS "${gtest_libs}")
set(UNITTEST_FILES
        spqlios_test.cpp
        spqlios_reim_conversions_test.cpp
        spqlios_reim_test.cpp
        spqlios_reim4_arithmetic_test.cpp
        spqlios_cplx_test.cpp
        spqlios_cplx_conversions_test.cpp
        spqlios_q120_ntt_test.cpp
        spqlios_q120_arithmetic_test.cpp
        spqlios_coeffs_arithmetic_test.cpp
        spqlios_vec_znx_big_test.cpp
        spqlios_znx_small_test.cpp
        spqlios_vmp_product_test.cpp
        spqlios_vec_znx_dft_test.cpp
        spqlios_svp_apply_test.cpp
        spqlios_svp_prepare_test.cpp
        spqlios_vec_znx_test.cpp
        spqlios_vec_rnx_test.cpp
        spqlios_vec_rnx_vmp_test.cpp
        spqlios_vec_rnx_conversions_test.cpp
        spqlios_vec_rnx_ppol_test.cpp
        spqlios_vec_rnx_approxdecomp_tnxdbl_test.cpp
        spqlios_zn_approxdecomp_test.cpp
        spqlios_zn_conversions_test.cpp
        spqlios_zn_vmp_test.cpp
        

)

add_executable(spqlios-test ${UNITTEST_FILES})
target_link_libraries(spqlios-test spqlios-testlib libspqlios ${gtest_libs})
target_include_directories(spqlios-test PRIVATE ${test_incs})
add_test(NAME spqlios-test COMMAND spqlios-test)
if (WIN32)
    # copy the dlls to the test directory
    cmake_minimum_required(VERSION 3.26)
    add_custom_command(
            POST_BUILD
            TARGET spqlios-test
            COMMAND ${CMAKE_COMMAND} -E copy
            -t $<TARGET_FILE_DIR:spqlios-test> $<TARGET_RUNTIME_DLLS:spqlios-testlib> $<TARGET_RUNTIME_DLLS:libspqlios>
            COMMAND_EXPAND_LISTS
    )
endif()

# benchmarks
add_executable(spqlios-cplx-fft-bench spqlios_cplx_fft_bench.cpp)
target_link_libraries(spqlios-cplx-fft-bench libspqlios ${benchmark_libs} pthread)
target_include_directories(spqlios-cplx-fft-bench PRIVATE ${test_incs})

if (X86 OR X86_WIN32)
    add_executable(spqlios-q120-ntt-bench spqlios_q120_ntt_bench.cpp)
    target_link_libraries(spqlios-q120-ntt-bench libspqlios ${benchmark_libs} pthread)
    target_include_directories(spqlios-q120-ntt-bench PRIVATE ${test_incs})

    add_executable(spqlios-q120-arithmetic-bench spqlios_q120_arithmetic_bench.cpp)
    target_link_libraries(spqlios-q120-arithmetic-bench libspqlios  ${benchmark_libs} pthread)
    target_include_directories(spqlios-q120-arithmetic-bench PRIVATE ${test_incs})
endif ()

if (X86 OR X86_WIN32)
    add_executable(spqlios_reim4_arithmetic_bench spqlios_reim4_arithmetic_bench.cpp)
    target_link_libraries(spqlios_reim4_arithmetic_bench ${benchmark_libs} libspqlios pthread)
    target_include_directories(spqlios_reim4_arithmetic_bench PRIVATE ${test_incs})
endif ()

if (DEVMODE_INSTALL)
    install(TARGETS spqlios-testlib)
endif()
