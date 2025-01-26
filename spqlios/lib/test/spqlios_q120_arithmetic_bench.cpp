#include <benchmark/benchmark.h>

#include <cstdint>

#include "spqlios/q120/q120_arithmetic.h"

#define ARGS Arg(128)->Arg(4096)->Arg(10000)

template <typeof(q120_vec_mat1col_product_baa_ref) f>
void benchmark_baa(benchmark::State& state) {
  const uint64_t ell = state.range(0);
  q120_mat1col_product_baa_precomp* precomp = q120_new_vec_mat1col_product_baa_precomp();

  uint64_t* a = new uint64_t[ell * 4];
  uint64_t* b = new uint64_t[ell * 4];
  uint64_t* c = new uint64_t[4];
  for (uint64_t i = 0; i < 4 * ell; i++) {
    a[i] = rand();
    b[i] = rand();
  }
  for (auto _ : state) {
    f(precomp, ell, (q120b*)c, (q120a*)a, (q120a*)b);
  }
  delete[] c;
  delete[] b;
  delete[] a;
  q120_delete_vec_mat1col_product_baa_precomp(precomp);
}

BENCHMARK(benchmark_baa<q120_vec_mat1col_product_baa_ref>)->Name("q120_vec_mat1col_product_baa_ref")->ARGS;
BENCHMARK(benchmark_baa<q120_vec_mat1col_product_baa_avx2>)->Name("q120_vec_mat1col_product_baa_avx2")->ARGS;

template <typeof(q120_vec_mat1col_product_bbb_ref) f>
void benchmark_bbb(benchmark::State& state) {
  const uint64_t ell = state.range(0);
  q120_mat1col_product_bbb_precomp* precomp = q120_new_vec_mat1col_product_bbb_precomp();

  uint64_t* a = new uint64_t[ell * 4];
  uint64_t* b = new uint64_t[ell * 4];
  uint64_t* c = new uint64_t[4];
  for (uint64_t i = 0; i < 4 * ell; i++) {
    a[i] = rand();
    b[i] = rand();
  }
  for (auto _ : state) {
    f(precomp, ell, (q120b*)c, (q120b*)a, (q120b*)b);
  }
  delete[] c;
  delete[] b;
  delete[] a;
  q120_delete_vec_mat1col_product_bbb_precomp(precomp);
}

BENCHMARK(benchmark_bbb<q120_vec_mat1col_product_bbb_ref>)->Name("q120_vec_mat1col_product_bbb_ref")->ARGS;
BENCHMARK(benchmark_bbb<q120_vec_mat1col_product_bbb_avx2>)->Name("q120_vec_mat1col_product_bbb_avx2")->ARGS;

template <typeof(q120_vec_mat1col_product_bbc_ref) f>
void benchmark_bbc(benchmark::State& state) {
  const uint64_t ell = state.range(0);
  q120_mat1col_product_bbc_precomp* precomp = q120_new_vec_mat1col_product_bbc_precomp();

  uint64_t* a = new uint64_t[ell * 4];
  uint64_t* b = new uint64_t[ell * 4];
  uint64_t* c = new uint64_t[4];
  for (uint64_t i = 0; i < 4 * ell; i++) {
    a[i] = rand();
    b[i] = rand();
  }
  for (auto _ : state) {
    f(precomp, ell, (q120b*)c, (q120b*)a, (q120c*)b);
  }
  delete[] c;
  delete[] b;
  delete[] a;
  q120_delete_vec_mat1col_product_bbc_precomp(precomp);
}

BENCHMARK(benchmark_bbc<q120_vec_mat1col_product_bbc_ref>)->Name("q120_vec_mat1col_product_bbc_ref")->ARGS;
BENCHMARK(benchmark_bbc<q120_vec_mat1col_product_bbc_avx2>)->Name("q120_vec_mat1col_product_bbc_avx2")->ARGS;

EXPORT void q120x2_vec_mat2cols_product_bbc_avx2(q120_mat1col_product_bbc_precomp* precomp, const uint64_t ell,
                                                 q120b* const res, const q120b* const x, const q120c* const y);
EXPORT void q120x2_vec_mat1col_product_bbc_avx2(q120_mat1col_product_bbc_precomp* precomp, const uint64_t ell,
                                                q120b* const res, const q120b* const x, const q120c* const y);

template <typeof(q120_vec_mat1col_product_bbc_ref) f>
void benchmark_x2c2_bbc(benchmark::State& state) {
  const uint64_t ell = state.range(0);
  q120_mat1col_product_bbc_precomp* precomp = q120_new_vec_mat1col_product_bbc_precomp();

  uint64_t* a = new uint64_t[ell * 8];
  uint64_t* b = new uint64_t[ell * 16];
  uint64_t* c = new uint64_t[16];
  for (uint64_t i = 0; i < 8 * ell; i++) {
    a[i] = rand();
  }
  for (uint64_t i = 0; i < 16 * ell; i++) {
    b[i] = rand();
  }
  for (auto _ : state) {
    f(precomp, ell, (q120b*)c, (q120b*)a, (q120c*)b);
  }
  delete[] c;
  delete[] b;
  delete[] a;
  q120_delete_vec_mat1col_product_bbc_precomp(precomp);
}

BENCHMARK(benchmark_x2c2_bbc<q120x2_vec_mat2cols_product_bbc_avx2>)->Name("q120x2_vec_mat2col_product_bbc_avx2")->ARGS;

template <typeof(q120_vec_mat1col_product_bbc_ref) f>
void benchmark_x2c1_bbc(benchmark::State& state) {
  const uint64_t ell = state.range(0);
  q120_mat1col_product_bbc_precomp* precomp = q120_new_vec_mat1col_product_bbc_precomp();

  uint64_t* a = new uint64_t[ell * 8];
  uint64_t* b = new uint64_t[ell * 8];
  uint64_t* c = new uint64_t[8];
  for (uint64_t i = 0; i < 8 * ell; i++) {
    a[i] = rand();
  }
  for (uint64_t i = 0; i < 8 * ell; i++) {
    b[i] = rand();
  }
  for (auto _ : state) {
    f(precomp, ell, (q120b*)c, (q120b*)a, (q120c*)b);
  }
  delete[] c;
  delete[] b;
  delete[] a;
  q120_delete_vec_mat1col_product_bbc_precomp(precomp);
}

BENCHMARK(benchmark_x2c1_bbc<q120x2_vec_mat1col_product_bbc_avx2>)->Name("q120x2_vec_mat1col_product_bbc_avx2")->ARGS;

BENCHMARK_MAIN();
