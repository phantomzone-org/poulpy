#include <benchmark/benchmark.h>

#include "spqlios/reim4/reim4_arithmetic.h"

void init_random_values(uint64_t n, double* v) {
  for (uint64_t i = 0; i < n; ++i)
    v[i] = (double(rand() % (UINT64_C(1) << 14)) - (UINT64_C(1) << 13)) / (UINT64_C(1) << 12);
}

// Run the benchmark
BENCHMARK_MAIN();

#undef ARGS
#define ARGS Args({47, 16384})->Args({93, 32768})

/*
 *  reim4_vec_mat1col_product
 *  reim4_vec_mat2col_product
 *  reim4_vec_mat3col_product
 *  reim4_vec_mat4col_product
 */

template <uint64_t X,
          void (*fnc)(const uint64_t nrows, double* const dst, const double* const u, const double* const v)>
void benchmark_reim4_vec_matXcols_product(benchmark::State& state) {
  const uint64_t nrows = state.range(0);

  double* u = new double[nrows * 8];
  init_random_values(8 * nrows, u);
  double* v = new double[nrows * X * 8];
  init_random_values(X * 8 * nrows, v);
  double* dst = new double[X * 8];

  for (auto _ : state) {
    fnc(nrows, dst, u, v);
  }

  delete[] dst;
  delete[] v;
  delete[] u;
}

#undef ARGS
#define ARGS Arg(128)->Arg(1024)->Arg(4096)

#ifdef __x86_64__
BENCHMARK(benchmark_reim4_vec_matXcols_product<1, reim4_vec_mat1col_product_avx2>)->ARGS;
// TODO: please remove when fixed:
BENCHMARK(benchmark_reim4_vec_matXcols_product<2, reim4_vec_mat2cols_product_avx2>)->ARGS;
#endif
BENCHMARK(benchmark_reim4_vec_matXcols_product<1, reim4_vec_mat1col_product_ref>)->ARGS;
BENCHMARK(benchmark_reim4_vec_matXcols_product<2, reim4_vec_mat2cols_product_ref>)->ARGS;
