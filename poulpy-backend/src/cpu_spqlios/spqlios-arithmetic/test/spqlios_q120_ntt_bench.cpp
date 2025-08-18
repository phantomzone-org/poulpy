#include <benchmark/benchmark.h>

#include <cstdint>

#include "spqlios/q120/q120_ntt.h"

#define ARGS Arg(1 << 10)->Arg(1 << 11)->Arg(1 << 12)->Arg(1 << 13)->Arg(1 << 14)->Arg(1 << 15)->Arg(1 << 16)

template <typeof(q120_ntt_bb_avx2) f>
void benchmark_ntt(benchmark::State& state) {
  const uint64_t n = state.range(0);
  q120_ntt_precomp* precomp = q120_new_ntt_bb_precomp(n);

  uint64_t* px = new uint64_t[n * 4];
  for (uint64_t i = 0; i < 4 * n; i++) {
    px[i] = (rand() << 31) + rand();
  }
  for (auto _ : state) {
    f(precomp, (q120b*)px);
  }
  delete[] px;
  q120_del_ntt_bb_precomp(precomp);
}

template <typeof(q120_intt_bb_avx2) f>
void benchmark_intt(benchmark::State& state) {
  const uint64_t n = state.range(0);
  q120_ntt_precomp* precomp = q120_new_intt_bb_precomp(n);

  uint64_t* px = new uint64_t[n * 4];
  for (uint64_t i = 0; i < 4 * n; i++) {
    px[i] = (rand() << 31) + rand();
  }
  for (auto _ : state) {
    f(precomp, (q120b*)px);
  }
  delete[] px;
  q120_del_intt_bb_precomp(precomp);
}

BENCHMARK(benchmark_ntt<q120_ntt_bb_avx2>)->Name("q120_ntt_bb_avx2")->ARGS;
BENCHMARK(benchmark_intt<q120_intt_bb_avx2>)->Name("q120_intt_bb_avx2")->ARGS;

BENCHMARK_MAIN();
