#include "q120_ntt.h"

#ifndef NDEBUG
#define CHECK_BOUNDS 1
#define VERBOSE
#else
#define CHECK_BOUNDS 0
#endif

#ifndef VERBOSE
#define LOG(...) ;
#else
#define LOG(...) printf(__VA_ARGS__);
#endif

typedef struct _q120_ntt_step_precomp {
  uint64_t q2bs[4];  // q2bs = 2^{bs-31}.q[k]
  uint64_t bs;       // inputs at this iterations must be in Q_n
  uint64_t half_bs;  // == ceil(bs/2)
  uint64_t mask;     // (1<<half_bs) - 1
  uint8_t reduce;
} q120_ntt_step_precomp;

typedef struct _q120_ntt_reduc_step_precomp {
  uint64_t modulo_red_cst[4];
  uint64_t mask;
  uint64_t h;
} q120_ntt_reduc_step_precomp;

typedef struct _q120_ntt_precomp {
  uint64_t n;  // NTT size (a power of 2)

  q120_ntt_step_precomp* level_metadata;
  uint64_t* powomega;
  q120_ntt_reduc_step_precomp reduc_metadata;

  uint64_t input_bit_size;
  uint64_t output_bit_size;
} q120_ntt_precomp;
