#ifndef SPQLIOS_Q120_COMMON_H
#define SPQLIOS_Q120_COMMON_H

#include <stdint.h>

#if !defined(SPQLIOS_Q120_USE_29_BIT_PRIMES) && !defined(SPQLIOS_Q120_USE_30_BIT_PRIMES) && \
    !defined(SPQLIOS_Q120_USE_31_BIT_PRIMES)
#define SPQLIOS_Q120_USE_30_BIT_PRIMES
#endif

/**
 * 29-bit primes and 2*2^16 roots of unity
 */
#ifdef SPQLIOS_Q120_USE_29_BIT_PRIMES
#define Q1 ((1u << 29) - 2 * (1u << 17) + 1)
#define OMEGA1 78289835
#define Q1_CRT_CST 301701286  // (Q2*Q3*Q4)^-1 mod Q1

#define Q2 ((1u << 29) - 5 * (1u << 17) + 1)
#define OMEGA2 178519192
#define Q2_CRT_CST 536020447  // (Q1*Q3*Q4)^-1 mod Q2

#define Q3 ((1u << 29) - 26 * (1u << 17) + 1)
#define OMEGA3 483889678
#define Q3_CRT_CST 86367873  // (Q1*Q2*Q4)^-1 mod Q3

#define Q4 ((1u << 29) - 35 * (1u << 17) + 1)
#define OMEGA4 239808033
#define Q4_CRT_CST 147030781  // (Q1*Q2*Q3)^-1 mod Q4
#endif

/**
 * 30-bit primes and 2*2^16 roots of unity
 */
#ifdef SPQLIOS_Q120_USE_30_BIT_PRIMES
#define Q1 ((1u << 30) - 2 * (1u << 17) + 1)
#define OMEGA1 1070907127
#define Q1_CRT_CST 43599465  // (Q2*Q3*Q4)^-1 mod Q1

#define Q2 ((1u << 30) - 17 * (1u << 17) + 1)
#define OMEGA2 315046632
#define Q2_CRT_CST 292938863  // (Q1*Q3*Q4)^-1 mod Q2

#define Q3 ((1u << 30) - 23 * (1u << 17) + 1)
#define OMEGA3 309185662
#define Q3_CRT_CST 594011630  // (Q1*Q2*Q4)^-1 mod Q3

#define Q4 ((1u << 30) - 42 * (1u << 17) + 1)
#define OMEGA4 846468380
#define Q4_CRT_CST 140177212  // (Q1*Q2*Q3)^-1 mod Q4
#endif

/**
 * 31-bit primes and 2*2^16 roots of unity
 */
#ifdef SPQLIOS_Q120_USE_31_BIT_PRIMES
#define Q1 ((1u << 31) - 1 * (1u << 17) + 1)
#define OMEGA1 1615402923
#define Q1_CRT_CST 1811422063  // (Q2*Q3*Q4)^-1 mod Q1

#define Q2 ((1u << 31) - 4 * (1u << 17) + 1)
#define OMEGA2 1137738560
#define Q2_CRT_CST 2093150204  // (Q1*Q3*Q4)^-1 mod Q2

#define Q3 ((1u << 31) - 11 * (1u << 17) + 1)
#define OMEGA3 154880552
#define Q3_CRT_CST 164149010  // (Q1*Q2*Q4)^-1 mod Q3

#define Q4 ((1u << 31) - 23 * (1u << 17) + 1)
#define OMEGA4 558784885
#define Q4_CRT_CST 225197446  // (Q1*Q2*Q3)^-1 mod Q4
#endif

static const uint32_t PRIMES_VEC[4] = {Q1, Q2, Q3, Q4};
static const uint32_t OMEGAS_VEC[4] = {OMEGA1, OMEGA2, OMEGA3, OMEGA4};

#define MAX_ELL 10000

// each number x mod Q120 is represented by uint64_t[4] with (non-unique) values (x mod q1, x mod q2,x mod q3,x mod q4),
// each between [0 and 2^32-1]
typedef struct _q120a q120a;

// each number x mod Q120 is represented by uint64_t[4] with (non-unique) values (x mod q1, x mod q2,x mod q3,x mod q4),
// each between [0 and 2^64-1]
typedef struct _q120b q120b;

// each number x mod Q120 is represented by uint32_t[8] with values (x mod q1, 2^32x mod q1, x mod q2, 2^32.x mod q2, x
// mod q3, 2^32.x mod q3, x mod q4, 2^32.x mod q4) each between [0 and 2^32-1]
typedef struct _q120c q120c;

typedef struct _q120x2b q120x2b;
typedef struct _q120x2c q120x2c;

#endif  // SPQLIOS_Q120_COMMON_H
