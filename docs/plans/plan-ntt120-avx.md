# Plan: NTT120 AVX2 Backend

## Status overview

| Component | Module | Status |
|-|-|-|
| NTT forward/inverse butterfly | `ntt120/ntt_avx.rs` | ✅ Done |
| mat_vec BBC — `NttMulBbc` / x2 / 2cols | `ntt120/mat_vec_avx.rs` | ✅ Done |
| `NTT120Avx` struct + all OEP trait wiring | `ntt120/mod.rs` + sub-modules | ✅ Done |
| `NttAdd/Sub/Negate*` — lazy q120b arithmetic | `ntt120/prim.rs` | ✅ Done |
| `VecZnxBig` normalization (`I128NormalizeOps`) | `ntt120/vec_znx_big_avx.rs` | ✅ Done |
| `VecZnxBig` i128 arithmetic (`I128BigOps`) | `ntt120/vec_znx_big_avx.rs` | ✅ Done |
| `NttFromZnx64` — `b_from_znx64` | `ntt120/arithmetic_avx.rs` | ✅ Done |
| `NttToZnx128` — `b_to_znx128` | `ntt120/arithmetic_avx.rs` | ✅ Done (hybrid) |
| `NttMulBbb` — `vec_mat1col_product_bbb` | `ntt120/arithmetic_avx.rs` | ✅ Done |
| `NttCFromB` — `c_from_b` | `ntt120/arithmetic_avx.rs` | ✅ Done |
| `NttExtract1BlkContiguous` | `ntt120/prim.rs` | skip (trivial copy) |
| `convolution.rs` | `ntt120/convolution.rs` | auto (composes above) |

---

## Already implemented — key notes

### `NttAdd/Sub/Negate*` (done)

`Q_SHIFTED[k] = Q[k] << 33 < 2^63` for Primes30 — single conditional subtract suffices.
`lazy_reduce`: XOR with MSB to convert unsigned `≥` to signed `cmpgt_epi64`, then
`andnot + sub` to conditionally subtract `q_s`. ~6 instructions per `__m256i`.

### `VecZnxBig` normalization (done)

Three `#[inline(always)]` chunk helpers sharing precomputed `NfcShifts`:
- `nfc_middle_chunk(s, lo_a, hi_a, lo_c, hi_c) → (lo_out, new_lo_c, new_hi_c)` — shared by
  `nfc_middle_step_avx2` (i128 input) and `nfc_middle_step_inplace_avx2` (i64 input).
- `nfc_final_chunk(s, lo_a, lo_c) → lo_out` — `hi_dpc` is never needed since
  `get_digit(base2k, x)` with `base2k ≤ 64` only reads bits `[0, base2k)` of x.

All three public kernels process 4 elements per chunk; scalar tail for `n % 4 != 0`.
Handles both `lsh == 0` and `lsh != 0` uniformly via `base2k_lsh = base2k - lsh`.

`i128` memory layout: `[lo: u64, hi: i64]` per element. Two `__m256i` loads give
`[lo0,hi0,lo1,hi1]` and `[lo2,hi2,lo3,hi3]`; deinterleave with `vpunpcklqdq/hi`.

### `VecZnxBig` i128 arithmetic (done)

`I128BigOps` methods in `ntt120/vec_znx_big.rs` dispatch to AVX2 kernels in
`vec_znx_big_avx.rs`: `vi128_add_avx2`, `vi128_sub_avx2`, `vi128_negate_avx2`,
`vi128_from_i64_avx2` — all using the same deinterleaved split `(lo, hi)` form.

---

## Remaining work — implementation plan

All four remaining scalar functions live in `poulpy-cpu-avx/src/ntt120/prim.rs`.
The AVX2 kernels go in a new file `poulpy-cpu-avx/src/ntt120/arithmetic_avx.rs`
(mirroring the `mat_vec_avx.rs` pattern), then wired in `prim.rs`.

### 1. `NttMulBbb` — `vec_mat1col_product_bbb_avx2`

**Scalar logic** (`mat_vec.rs:268`): for each of `ell` elements:
```
a = xl*yl,  s1 += a_lo,  s2 += a_hi + b_lo + c_lo
b = xl*yh,  s3 += b_hi + c_hi + d_lo,  s4 += d_hi
c = xh*yl,  d = xh*yh
```
Final reduction (per prime k):
```
t = s1l + s1h * 2^h
  + s2l * s2l_pow_red[k] + s2h * s2h_pow_red[k]
  + s3l * s3l_pow_red[k] + s3h * s3h_pow_red[k]
  + s4l * s4l_pow_red[k] + s4h * s4h_pow_red[k]
```
where `sXl = sX & mask_h2`, `sXh = sX >> h2`.

**AVX2 plan**: Each q120b element is one `__m256i` (4 × u64 — one per prime).
`_mm256_mul_epu32(a, b)` computes four 32×32→64 products simultaneously.

Inner loop (per element):
```rust
let xl = _mm256_and_si256(xv, mask32);        // lower 32 bits of each prime lane
let xh = _mm256_srli_epi64::<32>(xv);         // upper 32 bits shifted down
let yl = _mm256_and_si256(yv, mask32);
let yh = _mm256_srli_epi64::<32>(yv);
let a = _mm256_mul_epu32(xl, yl);             // xl*yl → 64 bits per lane
let b = _mm256_mul_epu32(xl, yh);
let c = _mm256_mul_epu32(xh, yl);
let d = _mm256_mul_epu32(xh, yh);
s1 += a & mask32
s2 += (a >> 32) + (b & mask32) + (c & mask32)
s3 += (b >> 32) + (c >> 32) + (d & mask32)
s4 += d >> 32
```

Final reduction: `s1h_pow_red` is prime-independent (= 2^h, scalar broadcast);
`s2l/s2h/s3l/s3h/s4l/s4h pow_red` are per-prime → load as `__m256i` from `BbbMeta`.
All `sXl`, `sXh` fit in 32 bits after masking/shifting, so `mul_epu32` works.

```rust
let h2 = meta.h;  // u64 shift immediate
let mask_h2 = _mm256_set1_epi64x(((1u64 << h2) - 1) as i64);
let h2_cnt  = _mm_cvtsi64_si128(h2 as i64);
// s1h_pow_red = 2^h (same for all primes — scalar broadcast)
let s1h_pow = _mm256_set1_epi64x(meta.s1h_pow_red as i64);
// s2..s4 pow_red: per-prime __m256i
let [s2l, s2h, s3l, s3h, s4l, s4h] = /* loadu from BbbMeta arrays */;

let split = |s| (_mm256_and_si256(s, mask_h2), _mm256_srl_epi64(s, h2_cnt));
let (s1l, s1h) = split(s1);
// ... similarly for s2..s4
let mut t = s1l;
t = add(t, mul_epu32(s1h, s1h_pow));  // prime-independent
t = add(t, mul_epu32(s2l, s2l_pow));
// ... 6 more mul_epu32 + add
storeu(res_ptr, t);
```

**File**: `arithmetic_avx.rs` — `pub(crate) unsafe fn vec_mat1col_product_bbb_avx2`.
**Wire**: `prim.rs` `NttMulBbb for NTT120Avx` calls `vec_mat1col_product_bbb_avx2` directly
(no `BbbMeta::new()` call per invocation — caller owns `meta` as before).

Wait — current scalar wires as: `let meta = BbbMeta::<Primes30>::new();` each call.
The AVX2 function takes `meta: &BbbMeta<Primes30>` for the reduction constants.
**Same calling convention** as the BBC functions.

---

### 2. `NttCFromB` — `c_from_b_avx2`

**Scalar logic**: for each of `nn` elements (= 4 u64 input → 8 u32 output):
```
r = x[4*j+k] % Q[k]
res[8*j + 2*k]     = r as u32
res[8*j + 2*k + 1] = (r << 32) % Q[k] as u32
```

**AVX2 plan**: process one q120b element (one `__m256i` = 4 × u64) per iteration,
producing one `__m256i` of q120c output (8 × u32 = one `__m256i`).

Key challenge: full modular reduction of u64 by Q[k] (~30-bit prime).
Inputs are q120b values in `[0, Q_SHIFTED[k])` = `[0, Q[k] << 33)`, so x < 2^63.

**Barrett reduction for u64 mod u30** using 32-bit multiplies:

Step 1 — split into 32-bit halves:
```
x_lo = x & mask32       (< 2^32)
x_hi = x >> 32          (< 2^31, since x < 2^63)
```
Step 2 — reduce `x_hi` to `[0, Q[k])`:
```
q_approx = mulhi32(x_hi, mu[k])   where mu[k] = floor(2^62 / Q[k])
r_hi     = x_hi - q_approx * Q[k]  (then conditional subtract if >= Q[k])
```
Since `x_hi < 2^31` and Q[k] < 2^31, one conditional subtract suffices.

Step 3 — combine:
```
tmp = r_hi * pow32[k] + x_lo    where pow32[k] = 2^32 mod Q[k]
r   = tmp mod Q[k]              (tmp < 2^31 * 2^30 + 2^32 ≈ 2^62 — one Barrett pass)
```
Step 4 — compute `r_shift = r * 2^32 mod Q[k]`:
```
r_shift = mulhi32(r, mu[k]) ... conditional subtract
```

In AVX2, each step uses `_mm256_mul_epu32` (reads lower 32 bits of each 64-bit lane),
`_mm256_srli_epi64::<32>` to move results, `_mm256_sub_epi64` for conditional subtract.

Output packing: r and r_shift are 32-bit values in lanes `[0,1,2,3]` of two `__m256i`s.
Pack into q120c layout (interleaved pairs) using `_mm256_unpacklo_epi32` + `vpermq`.

Precomputed constants (once, const/static): `mu[4]` and `pow32[4]` per prime set.
These can be added as new constants to `BbbMeta`/`BbcMeta`, or as a new `CFromBMeta` struct.

**File**: `arithmetic_avx.rs` — `pub(crate) unsafe fn c_from_b_avx2`.
**Wire**: `prim.rs` `NttCFromB for NTT120Avx`.

---

### 3. `NttFromZnx64` — `b_from_znx64_avx2`

**Scalar logic**: for each coefficient `x[j]`:
```
xl = x[j] as u64 & i64::MAX as u64    // strip sign bit
for k in 0..4:
    res[4*j+k] = xl + if x[j] < 0 { oq[k] } else { 0 }
    // oq[k] = Q[k] - (2^63 % Q[k])
```

**AVX2 plan**: process ONE input element per loop iteration; output one `__m256i`.
This vectorizes across the 4 primes (not across multiple input elements),
which is the natural unit since the output is one q120b = one `__m256i` per input.

```rust
// Precomputed once: oq_vec = [oq[0], oq[1], oq[2], oq[3]] as __m256i
let oq_vec  = _mm256_loadu_si256(oq.as_ptr() as *const __m256i);
let i64_max = _mm256_set1_epi64x(i64::MAX as i64);  // 0x7FFF...
let zero    = _mm256_setzero_si256();

for j in 0..nn:
    let xv     = _mm256_set1_epi64x(x[j]);           // broadcast to all 4 lanes
    let xl     = _mm256_and_si256(xv, i64_max);       // strip sign bit
    let sign   = _mm256_cmpgt_epi64(zero, xv);        // all-ones where x[j] < 0
    let add    = _mm256_and_si256(sign, oq_vec);       // oq[k] or 0 per lane
    storeu(res_ptr, _mm256_add_epi64(xl, add));
    res_ptr = res_ptr.add(1);  // advance by 4 u64
```

~5 instructions per input element. This replaces the 4-iteration inner k-loop.

**File**: `arithmetic_avx.rs` — `pub(crate) unsafe fn b_from_znx64_avx2`.
**Wire**: `prim.rs` `NttFromZnx64 for NTT120Avx`.

---

### 4. `NttToZnx128` — `b_to_znx128_avx2`

**Scalar logic**: CRT reconstruction for each of `nn` elements:
```
for k in 0..4:
    xk   = x[4*j+k] % Q[k]
    t    = (xk * CRT_CST[k]) % Q[k]
    tmp += t * (Q / Q[k])           // Q/Q[k] ≈ 2^90, t < 2^30
tmp %= total_Q                      // total_Q ≈ 2^120
res[j] = symmetric lift of tmp
```

**Complexity note**: unlike the other three, the final i128 accumulation and `% total_Q`
reduction are fundamentally scalar (total_Q is a 120-bit number). Full end-to-end
AVX2 acceleration is not feasible.

**Partial AVX2 plan** — accelerate steps 1-2 (the per-prime modular arithmetic):

Step 1 — `xk = x[4*j+k] % Q[k]` for k=0..3: use the same Barrett reduction as
`c_from_b_avx2` (same reduction, same constants). One `__m256i` → `xk[0..3]` as u32.

Step 2 — `t = (xk * CRT_CST[k]) % Q[k]`:
`xk < Q[k] < 2^31`, `CRT_CST[k] < Q[k] < 2^31` → product < 2^62.
Using `_mm256_mul_epu32(xk_vec, crt_vec)` where both inputs are 32-bit.
Then reduce mod Q[k] via Barrett (product < 2^62 < 2^32 * 2^30).

Result: `t[0..3]` in a `__m256i`, each lane holding t[k] as u64.

Step 3 — `tmp += t[k] * qm[k]` for k=0..3 and `tmp %= total_Q`:
`qm[k] = Q / Q[k] ≈ 2^90`. The product `t[k] * qm[k]` is ~120 bits.
This must remain scalar (i128 multiply + accumulate).

**Hybrid implementation**:
```rust
// AVX2: steps 1+2 — compute all four t[k] simultaneously
let t_vec = compute_crt_terms_avx2(xv, crt_cst_vec, q_vec, barrett_mu_vec);
// Scalar: step 3 — i128 accumulation
let t: [u64; 4] = extract(t_vec);
let tmp: i128 = t[0] as i128 * qm[0] + t[1] as i128 * qm[1]
              + t[2] as i128 * qm[2] + t[3] as i128 * qm[3];
// Scalar: final reduction + symmetric lift
```

**Speedup**: steps 1+2 are accelerated (~8 AVX2 instructions vs 4 × scalar Barrett passes);
step 3 is unchanged. The gain depends on relative costs, but steps 1+2 are non-trivial.

**File**: `arithmetic_avx.rs` — `pub(crate) unsafe fn b_to_znx128_avx2`.
**Wire**: `prim.rs` `NttToZnx128 for NTT120Avx`.

---

## Implementation order and dependencies

```
arithmetic_avx.rs (new file)
  └── vec_mat1col_product_bbb_avx2   ← NttMulBbb   (highest impact)
  └── c_from_b_avx2                  ← NttCFromB   (introduces Barrett constants)
  └── b_from_znx64_avx2              ← NttFromZnx64 (simplest)
  └── b_to_znx128_avx2               ← NttToZnx128  (reuses Barrett from c_from_b)
```

`c_from_b` and `b_to_znx128` share the same Barrett reduction constants (`mu[k]`,
`pow32[k]`) — implement once and reuse. Both can share a `BarrettMeta` struct alongside
the existing `BbbMeta` / `BbcMeta`.

## New file structure

```
poulpy-cpu-avx/src/ntt120/
  arithmetic_avx.rs       ← NEW: b_from_znx64_avx2, c_from_b_avx2,
                                  b_to_znx128_avx2, vec_mat1col_product_bbb_avx2
  mat_vec_avx.rs          existing (BBC)
  ntt_avx.rs              existing (NTT butterfly)
  vec_znx_big_avx.rs      existing (i128 ops + normalization)
  prim.rs                 existing (updated dispatch)
  mod.rs                  existing (may need `mod arithmetic_avx`)
```

## Barrett reduction constants (`BarrettMeta` or inline in `arithmetic_avx.rs`)

For Primes30, precomputable at compile time or lazily via `once_cell`:

```rust
// For each prime k: mu_k = floor(2^62 / Q[k])  — fits in u32 (< 2^32)
// For each prime k: pow32_k = 2^32 mod Q[k]    — fits in u32
// These are reused by both c_from_b and b_to_znx128.
const BARRETT_MU:    [u32; 4] = [...];
const POW32_MOD_Q:   [u32; 4] = [...];  // = 2^32 mod Q[k]
```

For `b_from_znx64`:
```rust
// oq[k] = Q[k] - (2^63 % Q[k]) — precomputed once, stored as [u64; 4]
const OQ: [u64; 4] = [...];
```

Both are Primes30-specific static constants (not generic over PrimeSet initially,
same as the BBC functions already hardcoded to Primes30).

---

## Testing

Re-use `poulpy_hal::test_suite` cross-backend helpers, comparing
`Module<NTT120Ref>` (scalar) vs `Module<NTT120Avx>` (AVX2).

New unit tests in `arithmetic_avx.rs` comparing against `b_from_znx64_ref`,
`c_from_b_ref`, `b_to_znx128_ref`, `vec_mat1col_product_bbb_ref` for random inputs.
