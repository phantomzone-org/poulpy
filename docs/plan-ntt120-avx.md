# Plan: NTT120 AVX2 Backend (`poulpy-cpu-avx` extension)

## Goal

Add an AVX2-accelerated variant of the NTT120 backend alongside the existing
`FFT64AVX` in `poulpy-cpu-avx`, or as a new crate `poulpy-cpu-ntt120-avx`.
The reference scalar implementation lives in `poulpy-cpu-ntt120/`.

---

## VecZnxBig: i128 vs (i64_lo, i64_hi) — design decision

### Why i128 is required at the logical level

After `intt_ref` + `b_to_znx128_ref`, each coefficient is a signed CRT
reconstruction in `(-Q/2, Q/2]`.  With Primes30: Q ≈ 2^120, so values
span ~120 bits signed.  A single i64 (±2^63) cannot hold this; two i64
are the minimum.  `i128` is correct at the Rust level.

### AVX2 register landscape

AVX2 `__m256i` = 256-bit = 4 × i64 lanes.

| Operation | Native AVX2 | Notes |
|-|-|-|
| 4×i64 add/sub | Yes | `_mm256_add/sub_epi64` |
| 4×i64 logical shift | Yes | `_mm256_srl/sll_epi64` |
| 4×i64 **arithmetic** shift right | **No** | AVX-512 only; emulate ≈ 3 instr |
| Carry detection for i64 add | No flag | `cmpgt` trick ≈ 2 instr |
| Native 128-bit add/sub | No | must split lo + carry |

### Memory layout of `VecZnxBig<_, NTT120Ref>`

Each limb stores `n` consecutive `i128` values.  On little-endian x86-64,
`i128` occupies 16 bytes as `[lo: u64, hi: u64]`.  An array of n elements:

```
[lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3, ...]   ← interleaved
```

A 256-bit AVX2 load reads `[lo0, hi0, lo1, hi1]` — **2 elements, interleaved**.
Extracting 4 lo words into one register requires `vpunpcklqdq + vperm2i128`
(deinterleave), costing ~2 instructions per load and ~2 per store.

### Split layout: (lo-vector, hi-vector)

If elements were stored as `[lo0, lo1, lo2, lo3, ..., hi0, hi1, hi2, hi3, ...]`,
each AVX2 load directly gives 4 lo (or 4 hi) values — no shuffle needed.
This is **2× better data parallelism**.

### Recommendation: keep `ScalarBig = i128`, reinterpret in AVX2 kernels

To avoid breaking poulpy-core / the HAL type system:

- Keep `ScalarBig = i128` at the Rust type level (no change to any trait or layout).
- In AVX2 kernel functions, cast `&mut [i128]` → `&mut [i64]` via
  `bytemuck::cast_slice_mut` (safe: i128 = 2 × i64, identical alignment).
- Deinterleave once at function entry (4 shuffles per 32 bytes = per 2 elements),
  process in "split" form (4 elements per register pair), re-interleave at exit.
- Shuffle overhead is amortised over n ≥ 1024 elements.

### Normalization: the critical hot loop

`get_digit_i128(base2k, x)` = sign-extend low `base2k` bits.
`get_carry_i128(base2k, x, digit)` = `(x − digit) >> base2k` (arithmetic).

For `base2k ≤ 64` (always true in practice; typical: 14–60):

**Split (lo, hi) — 4 elements per AVX2 pair:**

```
digit_vec = sra64_emul(sll_epi64(lo_vec, 64-base2k), 64-base2k)   // ~5 instr
diff_lo   = sub_epi64(lo_vec, digit_vec)                           //  1 instr
carry_lo  = srl_epi64(diff_lo, base2k)                             //  1 instr (logical ok: diff ≡ 0 mod 2^base2k)
carry_mid = sll_epi64(hi_vec,  64-base2k)                          //  1 instr
carry_lo  = or_epi64(carry_lo, carry_mid)                          //  1 instr
carry_hi  = sra64_emul(hi_vec, base2k)                             // ~3 instr
```
≈ 12 instr / 4 elements = **3 instr/element**.

**Interleaved i128 (current layout, no deinterleave) — 2 elements per register:**

Same arithmetic but cross-lane 128-bit shifts required for the `(128−base2k)` shift.
≈ 14–16 instr / 2 elements = **7–8 instr/element**.

**Net gain from split processing: ≈ 2–2.5× on normalization.**

### Simple ops (add/sub/negate)

Split: 2 `vadd` + carry detection ≈ 4 instr / 4 elements.
Interleaved: deinterleave + same + interleave ≈ 8 instr / 2 elements.
≈ 2× faster.

### Residual cost that cannot be eliminated in AVX2

The arithmetic right shift of i64 requires a 3-instruction emulation regardless
of i128 vs (i64,i64).  AVX-512 (`_mm256_srav_epi64` with AVX-512VL) eliminates
this, but that is out of scope for the AVX2 target.

### Throughput summary

| Aspect | i128 interleaved | (lo,hi) split (via cast) |
|-|-|-|
| AVX2 elements / 256-bit register | 2 | 4 |
| Normalization throughput | 1× | ~2–2.5× |
| Add/sub/negate throughput | 1× | ~2× |
| Digit extraction (base2k ≤ 64) | cross-lane shift | stays in lo register |
| poulpy-core / HAL changes | none | none (cast at kernel boundary) |
| Memory footprint | same | same |

---

## Implementation Plan

### Phase 1: NTT forward/inverse — AVX2 butterfly

Port `q120_ntt_avx2.c` from spqlios-arithmetic.

Files to create (in `poulpy-cpu-avx/src/ntt120/` or new crate):

- `ntt_avx.rs` — forward NTT butterfly using AVX2 `_mm256_*` intrinsics.
- `intt_avx.rs` — inverse NTT butterfly.
- `arithmetic_avx.rs` — `split_precompmul`, `modq_red`, lazy-Barrett in AVX2.

Reference: [`spqlios/q120/q120_ntt_avx2.c`](https://github.com/tfhe/spqlios-arithmetic/blob/main/spqlios/q120/q120_ntt_avx2.c)

These operate on `&mut [u64]` (q120b layout, 4 u64 per coefficient),
identical to the scalar reference — no type-system change.

### Phase 2: VecZnxBig AVX2 ops (add/sub/negate/from_small)

In the AVX2 kernels:
```rust
let data: &mut [i64] = bytemuck::cast_slice_mut(limb_i128_slice);
// data is [lo0, hi0, lo1, hi1, ...] — deinterleave per batch of 4
```

Deinterleave kernel (processes 4 i128 = 8 i64 at once):
```
lo_vec = vpunpcklqdq(a, b)   // [lo0, lo1, lo2, lo3]
hi_vec = vpunpckhqdq(a, b)   // [hi0, hi1, hi2, hi3]
// ... process ...
// re-interleave:
a = vpunpcklqdq(lo_vec, hi_vec)  // [lo0, hi0, lo1, hi1]
b = vpunpckhqdq(lo_vec, hi_vec)  // [lo2, hi2, lo3, hi3]
```

Operations: `vadd_i128`, `vsub_i128`, `vneg_i128`, `vfrom_small_i128`.

### Phase 3: VecZnxBig AVX2 normalization

This is the highest-value target.  Implement `nfc_middle_step` and
`nfc_final_step_inplace` using the split-(lo,hi) AVX2 recipe above.

Key helper: `sra_epi64_avx2(v, imm)` emulated as:
```rust
// arithmetic right shift by imm ∈ [1,63]
let sign = _mm256_srai_epi32(_mm256_shuffle_epi32(v, 0xF5), 31); // broadcast sign
let shifted = _mm256_srli_epi64(v, imm);
let mask = _mm256_slli_epi64(_mm256_cmpeq_epi64(v,v), 64-imm);   // ones in top imm bits
_mm256_or_si256(shifted, _mm256_and_si256(sign, mask))
```

### Phase 4: mat_vec BBC AVX2

Port `q120_arithmetic_avx2.c` — the `vec_mat1col_product_bbc_avx2` function.
This is the inner product `Σ q120b[j] × q120c[j]` that drives SVP/VMP.

This operates purely on `u64`/`u32` slices (q120b/q120c), no i128 involved.

### Phase 5: Wire into poulpy-cpu-avx backend

Option A: new struct `NTT120AVX` in `poulpy-cpu-avx`.
Option B: new crate `poulpy-cpu-ntt120-avx` (cleaner separation).

Either way: implement `NttHandleProvider` for the AVX handle and add AVX
dispatch in the OEP impls.

---

## Dependency on `unsafe` / feature flags

- Target feature: `avx2,fma` (same as `FFT64AVX`).
- All AVX2 functions marked `#[target_feature(enable = "avx2")]`.
- Fallback to scalar `NTT120Ref` when AVX2 unavailable.

---

## Testing

Re-use `poulpy_hal::test_suite` cross-backend helpers, comparing:
- `Module<NTT120Ref>` (scalar) vs `Module<NTT120AVX>` (AVX2)

All existing `poulpy-cpu-ntt120` tests serve as the correctness oracle.
