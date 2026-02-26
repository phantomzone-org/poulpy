# Plan: NTT120 AVX2 Backend — remaining work

## What has been implemented

| Component | Module | Status |
|---|---|---|
| NTT forward/inverse butterfly | `ntt120/ntt.rs` | ✅ Done |
| mat_vec BBC (SVP/VMP hot path) | `ntt120/mat_vec_avx.rs` | ✅ Done |
| `NTT120Avx` struct + all OEP trait wiring | `ntt120/mod.rs` + sub-modules | ✅ Done |

---

## Remaining work

### Phase 2: VecZnxBig AVX2 ops (add/sub/negate/from_small)

All `VecZnxBig*` ops in `ntt120/vec_znx_big.rs` currently delegate to scalar
reference functions.  The AVX2 gain is ~2× on all of these.

#### Memory layout

Each limb stores `n` consecutive `i128` values.  On little-endian x86-64,
`i128` occupies 16 bytes as `[lo: u64, hi: u64]`.  An array of n elements:

```
[lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3, ...]   ← interleaved
```

A 256-bit AVX2 load reads `[lo0, hi0, lo1, hi1]` — **2 elements, interleaved**.
In AVX2 kernels, cast `&mut [i128]` → `&mut [i64]` via `bytemuck::cast_slice_mut`
(safe: i128 = 2 × i64, identical alignment), then deinterleave:

```
// deinterleave (processes 4 i128 = 8 i64 at once):
lo_vec = vpunpcklqdq(a, b)   // [lo0, lo1, lo2, lo3]
hi_vec = vpunpckhqdq(a, b)   // [hi0, hi1, hi2, hi3]
// ... process in split form ...
// re-interleave:
a = vpunpcklqdq(lo_vec, hi_vec)  // [lo0, hi0, lo1, hi1]
b = vpunpckhqdq(lo_vec, hi_vec)  // [lo2, hi2, lo3, hi3]
```

Operations to implement in a new `ntt120/vec_znx_big_avx.rs`:

- `vadd_i128_avx2`: split → `add_epi64` + carry via `cmpgt` trick (≈ 4 instr / 4 elems)
- `vsub_i128_avx2`: split → `sub_epi64` + borrow
- `vneg_i128_avx2`: `(0 - lo, 0 - hi - borrow)`
- `vfrom_small_avx2`: sign-extend i64 → i128 using arithmetic shift broadcast

Then wire into the relevant `unsafe impl` blocks in `vec_znx_big.rs`.

---

### Phase 3: VecZnxBig AVX2 normalization

This is the highest-value remaining target.

The normalization kernel (`nfc_middle_step` / `nfc_final_step_inplace`) iterates
over all n coefficients extracting `base2k`-bit digits and propagating carries.
The split-(lo,hi) layout enables processing 4 elements per AVX2 register pair,
vs 2 with the interleaved layout.

**Estimated throughput: ~3 instr/element vs ~7–8 scalar.**

#### Key helper: arithmetic right shift of i64 (AVX2 emulation)

AVX2 has no `_mm256_srav_epi64`; emulate for `imm ∈ [1,63]`:

```rust
// sra_epi64_avx2(v, imm) — arithmetic right shift by runtime imm
let sign = _mm256_srai_epi32(_mm256_shuffle_epi32(v, 0xF5), 31); // broadcast sign bits
let shifted = _mm256_srl_epi64(v, _mm_cvtsi64_si128(imm as i64));
let mask = _mm256_sll_epi64(_mm256_cmpeq_epi64(v, v), _mm_cvtsi64_si128((64 - imm) as i64));
_mm256_or_si256(shifted, _mm256_and_si256(sign, mask))
```

For constant `imm` known at compile time, use `_mm256_srli_epi64::<IMM>` and
`_mm256_slli_epi64::<{64-IMM}>` instead.

#### Normalization recipe for `base2k ≤ 64` (split lo/hi):

```
digit_vec = sra64(sll_epi64(lo_vec, 64-base2k), 64-base2k)   // sign-extend low base2k bits
diff_lo   = sub_epi64(lo_vec, digit_vec)                      // lo - digit (exact mod 2^base2k)
carry_lo  = srl_epi64(diff_lo, base2k)                        // logical shift ok: diff ≡ 0 mod 2^base2k
carry_mid = sll_epi64(hi_vec,  64-base2k)                     // hi bits that spill into carry_lo
carry_lo  = or_epi64(carry_lo, carry_mid)
carry_hi  = sra64(hi_vec, base2k)
```

Implement in `ntt120/vec_znx_big_avx.rs` as `nfc_middle_step_avx2` and
`nfc_final_step_inplace_avx2`, then wire into `vec_znx_big.rs`.

---

## Dependency on `unsafe` / feature flags

- All AVX2 functions: `#[target_feature(enable = "avx2")]`.
- Fallback to scalar when AVX2 unavailable (handled by `NTT120Avx::new()` panic).

## Testing

Re-use `poulpy_hal::test_suite` cross-backend helpers, comparing
`Module<NTT120Ref>` (scalar) vs `Module<NTT120Avx>` (AVX2).
