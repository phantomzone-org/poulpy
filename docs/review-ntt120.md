# NTT120 Code Review

**Scope:** `poulpy-hal/src/reference/ntt120/`, `poulpy-cpu-ref/src/ntt120/`, `poulpy-cpu-avx/src/ntt120/`

---

## High-Level Risk Summary

The NTT120 implementation is a well-structured, largely faithful port of `spqlios-arithmetic`.
The layer separation (HAL / cpu-ref / cpu-avx) is clean, the trait architecture elegantly
sidesteps the orphan rule, and the AVX2 kernels correctly mirror the scalar reference.
Documentation is unusually thorough for this class of code.

That said, two issues are severe enough to produce **silent wrong results** in production
cryptographic computations. One concerns the SVP accumulate API contract (data silently
clobbered), and one concerns the pairwise convolution algorithm (product-of-sums instead of
sum-of-products). Beyond these, the codebase carries several high-priority performance and
safety time-bombs, most of which stem from the same root cause: the `BbcMeta` / `BbbMeta`
metadata design does not extend cleanly to the `NttMulBbb` hot path, forcing per-call
recomputation.

---

## Prioritized Issues

### Critical

---

#### C1. `ntt120_svp_apply_dft_to_dft_add` silently zeroes output limbs it should leave untouched

**File:** `poulpy-hal/src/reference/ntt120/svp.rs:189–192`

```rust
// Limbs beyond b.size(): zero out (clear any stale data).
for j in min_size..res_size {
    BE::ntt_zero(cast_slice_mut(res.at_mut(res_col, j)));
}
```

This is the **accumulate** variant (`_add` suffix). A caller who first zeroes `res`, then
calls `svp_apply_dft_to_dft_add` to accumulate `a ⊙ b₀ + a ⊙ b₁ + …` will see the earlier
accumulations destroyed for limbs where `b.size() < res.size()`. The overwrite variant
(`ntt120_svp_apply_dft_to_dft`) correctly zeroes them; the accumulate variant must not. The
VMP `save_blk_add` never zeroes out-of-range slots — it is consistent; SVP is not.

**Fix:** Remove the zeroing loop from `ntt120_svp_apply_dft_to_dft_add` entirely. Any caller
that requires the "out-of-range limbs are zero" invariant must either call the overwrite
variant first or zero `res` manually before accumulating.

---

#### C2. `ntt120_cnv_pairwise_apply_dft` computes product-of-sums, not sum-of-products

**File:** `poulpy-hal/src/reference/ntt120/convolution.rs:376–403`

```rust
// Docstring claims: res = a[:,col_i]⊙b[:,col_i] + a[:,col_j]⊙b[:,col_j]
// Actual computation:
a_sum[2 * k] = ((ai_k % q) + (aj_k % q)) as u32;  // a_i + a_j, hi = 0
b_sum[2 * k]     = bi[2 * k] + bj[2 * k];
b_sum[2 * k + 1] = bi[2 * k + 1] + bj[2 * k + 1];
accum_mul_q120_bc(&mut s, &a_sum, &b_sum);
```

`accum_mul_q120_bc` computes `x_lo * y_lo + x_hi * y_hi`. With `a_sum_hi = 0` and
`b_sum = bᵢ + bⱼ`, this yields `(aᵢ + aⱼ) · (bᵢ + bⱼ) mod Q`, which expands to
`aᵢbᵢ + aᵢbⱼ + aⱼbᵢ + aⱼbⱼ`. The cross-terms `aᵢbⱼ + aⱼbᵢ` are non-zero in general.
The algorithm is therefore incorrect as written unless the call context guarantees these
cancel — which is not documented or enforced anywhere.

**Fix:** Either document the exact algebraic invariant that makes cross-terms vanish in all
valid call sites, or replace with two independent accumulations:

```rust
accum_mul_q120_bc(&mut s, ai, bi);
accum_mul_q120_bc(&mut s, aj, bj);
```

The test `test_convolution_pairwise` must explicitly verify the non-trivial case where
`col_i ≠ col_j` and both columns carry non-zero data.

---

### High

---

#### H1. `NttMulBbb` recomputes `BbbMeta` on every invocation

**Files:** `poulpy-cpu-ref/src/ntt120/prim.rs:176–179`, `poulpy-cpu-avx/src/ntt120/prim.rs:329–333`

```rust
impl NttMulBbb for NTT120Ref {
    fn ntt_mul_bbb(ell: usize, res: &mut [u64], a: &[u64], b: &[u64]) {
        let meta = BbbMeta::<Primes30>::new();  // recomputed every call
        vec_mat1col_product_bbb_ref::<Primes30>(&meta, ell, res, a, b);
    }
}
```

`BbbMeta::new()` iterates over `h ∈ [16, 32)` computing `compute_bit_size_red` (which calls
`pow2_mod` four times with 64-bit binary exponentiation). This is ~64 integer operations per
invocation. For an NTT of size `n = 2^16`, `NttMulBbb` is called `n` times per element,
making this O(n) wasted work per operation.

`BbcMeta` is correctly cached in the module handle via `NttModuleHandle::get_bbc_meta()`.
`BbbMeta` should receive identical treatment: add it to the handle alongside `BbcMeta`, and
extend the `NttMulBbb` trait signature to accept `&BbbMeta<Primes30>` as a parameter (or
cache it in the backend struct).

---

#### H2. `Q_SHIFTED` is defined independently in four places

**Files:** `poulpy-hal/src/reference/ntt120/types.rs:132`, `poulpy-hal/src/reference/ntt120/vmp.rs:42`,
`poulpy-cpu-ref/src/ntt120/prim.rs:24`, `poulpy-cpu-avx/src/ntt120/prim.rs:54`

All four definitions are byte-for-byte identical. With `Primes30` hardcoding they agree
today, but a future refactor adding `Primes29`/`Primes31` dispatch will silently miss one of
them. `poulpy-cpu-ref` and `poulpy-cpu-avx` should import from
`poulpy_hal::reference::ntt120::types::Q_SHIFTED`; the `vmp.rs` copy should also be
eliminated.

---

#### H3. AVX2 pointer-stepping kernels have no release-mode bounds checks

**Files:** `poulpy-cpu-avx/src/ntt120/arithmetic_avx.rs:172–186`, `poulpy-cpu-avx/src/ntt120/ntt.rs:119–132`

```rust
pub(crate) unsafe fn b_from_znx64_avx2(nn: usize, res: &mut [u64], x: &[i64]) {
    // Only debug_assert! guards; no release-mode check on res.len()
    let mut r_ptr = res.as_mut_ptr() as *mut __m256i;
    for &xval in &x[..nn] {    // panics if nn > x.len(), but...
        _mm256_storeu_si256(r_ptr, ...);
        r_ptr = r_ptr.add(1);  // res.len() >= 4*nn is never asserted in release
    }
}
```

The `&x[..nn]` slice index panics if `nn > x.len()`, but nothing prevents `res.len() < 4 * nn`
in release builds. The write-through-raw-pointer chain then produces silent memory corruption.
All such functions (the full family in `arithmetic_avx.rs`, `ntt.rs`, `mat_vec_avx.rs`) should
either promote the `debug_assert!` to `assert!` at entry, or explicitly document (with a
reasoning comment) why the debug-only guard is sufficient given the trait-impl call sites.

---

#### H4. Negation of zero produces `Q_SHIFTED[k]`, not `0`

**Files:** `poulpy-cpu-ref/src/ntt120/prim.rs:131–140`, `poulpy-cpu-avx/src/ntt120/prim.rs:179–192`

```rust
// ref:
res[idx] = q_s - a[idx] % q_s;    // when a[idx] == 0: result = q_s

// avx:
_mm256_storeu_si256(r_ptr, _mm256_sub_epi64(q_s, av));  // when av == 0: result = q_s
```

Both return `Q_SHIFTED[k]` for a zero input. `Q_SHIFTED[k] ≡ 0 (mod Q[k])`, so this is
arithmetically valid in lazy arithmetic. However, the output range is `(0, Q_SHIFTED]`, not
`[0, Q_SHIFTED)`. Any code that tests `val == 0` after negation to detect a zero element will
fail. The add/sub/negate family should explicitly document that the output range is
`(0, Q_SHIFTED]` for zero input, and that only `val % Q[k] == 0` (not `val == 0`) is a
correct zero test.

---

### Medium

---

#### M1. Tests only exercise `n = 256`, leaving critical code paths untested

**File:** `poulpy-cpu-avx/src/ntt120/tests.rs:18`

```rust
size = 1<<8,   // n = 256 for all cross-backend test suites
```

The by-level / by-block transition in the AVX2 NTT occurs at `CHANGE_MODE_N = 1024`. Only
the by-block phase (`n ≤ 1024`) is exercised. The by-level phase (`n > 1024`, up to `2^16`)
and the full twiddle-table traversal at maximum size are never tested. At minimum, add tests
for `n ∈ {1024, 2048, 65536}` in the NTT-specific unit tests and in the cross-backend suite.

---

#### M2. `NttHandleProvider` unsafe invariant is documentation-only

**File:** `poulpy-hal/src/reference/ntt120/vec_znx_dft.rs:77–84`

```rust
/// # Safety
/// Implementors must ensure the returned references are valid for the lifetime
/// of `&self` and that the tables were fully initialised before first use.
pub unsafe trait NttHandleProvider { ... }
```

The blanket impl calls `unsafe { (&*self.ptr()).get_ntt_table() }` where `ptr()` is a raw
pointer to `B::Handle`. There is no runtime check that the handle was initialized before
`Module::new()` completes. A `B::Handle` that implements `NttHandleProvider` with
uninitialized tables due to a constructor bug will return garbage references with no
diagnostic. Consider adding a runtime flag (checked in debug mode) that panics if the handle
is accessed before initialization is complete.

---

#### M3. `Primes29` and `Primes31` are dead code at the HAL boundary

**File:** `poulpy-hal/src/reference/ntt120/vec_znx_dft.rs:57–64`

```rust
pub trait NttModuleHandle {
    fn get_ntt_table(&self) -> &NttTable<Primes30>;    // hardcoded
    fn get_intt_table(&self) -> &NttTableInv<Primes30>;
    fn get_bbc_meta(&self) -> &BbcMeta<Primes30>;
}
```

The `PrimeSet` trait and `Primes29`/`Primes31` structs are fully implemented and unit-tested
in `ntt.rs`, but unreachable through the HAL public API. The documented extension path ("add
an associated type `PrimeSet` to `NttModuleHandle`") should either be tracked as a concrete
issue or the dead code removed to avoid confusion about what is actually supported.

---

#### M4. `fill_omegas` performs an unnecessary `as i64` cast

**File:** `poulpy-hal/src/reference/ntt120/ntt.rs:163–165`

```rust
fn fill_omegas<P: PrimeSet>(n: usize) -> [u32; 4] {
    std::array::from_fn(|k| modq_pow(P::OMEGA[k], (1 << 16) / n as i64, P::Q[k]))
}
```

`(1 << 16) / n as i64` converts `usize` to `i64` before the division. More importantly,
`fill_omegas` does not check `n >= 1`; it relies entirely on the caller's `assert!`. If
called with `n = 0`, this panics with an integer division-by-zero at runtime without a
helpful message. `fill_omegas` should express its own precondition via a `debug_assert!(n >=
1)` and document that `n` must be a power of two in `[1, 2^16]`.

---

#### M5. `ntt120_cnv_prepare_left` bypasses the backend trait dispatch

**File:** `poulpy-hal/src/reference/ntt120/convolution.rs:85–86`

```rust
b_from_znx64_ref::<Primes30>(n, res_u64, a.at(col, j));
ntt_ref(table, res_u64);   // hardcoded scalar NTT, not BE::ntt_dft_execute
```

All other prepare entry points (SVP prepare, VMP prepare) dispatch through
`BE::ntt_dft_execute` and `BE::ntt_from_znx64`, allowing the AVX2 backend to substitute
its accelerated kernels. `ntt120_cnv_prepare_left` always runs the scalar reference NTT,
silently degrading performance when used with `NTT120Avx`. The `BE` parameter already
constrains the backend — add `+ NttDFTExecute<NttTable<Primes30>> + NttFromZnx64` to the
where clause and dispatch through the trait.

---

### Low

---

#### L1. `b_to_znx128_avx2` scalar CRT loop dominates for large `n`

**File:** `poulpy-cpu-avx/src/ntt120/arithmetic_avx.rs:383–393`

```rust
// Step 3 (scalar): CRT accumulation in i128
let mut tmp: i128 = 0;
for k in 0..4 {
    tmp += t_arr[k] as i128 * qm[k];
}
```

The AVX2 path accelerates `xk = x % Q` and `t = xk * CRT_CST % Q`, but the i128
accumulation is scalar. For `n = 65536`, this inner loop runs 65536 times. The `qm` values
are ~90-bit integers (`q[1]*q[2]*q[3] ≈ 2^90`), so this genuinely requires 128-bit
arithmetic. A vectorized approach using manual 64×64→128 decomposition could halve the
reconstruction cost. Worth noting as a future-work item, since `b_to_znx128` is called on
every `idft_apply` for every limb.

---

#### L2. `NttMulBbc` argument labeling is inverted at call sites

**File:** `poulpy-hal/src/reference/ntt120/svp.rs:119–126`

```rust
// Trait: fn ntt_mul_bbc(..., a: &[u32], b: &[u32])
//   where a = q120b operand, b = q120c prepared operand
BE::ntt_mul_bbc(
    meta, 1, &mut res_u64[...],
    &b_u32[...],   // passed as `a` — but named `b` in this function (the VecZnxDft)
    &a_u32[...],   // passed as `b` — but named `a` in this function (the SvpPPol)
);
```

The BBC product is commutative in its result, so this is mathematically correct. However,
the `a`/`b` naming in the trait conflicts with the `a`/`b` naming at the call site, creating
a persistent readability hazard. Rename the trait parameters to `ntt_coeff: &[u32]` (q120b)
and `prepared: &[u32]` (q120c) to match the semantic roles.

---

#### L3. `sra_epi64` does not guard against out-of-range `imm`

**File:** `poulpy-cpu-avx/src/ntt120/vec_znx_big_avx.rs:115–128`

The function works correctly for `imm ∈ [0, 64]` but this precondition is neither stated
nor asserted. All current call sites pass values derived from `base2k` and `lsh` which are
bounded appropriately by the normalization algorithm. Add `debug_assert!(imm <= 64)` at the
function entry to make the contract explicit.

---

## Refactoring Suggestions

**1. Cache `BbbMeta` in the module handle.**
Extend `NttHandleProvider` (or `NttModuleHandle`) with `fn get_bbb_meta() ->
&BbbMeta<Primes30>`, populate it once during module construction, and thread it through
`NttMulBbb`. This removes the hot-path recomputation (H1) and makes `BbbMeta` consistent
with `BbcMeta`.

**2. Consolidate `Q_SHIFTED` to a single canonical definition.**
Remove the three backend copies and import from `poulpy_hal::reference::ntt120::types::Q_SHIFTED`
everywhere. If `Primes30` hardcoding is ever lifted, there will be a single definition to
update.

**3. Introduce a newtype for lazy q120b limbs.**
A `Q120bLimb(u64)` with `impl Add` / `impl Sub` that encodes the lazy arithmetic invariants
(`value ∈ [0, 2·Q_SHIFTED)`) would allow the type system to enforce preconditions that are
currently only documented as prose, and would make the negation-of-zero output range (H4)
explicit at the type level.

**4. Fix `ntt120_cnv_prepare_left` to use `BE` dispatch.**
A one-line trait bound addition removes a silent performance regression for any AVX2 caller
(M5), and is consistent with every other prepare function in the codebase.

**5. Expand test sizes to cover the by-level phase and near-maximum `n`.**
Add a parameterized NTT test over `n ∈ {2, 4, 8, 256, 1024, 2048, 65536}` to the AVX2
test suite. The `CHANGE_MODE_N = 1024` boundary is the highest-risk transition point and is
currently not covered.

---

## Follow-up Questions

1. **`ntt120_cnv_pairwise_apply_dft` (C2):** Is the product-of-sums formula intentional?
   If so, what algebraic structure in the call context guarantees that
   `a[:,i] ⊙ b[:,j] + a[:,j] ⊙ b[:,i] = 0` for all valid inputs? This must be
   documented with a proof sketch or a reference to the relevant FHE construction.

2. **`ntt120_svp_apply_dft_to_dft_add` (C1):** What is the specified postcondition for
   output limbs beyond `min(b.size(), res.size())`? Is zeroing those limbs intentional (a
   form of "output has exactly `b.size()` valid limbs after this call") or is it a
   copy-paste error from the overwrite variant?

3. **Barrett `r < 3Q` claim (`barrett_reduce` in `arithmetic_avx.rs:130–149`):** Is the
   two-conditional-subtract correctness formally verified for all Primes30 values, or is it
   inherited from the C port's test coverage? Given the split-quotient computation across
   two floor operations (`q_hi` and `q_lo`), a concise argument or reference would be
   appropriate in the function's doc comment.

4. **`NttMulBbc` argument order (L2):** Should the trait parameters be renamed to
   `ntt_coeff` and `prepared` (matching the semantic roles) instead of the ambiguous `a`/`b`
   that conflict with callers' local naming?

5. **`NttHandleProvider` initialization order (M2):** Is there a documented guarantee that
   `B::Handle` is always fully initialized before `Module<B>` is returned from its
   constructor? If so, where is that guarantee established — in `ModuleNewImpl::new_impl`,
   or in the `Backend` trait contract?
