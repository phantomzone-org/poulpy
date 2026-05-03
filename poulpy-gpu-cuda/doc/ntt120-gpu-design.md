# NTT120 GPU Backend Design

Design for implementing `VecZnx → VecZnxDft → VecZnxBig → VecZnx` over the
NTT120 representation on CUDA, reusing and adapting NTT kernels from `cheddar-fhe`.

---

## Background

### What NTT120 is

NTT120 is **not** a 120-bit NTT. It is a CRT representation over four ~30-bit
NTT-friendly primes whose product Q ≈ 2^120.

Default prime set (`Primes30`, matching spqlios-arithmetic):

| k | Prime formula | Value |
|---|---------------|-------|
| 0 | `(1<<30) - 2*(1<<17) + 1` | 1 073 479 681 |
| 1 | `(1<<30) - 17*(1<<17) + 1` | 1 071 513 601 |
| 2 | `(1<<30) - 23*(1<<17) + 1` | 1 070 727 169 |
| 3 | `(1<<30) - 42*(1<<17) + 1` | 1 068 236 801 |

All four support a primitive `2^17`-th root of unity, so NTT sizes up to `2^16`
are supported.

### Cheddar-fhe NTT kernels

Five CUDA kernel templates in `cheddar-fhe/src/core/NTT.cu`:

| Kernel | Phase | Algorithm |
|--------|-------|-----------|
| `NTTPhase1<word, log_degree>` | 1 of 2 | Forward Cooley-Tukey |
| `NTTPhase2<word, log_degree>` | 2 of 2 | Forward + OT twiddle |
| `INTTPhase1<word, log_degree>` | 1 of 2 | Inverse Gentleman-Sande |
| `INTTPhase2<word, log_degree, elem_func>` | 2 of 2 | Inverse + optional per-element fn |
| `NTTPhase2ForModDown<word, log_degree>` | fused | ModDown variant |

Key properties:
- Template-instantiated at compile time for `log_degree ∈ [12, 16]`
- **Multi-prime native**: `blockIdx.y` indexes the prime; one grid covers all primes
- Data offset per prime: `y_idx << log_degree` — each prime's data is a contiguous block
- Lazy Montgomery arithmetic with signed words in `(-q, q)`
- Two-phase for n=2^16: Phase1 (7 stages, blockDim=128) + Phase2 (9 stages, blockDim=64)

---

## Layout and kernel adaptation analysis

### Memory traffic baseline (n = 2^16, 4 primes, 1 limb)

The table below counts global memory bytes for the full forward+inverse pass
assuming the naive plan of separate kernels with prime-major `u32` buffers.

| Step | Read | Write | Total |
|------|------|-------|-------|
| `ntt120_reduce` (i64 → u32) | 512 KB | 1 MB | 1.5 MB |
| NTT Phase1 | 1 MB | 1 MB | 2.0 MB |
| NTT Phase2 | 1 MB | 1 MB | 2.0 MB |
| INTT Phase1 | 1 MB | 1 MB | 2.0 MB |
| INTT Phase2 + normalize | 1 MB | 1 MB | 2.0 MB |
| `ntt120_crt` (i32 → Big32) | 1 MB | 1 MB | 2.0 MB |
| **Total** | | | **11.5 MB** |

The pointwise multiply (SVP/VMP) is the same regardless of layout — it is not
counted here.

### Why a layout change alone does not help

The two candidate alternatives to prime-major SoA are:

**AoS `[coeff][4 u32]`**: coefficient-interleaved, matching the CPU q120b layout.

- Reduce/CRT: stride-1 access → coalesced ✓
- NTT Phase1 load: each thread loads `kPerThreadElems=16` elements for one prime.
  In AoS the stride between same-prime elements is 4, so a warp of 32 threads
  would access 32 × 4 = 128 consecutive u32s — not sequentially — requiring
  a non-coalesced scatter read. Adapting Phase1 to collect its elements from
  interleaved positions adds shuffle overhead and is difficult to vectorize.
- Net verdict: **AoS trades NTT coalescing for reduce/CRT coalescing. Not worth
  it** — the NTT phases touch 4× more data than the reduce/CRT pair.

**4-prime fused NTT (all primes in one CTA)**:

- Register pressure for Phase1 at kPerThreadElems=16: each prime needs ~16 data +
  ~16 twiddle + ~10 temporaries ≈ 45 registers. For 4 primes simultaneously:
  data arrays must all stay live → at minimum 4 × 16 = 64 data registers, total
  ≈ 85 registers/thread.
- At blockDim=128: 85 × 128 = 10 880 registers/block; 65 536/SM → 6 blocks →
  ~37% occupancy and 4 × 8 KB = 32 KB shared memory (tight, feasible on Ampere).
- The lower occupancy costs latency hiding. The gain is eliminating 4 separate
  Phase1 launches; but these launches already run concurrently on different SMs
  via `blockIdx.y`, so the serialization cost is already zero.
- Net verdict: **register pressure erases the benefit**. The separate-launch
  prime-major model is already parallel at the SM level.

### Where adaptation actually pays: boundary fusions

The redundant global memory traffic is at the **boundaries** between kernels:

```
Naive:   [reduce writes u32]  →  [Phase1 reads u32]   = 1 MB wasted
         [Phase2 writes u32]  →  [CRT reads u32]       = 1 MB wasted
```

Both pairs can be eliminated by fusing the boundary kernel into the adjacent NTT
phase:

**Fusion 1 — reduce into Phase1 input (`NTTPhase1_i64`):**

Phase1 already reads one `kPerThreadElems`-wide chunk per thread from global
memory. Change its input pointer type from `u32*` to `i64*` and perform the
modular reduction in registers at load time:

```cpp
// Before (existing cheddar): load u32 directly
basic::VectorizedMove<u32, kPerThreadElems>(local, src_limb + x_idx * kPerThreadElems);

// After (adapted): load i64, reduce mod prime
int64_t raw[kPerThreadElems];
basic::VectorizedMove<int64_t, kPerThreadElems>(raw, src_i64 + x_idx * kPerThreadElems);
for (int j = 0; j < kPerThreadElems; j++) {
    int64_t r = raw[j] % (int64_t)prime;
    local[j] = (int32_t)(r < 0 ? r + prime : r);
}
```

The i64 load uses 8-byte elements instead of 4-byte, but the same warp reads
the same number of cache lines. The modular reduction is 1 integer divide per
element (or a Barrett multiply — the compiler will use its fastest path for the
constant prime). This eliminates the separate `ntt120_reduce` launch and its
1 MB write + 1 MB read round-trip.

**Fusion 2 — CRT accumulation into Phase2 output — analyzed, not adopted:**

It is tempting to have `INTTPhase2` write `i128` directly instead of `i32`,
folding the Garner step into the final write loop. The obstacle is coordination:
`INTTPhase2` launches one CTA per prime, and CRT requires all four primes'
residues for the same coefficient simultaneously. Combining them without atomics
requires all four prime-CTAs to rendezvous (cooperative groups or a kernel
barrier), which negates the benefit and adds significant complexity.

The practical boundary is: `INTTPhase2+MultConstNormalize` writes normalized
signed `i32` per prime to a scratch buffer, then a small `ntt120_crt` kernel
reads four `i32` values per coefficient (a 4-way coalesced read) and writes one
`Big32` (four `u32` stores). At blockDim=128 this is 4 loads + 4 stores per
thread — fast enough that the launch overhead is the dominant cost, not the
kernel itself.

**Revised traffic with boundary fusions:**

| Step | Read | Write | Total |
|------|------|-------|-------|
| NTT Phase1 (fused reduce, reads i64) | 512 KB | 1 MB | 1.5 MB |
| NTT Phase2 | 1 MB | 1 MB | 2.0 MB |
| INTT Phase1 | 1 MB | 1 MB | 2.0 MB |
| INTT Phase2 + normalize | 1 MB | 1 MB | 2.0 MB |
| `ntt120_crt` (small: 4×i32 → Big32) | 1 MB | 1 MB | 2.0 MB |
| **Total** | | | **9.5 MB** |

Savings: **2 MB / 17%** with straightforward kernel modifications, no layout
change, no occupancy cost.

### Conclusion: keep prime-major SoA, adapt at the input boundary

The prime-major SoA layout is correct. Cheddar's Phase1/Phase2 structure is
optimal for this layout. The only adaptation needed is:

1. **Adapt `NTTPhase1` to accept `i64` input** — eliminates `ntt120_reduce` as
   a separate kernel. This is a small change to the load path only; all butterfly
   logic is unchanged.
2. **Keep `INTTPhase2` + separate `ntt120_crt`** — the CRT step is fast enough
   (32 B read + 16 B write per coefficient, no compute besides 4 multiplies) that
   a separate kernel launch is not a bottleneck. Full Phase2+CRT fusion adds
   complexity for marginal gain.

This conclusion applies to the **dynamic DFT vector (`VecZnxDft`) only** — it
stays prime-major throughout. SVP, VMP, and convolution prepared families each
get their own specialized layout (coefficient-local q120c, block-major q120c,
and x2-packed i32/q120c respectively) chosen for their specific access pattern,
not for NTT coalescing. See the per-family layout sections below.

---

## New backend type

```rust
pub struct CudaNtt120Backend;
```

Separate from `CudaGpuBackend` (FFT64) because `ScalarPrep` and `ScalarBig` differ:

| Associated type | FFT64 GPU | NTT120 GPU |
|-----------------|-----------|------------|
| `ScalarPrep` | `f64` | `i32` |
| `ScalarBig` | `i64` | `Big32` |
| `Handle` | `CudaFft64Handle` | `CudaNtt120Handle` |

**`ScalarPrep = i32`**: Cheddar's Montgomery arithmetic keeps NTT residues as
signed values in `(-q, q)`. For Primes30, `q < 2^30` so the lazy range `(-q,q)`
and the normalized range `(-q/2, q/2)` both fit in `i32`. Using `i32` throughout
avoids a sign conversion at every NTT output boundary. Each `bytes_of_*` formula
is overridden per prepared-domain family (see sections below); `ScalarPrep` only
anchors the `VecZnxDft` element size directly.

**`ScalarBig = Big32`**: `Big32` is a newtype over `[u32; 4]` (little-endian
128-bit two's-complement), defined with `Copy + Zero + Display + Pod` impls.
`size_of::<Big32>() = 16 = size_of::<i128>()`, so `bytes_of_vec_znx_big` needs
no override. CRT kernels may use transient `__int128` internally but write
`Big32` at the store boundary.

The `ZnxView::at()` host-slice accessor returns `&[ScalarPrep]`. For
device-only prepared containers (`VecZnxDft`, `SvpPPol`, `VmpPMat`) on this
backend it should never be called; the host mirror is only meaningful for
`VecZnx` and `VecZnxBig`.

---

## Design principle: one layout per prepared-domain family

The CPU NTT120 path uses a coefficient-local scalar (`q120b`/`q120c`) across
all prepared containers, which is convenient for host reinterpretation. On GPU,
this is suboptimal because different families have different access patterns:

| Family | Hot operation | Natural layout |
|--------|---------------|----------------|
| `VecZnxDft` | NTT/INTT per prime | prime-major SoA |
| `SvpPPol` | pointwise multiply | coefficient-local q120c |
| `VmpPMat` | row-reduction | block-major q120c (1x per out_vec) |
| `CnvPVecL` | convolution left | x2-packed i32 (pairs NTT coefficients) |
| `CnvPVecR` | convolution right × reuse | x2 q120c |

Using a single layout for all of them causes either bandwidth waste (loading
all four primes when only one is needed) or repeated packing on hot paths.

---

## Per-family layouts

### `VecZnxDft` — batched prime-major SoA, centered `i32`

```
device layout: [batch][prime][coeff]
  batch = col * size + limb
  prime ∈ [0, 4)
  coeff ∈ [0, n)

linear index: ((batch * 4) + prime) * n + coeff
element type: i32  — centered, range (-q, q) under lazy reduction
```

Residues stay signed throughout — no conversion at NTT output boundaries.
Maps to `blockIdx.y = prime`, `blockIdx.z = batch` (or a flattened equivalent).
No transposition needed at any stage.

Buffer sizing override (×4 for 4 primes):

```rust
fn bytes_of_vec_znx_dft(n: usize, cols: usize, size: usize) -> usize {
    4 * n * cols * size * size_of::<i32>()
}
```

### `VecZnxBig` — `Big32` (128-bit in 32-bit limbs)

```
device layout: [batch][coeff][word]
  batch = col * size + limb
  coeff ∈ [0, n)
  word  ∈ [0, 4)

linear index: ((batch * n + coeff) * 4) + word
element type: u32  — little-endian two's-complement 128-bit integer
```

`size_of::<Big32>() = 16`, so `bytes_of_vec_znx_big` uses the default formula
with no override needed. CRT kernels accumulate using transient `__int128` and
write `Big32` at the store boundary.

### `SvpPPol` — coefficient-local q120c

SVP is pointwise: one thread reads one prepared coefficient and one NTT
coefficient, multiplies, and writes. Coefficient-local layout minimises scatter:

```
device layout: [col][coeff][prime][lane]
  lane 0: r mod q          (u32)
  lane 1: r * 2^32 mod q   (u32)

linear index: ((col * n + coeff) * 4 + prime) * 2 + lane
element type: u32
```

Buffer sizing override:

```rust
fn bytes_of_svp_ppol(n: usize, cols: usize) -> usize {
    n * cols * 4 * 2 * size_of::<u32>()  // 4 primes × 2 lanes (q120c)
}
```

Preparation (`svp_prepare`): run forward NTT on the scalar polynomial (same
pipeline as `vec_znx_dft_apply`), then convert from prime-major `i32` NTT output
to the coefficient-local q120c layout.

### `VmpPMat` — block-major q120c

VMP is a row reduction: many input rows contribute to each output coefficient.
Output vectors are flattened the same way as `batch` for `VecZnxDft`:

```
out_vec  = col_out * size + limb         — flattened output vector index
out_vecs = cols_out * size               — total number of output vectors

in_row   = row * cols_in + col_in        — flattened input row
in_rows  = rows * cols_in                — total number of input rows
```

Layout:

```
device layout: [blk][out_vec][in_row][q120c_lane]
  blk     ∈ [0, n/2)       — pair of adjacent NTT coefficient indices
  out_vec ∈ [0, out_vecs)  — flattened output vector = col_out * size + limb
  in_row  ∈ [0, in_rows)   — flattened input row = row * cols_in + col_in
  q120c_lane ∈ [0, 8)      — 4 primes × 2 Montgomery lanes = 8 u32

Lane decomposition:
  lane8 = prime * 2 + prep_lane
  prime     ∈ [0, 4)
  prep_lane ∈ {0, 1}   — 0: r mod q,  1: r * 2^32 mod q

element type: u32

linear index: (((blk * out_vecs) + out_vec) * in_rows + in_row) * 8 + lane8
```

One CTA per `(blk, out_vec)` reduces over `in_rows`.

```
gridDim: (n/2, out_vecs),  blockDim: chosen to parallelise the in_row reduction
```

For each iteration of the in_row loop, a thread loads:
- `input[in_row][coeff]`: 4 × i32 (one NTT coefficient, 4 primes) from `VecZnxDft`
- `mat[blk][out_vec][in_row]`: 8 × u32 (q120c)
- 4 Montgomery multiply-accumulates → one q120b accumulator

The VecZnxDft input (4 × n × i32 = 1 MB for n=2^16) fits comfortably in L2
cache on any target GPU. All `out_vecs` CTAs covering the same `(blk, in_row)`
read the same 4-i32 input slice; the L2 serves them from one DRAM fetch, giving
the same input reuse that an x2 software-pairing would provide — without the
register pressure cost.

Buffer sizing override:

```rust
fn bytes_of_vmp_pmat(
    n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize,
) -> usize {
    let out_vecs = cols_out * size;
    let in_rows = rows * cols_in;
    (n / 2) * out_vecs * in_rows * 8 * size_of::<u32>()
}
```

Preparation (`vmp_prepare`): for each `(col_out, limb)` pair, run forward NTT
into prime-major `i32` form, then convert to q120c (Montgomery constant pair per
prime) and write into the block-major layout.

**Why not x2 output-pairing?** An x2 layout groups two adjacent `out_vec`s per
CTA, saving one input re-read per pair. On CPU this maps to a SIMD register lane;
on GPU the same reuse comes free from the L2 cache across independent CTAs. The
x2 trade-off on GPU is: ~40 registers/thread instead of ~25, lower occupancy,
fewer independent CTAs for the scheduler, and a more complex kernel — in exchange
for savings that the L2 already provides. Profile first; switch to x2 only if
profiling shows L2 misses dominating the input reads.

### `CnvPVecL` — x2-packed signed left operand (q120x2b32)

The convolution apply kernel consumes pairs of adjacent NTT coefficients from
all four primes together. Prepacking into x2 blocks at preparation time removes
repeated hot-path gather on every apply call:

```
device layout: [batch][blk][lane8]
  batch = col * size + limb
  blk   ∈ [0, n/2)             — pair of adjacent coefficient indices
  lane8 ∈ [0, 8)               — one q120x2b32 block = 8 × i32

linear index: ((batch * (n/2)) + blk) * 8 + lane8
lane meaning: lane8 = pair_lane * 4 + prime
  pair_lane ∈ {0, 1}           — which of the two adjacent coefficients
  prime     ∈ [0, 4)
element type: i32  — centered, lazy range (-q, q) after forward NTT
```

Buffer sizing override:

```rust
fn bytes_of_cnv_pvec_l(n: usize, cols: usize, size: usize) -> usize {
    (n / 2) * cols * size * 8 * size_of::<i32>()
}
```

Preparation (`cnv_prepare_left`): run forward NTT into prime-major `i32` form,
then pack pairs of adjacent coefficients into the q120x2b32 block layout.

### `CnvPVecR` — x2-packed prepared right operand (q120x2c)

```
device layout: [batch][blk][lane16]
  blk    ∈ [0, n/2)            — pair of adjacent coefficient indices
  lane16 ∈ [0, 16)             — one q120x2c block = 16 × u32

linear index: ((batch * (n/2)) + blk) * 16 + lane16
lane meaning: lane16 = pair_lane * 8 + prime * 2 + prep_lane
  pair_lane ∈ {0, 1}
  prime     ∈ [0, 4)
  prep_lane ∈ {0, 1}           — 0: r mod q, 1: r * 2^32 mod q
element type: u32
```

Buffer sizing override:

```rust
fn bytes_of_cnv_pvec_r(n: usize, cols: usize, size: usize) -> usize {
    (n / 2) * cols * size * 16 * size_of::<u32>()
}
```

Preparation (`cnv_prepare_right`): same forward NTT + convert to Montgomery
constants (q120c), then pack pairs into the q120x2c block layout.

---

## Handle: `CudaNtt120Handle`

Allocated once at module creation, uploaded to device, never modified.

```rust
pub struct CudaNtt120Handle {
    /// Forward twiddle factors, prime-major [4 × n] u32 (LSB portion)
    twiddle_fwd:     CudaSlice<u32>,
    /// Forward twiddle MSB: [4 × (n / LSB_SIZE)] u32
    twiddle_fwd_msb: CudaSlice<u32>,
    /// Inverse twiddle factors, prime-major [4 × n] u32
    twiddle_inv:     CudaSlice<u32>,
    /// Inverse twiddle MSB
    twiddle_inv_msb: CudaSlice<u32>,
    /// n^{-1} mod Q[k] in Montgomery form, one per prime
    inv_n_mont: [u32; 4],
    /// Primes Q[k]
    primes: CudaSlice<u32>,
    /// Q[k]^{-1} mod 2^32 as i32  — Cheddar's `q_inv` / `InvModBase(q)`
    inv_primes: CudaSlice<i32>,
    log_n: usize,
}
```

Twiddle factors for `Primes30` are generated on the CPU, replicating
Cheddar's `NTTHandler::PopulateTwiddleFactors()`: sequential powers of the
primitive 2n-th root, bit-reversed, converted to Montgomery form (`a * 2^32 mod q`).
This differs from `poulpy-cpu-ref`'s `NttTable` which stores packed u64 pairs.
The LSB/MSB split follows Cheddar's convention: `LSB_SIZE = 32` always, so `MSB_SIZE = n/32`.

---

## Kernel reuse and adaptations from cheddar

| Kernel | Origin | Change |
|--------|--------|--------|
| `NTTPhase1<u32, log_n>` | Cheddar, direct | **Adapted**: (1) load path accepts `i64*` input with inline modular reduction; (2) `blockIdx.z = batch` added (Cheddar has no batch dimension) |
| `NTTPhase2<u32, log_n>` | Cheddar, direct | **Adapted**: `blockIdx.z = batch` added |
| `INTTPhase1<u32, log_n>` | Cheddar, direct | **Adapted**: `blockIdx.z = batch` added |
| `INTTPhase2<u32, log_n, MultConstNormalize>` | Cheddar, direct | **Adapted**: `blockIdx.z = batch` added; `elem_func` applies n^{-1} and centers output to `(-q/2, q/2)` |

`MultConstNormalize` is already defined in `NTTUtils.cuh`:

```cpp
template <typename word>
__device__ __inline__ void MultConstNormalize(
    make_signed_t<word> &result, const make_signed_t<word> a,
    const word b, const word prime, const make_signed_t<word> montgomery)
{
    auto temp = __mult_montgomery_lazy<word>(a, b, prime, montgomery);
    if (temp < 0) temp += prime;
    if (temp > (prime >> 1)) temp -= prime;
    result = temp;
}
```

The per-prime constant `b` is `inv_n_mont[y_idx]`, supplied through `src_const`.

### Adapted `NTTPhase1` load path

`NTTPhase1` uses a **strided** global load (stride = `1 << (log_degree -
kStageMerging)`), not a sequential `VectorizedMove`. This differs from
`INTTPhase1`, which does use a sequential `VectorizedMove`. The adapted i64 load
preserves the same strided pattern:

```cpp
// Original cheddar NTTPhase1 load (strided — NOT sequential like INTTPhase1):
//   int stage_group_idx = threadIdx.x >> kLogWarpBatching;
//   int batch_idx = threadIdx.x & ((1 << kLogWarpBatching) - 1);
//   const u32 *load_ptr = src_limb + batch_idx
//                         + (blockIdx.x << kLogWarpBatching)
//                         + (stage_group_idx << (log_degree - kNumStages));
//   for (int i = 0; i < kPerThreadElems; i++)
//       local[i] = StreamingLoad<u32>(load_ptr + (i << (log_degree - kStageMerging)));

// Adapted load for i64 input — same strided pattern, reduce inline:
int stage_group_idx = threadIdx.x >> kLogWarpBatching;
int batch_idx = threadIdx.x & ((1 << kLogWarpBatching) - 1);
int z = blockIdx.z;   // batch index (new — Cheddar has no blockIdx.z)
const int64_t *load_ptr = global_src_i64 + z * n + batch_idx
                          + (blockIdx.x << kLogWarpBatching)
                          + (stage_group_idx << (log_degree - kNumStages));
for (int i = 0; i < kPerThreadElems; i++) {
    int64_t raw = basic::StreamingLoad<int64_t>(
        load_ptr + (i << (log_degree - kStageMerging)));
    int64_t r = raw % (int64_t)prime;
    local[i] = (int32_t)(r < 0 ? r + (int64_t)prime : r);
}
```

All butterfly logic after this point is unchanged. The `%` operation on a
loop-invariant prime will be strength-reduced by the compiler to a multiply-shift
(Barrett reduction), keeping throughput high.

---

## New kernels to write

### `ntt120_crt` — post-INTT signed i32 residues → Big32 CRT reconstruction

Signed `i32` residues produced by `INTTPhase2+MultConstNormalize` are in
`(-q/2, q/2)`. One thread per coefficient performs Garner reconstruction using
a transient `__int128` accumulator, then decomposes the result into `Big32`
(`[u32; 4]`, little-endian two's-complement) for persistent storage. The four
prime residues for coefficient `i` are at `src[0*n+i]` … `src[3*n+i]` — four
coalesced loads from strided-but-sequential prime blocks:

```cpp
// __constant__ memory:
//   q[4]       = Primes30::Q
//   crt_cst[4] = Primes30::CRT_CST = (Q/Q[k])^{-1} mod Q[k]
//   qoverqk[4] = Q / Q[k]          (~90-bit each, stored as __int128)

__global__ void ntt120_crt(
    uint32_t *dst,       // Big32 output: dst[4*i .. 4*i+3], little-endian
    const int32_t *src,  // prime-major signed i32: src[k*n + i]
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __int128 acc = 0;
    for (int k = 0; k < 4; k++) {
        int64_t r = (int64_t)src[k * n + i];
        int64_t t = (r * (int64_t)crt_cst[k]) % (int64_t)q[k];
        acc += (__int128)t * qoverqk[k];
    }
    // Decompose transient __int128 → Big32 = [u32; 4] little-endian
    uint32_t *out = dst + 4 * i;
    out[0] = (uint32_t)(acc);
    out[1] = (uint32_t)(acc >> 32);
    out[2] = (uint32_t)(acc >> 64);
    out[3] = (uint32_t)(acc >> 96);
}
```

`__int128` lowers to `mul.wide.s64` + `add.cc`/`addc` PTX.
Grid: `gridDim = (ceil(n/128), cols)`, blockDim=128. (`gridDim.y = col` indexes the output column; `i = blockIdx.x * 128 + threadIdx.x` indexes the coefficient.)

### `ntt120_crt_accum` — accumulating variant for multi-limb IDFT

When `vec_znx_idft_apply` processes multiple DFT limbs accumulating into one
`VecZnxBig`, a fused accumulate variant avoids a separate read-modify-write.
It reads the existing `Big32`, rehydrates to `__int128`, adds the Garner result,
and decomposes back to `Big32`:

```cpp
__global__ void ntt120_crt_accum(
    uint32_t *acc,       // Big32 accumulator: acc[4*i .. 4*i+3], read and written
    const int32_t *src,  // prime-major signed residues for this limb
    int n) { /* same Garner body; rehydrate Big32 → __int128, acc += result, decompose */ }
```

---

## Transformation pipeline

### `vec_znx_dft_apply` (VecZnx → VecZnxDft)

All selected `(col, limb)` pairs are processed simultaneously. `batch = col * size + limb`
is the flattened index; no serial loop — all batches launch in parallel.

Grid convention: `gridDim = (x, y=prime, z=batch)`.
- `blockIdx.x` → coefficient group (`coeff = blockIdx.x * blockDim.x * kPerThreadElems`)
- `blockIdx.y` → prime ∈ [0, 4)
- `blockIdx.z` → batch = col × size + limb

```
1. NTTPhase1_i64<u32, log_n>          ← adapted: reads i64 directly, reduces inline
       src: VecZnx i64  [n × i64 per batch]
       dst: VecZnxDft i32  [4n × i32 per batch, prime-major]
       gridDim: (n / (blockDim × kPerThreadElems), 4, batch)
                for log_n=16: (32, 4, batch),  blockDim=128,  kPerThreadElems=16
       shared: blockDim × kPerThreadElems × 4 = 8 KB per CTA

2. NTTPhase2<u32, log_n>              ← unchanged
       gridDim: (n / (blockDim × kPerThreadElems), 4, batch)
                for log_n=16: (128, 4, batch), blockDim=64,   kPerThreadElems=8
       shared: blockDim × kPerThreadElems × 4 = 2 KB per CTA
```

No separate `ntt120_reduce` kernel. The i64 input is consumed directly in
Phase1.

### `vec_znx_idft_apply` (VecZnxDft → VecZnxBig)

For each `col`, accumulate all `size` DFT limbs into the VecZnxBig. Limbs are
processed serially (accumulation dependency); columns are processed in parallel.
Requires a scratch `i32[cols × 4n]` buffer (1 MB per col for n=2^16), allocated
once from `ScratchArena` and reused across limbs.

Grid convention for INTT kernels: `gridDim = (x, y=prime, z=col)`.
- `blockIdx.y` → prime ∈ [0, 4)
- `blockIdx.z` → col (limb is fixed by the outer loop)

Grid convention for CRT: `gridDim = (x, y=col)`.
- `blockIdx.x` → coefficient group
- `blockIdx.y` → col

```
For each limb in 0..size:

  1. INTTPhase1<u32, log_n>            ← unchanged
         src: VecZnxDft i32 for this limb   [4n × i32 per col, prime-major]
         dst: scratch i32[4n] per col
         gridDim: (n / (blockDim × kPerThreadElems), 4, cols)
                  for log_n=16: (128, 4, cols), blockDim=64, kPerThreadElems=8

  2. INTTPhase2<u32, log_n, MultConstNormalize>   ← unchanged
         src_const: inv_n_mont[4]
         src/dst: scratch i32[4n] per col
         output: signed i32[4n] per col in (-q/2, q/2)
         gridDim: (n / (blockDim × kPerThreadElems), 4, cols)
                  for log_n=16: (32, 4, cols), blockDim=128, kPerThreadElems=16

  3. ntt120_crt_accum
         src: scratch i32[4n] per col
         acc: VecZnxBig Big32 ptr        [n × Big32 = n × 4 × u32 per col]
         gridDim: (ceil(n / 128), cols),  blockDim=128
```

### `vec_znx_big_normalize` (VecZnxBig → VecZnx)

No NTT needed. One thread per coefficient.

`VecZnxBig` layout is `[batch][coeff][word]` with `batch = col * size + limb`.
Normalize extracts one `i64` output limb per `(col, limb)` pair (i.e., per batch
entry). All batches are processed in parallel.

Grid convention: `gridDim = (x, y=batch)`.
- `blockIdx.x` → coefficient group (`i = blockIdx.x * 128 + threadIdx.x`)
- `blockIdx.y` → batch = col × size + limb_in_big

```
gridDim: (ceil(n / 128), cols * size),  blockDim=128
```

Steps per thread (coefficient `i`, batch `b`):

1. Load `Big32 = [u32; 4]` at `big[b][i]` — little-endian 128-bit two's-complement.
2. Right-shift by `base2k * output_limb_index` bits using 32-bit sub-word arithmetic
   (shifts and masks on the four `u32` words — no `__int128` needed).
3. Extract the low 64 bits as a signed `i64` and write to the output `VecZnx`.

---

## Files to create

```
poulpy-gpu-cuda/
  doc/
    ntt120-gpu-design.md          ← this file
  src/
    ntt120/
      mod.rs                      — CudaNtt120Backend, CudaNtt120Handle, Backend impl,
                                    bytes_of_* overrides
      hal_impl.rs                 — HalVecZnxDftImpl, HalVecZnxBigImpl, HalModuleImpl
      twiddle.rs                  — CPU-side Primes30 twiddle generation → device upload
      tests.rs                    — parity tests vs NTT120Ref
  cuda/
    ntt120_crt.cu                 — i32[4n] → Big32[n] Garner reconstruction (~60 lines)
    ntt120_ntt.cu                 — wrappers instantiating adapted Cheddar templates
                                    for word=u32, i64-input Phase1, each log_degree
```

Cheddar headers (`NTT.cuh`, `NTTUtils.cuh`, `Basic.cuh`) are included directly.
The adaptation to Phase1 is a targeted change to the load path only; a thin
wrapper template `NTTPhase1_i64` instantiates the adapted version. All five
kernel templates are instantiated for `u32` and `log_degree ∈ {12, 13, 14, 15, 16}`,
compiled by `build.rs` via `cc::Build`.

---

## Implementation order

1. **NTT core** (`VecZnxDft` forward/inverse, `VecZnxBig`): highest value,
   enables testing the round-trip `dft_apply → idft_apply → normalize` against
   `NTT120Ref`.

2. **SVP** (`SvpPPol` preparation + apply): needed for scalar × vector products.

3. **VMP** (`VmpPMat` preparation + apply): needed for key-switching and
   external product.

4. **Convolution** (`CnvPVecL`, `CnvPVecR`): the final-state design uses x2-packed
   `i32` for the left operand and x2-packed q120c for the right operand (see
   per-family layout sections). As a staged rollout, the right operand can
   temporarily fall back to prime-major format and be packed on the fly; switch
   to the prepacked `CnvPVecR` layout once the apply kernel is stable.

---

## What does NOT change

- `CudaGpuBackend`, `CudaFft64Handle` — untouched
- `CudaBuf`, `CudaBufRef`, `CudaBufMut` — shared buffer types, reused
- `cuda_context()`, `cuda_stream()` — shared singletons
- HAL trait definitions in `poulpy-hal` — no changes needed

---

## Correctness invariant

For any `VecZnx a`:

```
NTT120Ref::normalize(NTT120Ref::idft(NTT120Ref::dft(a))) == a
CudaNtt120::normalize(CudaNtt120::idft(CudaNtt120::dft(a))) == a

and for the same a, both produce the same VecZnx output
```

Both backends use the same `Primes30` constants, the same twiddle-factor
definition (`OMEGA[k]`-derived), and the same Garner CRT constants
(`CRT_CST[k]`). The prime-major device layout is internal; only the final
`VecZnx` host bytes need to match.

Parity test: upload a random `VecZnx`, run the full round-trip on both backends,
download and compare host bytes.
