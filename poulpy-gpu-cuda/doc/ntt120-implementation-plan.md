# NTT120 GPU Backend — Implementation Plan

Incremental build order for `CudaNtt120Backend`. Each step is independently
testable. Steps 1–3 have no CUDA compilation dependency. Step 4 is the build
plumbing checkpoint — nothing past it starts until `cargo build` is clean.
Steps 5–7 are the primary correctness gate and should be treated as one
milestone.

---

## Step 1 — `Big32` type
**Pure Rust. No CUDA.**
**File:** `src/ntt120/types.rs`

- `#[repr(C)] pub struct Big32(pub [u32; 4])`
- Impls: `Copy`, `Clone`, `Debug`, `Display` (hex words), `bytemuck::Pod`,
  `bytemuck::Zeroable`, `num_traits::Zero` → `Big32([0u32; 4])`

**Test:** `size_of::<Big32>() == 16`, bytemuck roundtrip.

---

## Step 2 — Backend skeleton
**Pure Rust. No CUDA.**
**File:** `src/ntt120/mod.rs`

- `pub struct CudaNtt120Backend;`
- `pub struct CudaNtt120Handle { /* empty stub */ }`
- `impl Backend for CudaNtt120Backend`:
  - `ScalarPrep = i32`
  - `ScalarBig  = Big32`
  - `OwnedBuf   = CudaBuf`      ← reuse from existing crate
  - `BufRef<'a> = CudaBufRef<'a>`
  - `BufMut<'a> = CudaBufMut<'a>`
  - `Handle     = CudaNtt120Handle`
  - `Location   = Device`
  - `alloc_bytes` / `from_host_bytes` / transfer impls ← copy from `CudaGpuBackend`
- Override `bytes_of_*`:

```rust
bytes_of_vec_znx_dft(n, cols, size)      = 4 * n * cols * size * 4
bytes_of_vec_znx_big                     = default  // Big32 is 16 B = size_of::<i128>()
bytes_of_svp_ppol(n, cols)               = n * cols * 4 * 2 * 4
bytes_of_vmp_pmat(n, rows, ci, co, size) = (n/2) * (co*size) * (rows*ci) * 8 * 4
bytes_of_cnv_pvec_left(n, cols, size)    = (n/2) * cols * size * 8 * 4
bytes_of_cnv_pvec_right(n, cols, size)   = (n/2) * cols * size * 16 * 4
```

**Test:** `bytes_of_vec_znx_dft(1<<16, 2, 4) == expected`.

---

## Step 3 — Twiddle generation + `HalModuleImpl`
**Pure Rust. Device upload only.**
**Files:** `src/ntt120/twiddle.rs`, `src/ntt120/hal_impl.rs`

- Replicate Cheddar's `NTTHandler::PopulateTwiddleFactors()` — sequential
  powers of the primitive 2n-th root, bit-reversed, stored in Montgomery form
  (`a * 2^32 mod q` as `u32`). This is NOT ported from `poulpy-cpu-ref`'s
  `NttTable` which uses a different packed-u64 format.
- Twiddle layout (`LSB_SIZE=32` always, `MSB_SIZE=n/32`):

```
fwd_twiddle:     [4 primes × n]      u32  (LSB portion, Montgomery form)
fwd_twiddle_msb: [4 primes × n/32]   u32  (every 32nd element of fwd_twiddle)
inv_twiddle:     same shape
inv_twiddle_msb: same shape
inv_n_mont:      [u32; 4]            n^{-1} mod q[k] in Montgomery form
primes:          [u32; 4]
inv_primes:      [i32; 4]            q[k]^{-1} mod 2^32 as i32  (Cheddar's InvModBase / q_inv)
```

- Upload all tables via `cuda_stream().clone_htod()`.
- Populate `CudaNtt120Handle` fields.
- `unsafe impl HalModuleImpl<CudaNtt120Backend> for CudaNtt120Backend`

**Test:** `Module::new(1<<16)` constructs without panic.

---

## Step 4 — CUDA build infrastructure
**`build.rs` + stub `.cu` files.**
**Files:** `build.rs`, `cuda/ntt120_ntt.cu`, `cuda/ntt120_crt.cu`

- `build.rs`: `cc::Build::new().cuda(true).include("cheddar-fhe/include/core").file(...).compile(...)`
- Create both `.cu` files as stubs declaring kernel signatures with empty bodies.

**Goal:** `cargo build` succeeds end-to-end before any kernel body is written.

---

## Step 5 — `vec_znx_dft_apply`: `VecZnx → VecZnxDft`
**First real CUDA work.**
**Files:** `cuda/ntt120_ntt.cu`, `src/ntt120/hal_impl.rs`

### Kernel: `NTTPhase1_i64`
Copy `NTTPhase1` from Cheddar; replace load path only:

```cpp
// Original: VectorizedMove<u32, kPerThreadElems>(local, src_u32 + ...);

const int64_t *src = global_src_i64 + blockIdx.z * n + (x_idx << kStageMerging);
int64_t raw[kPerThreadElems];
VectorizedMove<int64_t, kPerThreadElems>(raw, src);
for (int j = 0; j < kPerThreadElems; j++) {
    int64_t r = raw[j] % (int64_t)prime;   // Barrett on loop-invariant prime
    local[j] = (int32_t)(r < 0 ? r + prime : r);
}
```

Instantiate `NTTPhase2<u32, log_n>` unchanged from Cheddar.

### Rust glue
`unsafe impl HalVecZnxDftImpl<CudaNtt120Backend>` — `vec_znx_dft_apply`:

```
gridDim Phase1: (n / (128 × 16), 4, batch),  blockDim=128
gridDim Phase2: (n / (64  ×  8), 4, batch),  blockDim=64
```

**Test:** upload `VecZnx`, run `vec_znx_dft_apply`, download `i32` residues,
compare `r % q` against `NTT120Ref` `q120b` residues for each prime.

---

## Step 6 — `vec_znx_idft_apply`: `VecZnxDft → VecZnxBig`
**Files:** `cuda/ntt120_ntt.cu`, `cuda/ntt120_crt.cu`, `src/ntt120/hal_impl.rs`

### Kernels (NTT)
- Instantiate `INTTPhase1<u32, log_n>` unchanged.
- Instantiate `INTTPhase2<u32, log_n, MultConstNormalize>` unchanged.
  (`MultConstNormalize` already in `NTTUtils.cuh`; pass `inv_n_mont[prime]`
  as `src_const`.)

### Kernels (CRT)
Upload CRT constants to `__constant__` memory at handle init:
`crt_q[4]`, `crt_cst[4]` (Garner constants), `crt_qoverqk[4]` (`__int128`).

```cpp
// ntt120_crt — single-limb, no accumulation
__global__ void ntt120_crt(uint32_t *dst, const int32_t *src, int n) {
    int i = blockIdx.x * 128 + threadIdx.x,  col = blockIdx.y;
    if (i >= n) return;
    __int128 acc = 0;
    for (int k = 0; k < 4; k++) {
        int64_t r = src[col * 4 * n + k * n + i];
        int64_t t = (r * (int64_t)crt_cst[k]) % (int64_t)crt_q[k];
        acc += (__int128)t * crt_qoverqk[k];
    }
    uint32_t *out = dst + (col * n + i) * 4;
    out[0] = (uint32_t)(acc);       out[1] = (uint32_t)(acc >> 32);
    out[2] = (uint32_t)(acc >> 64); out[3] = (uint32_t)(acc >> 96);
}

// ntt120_crt_accum — accumulating variant for multi-limb IDFT
__global__ void ntt120_crt_accum(uint32_t *acc, const int32_t *src, int n) {
    // rehydrate Big32 → __int128, acc += Garner result, decompose back to Big32
}
```

### Rust glue
`vec_znx_idft_apply`:
- Allocate scratch `i32[cols × 4n]` from `ScratchArena`.
- For each limb: launch `INTTPhase1`, `INTTPhase2`, `ntt120_crt_accum`.

```
gridDim INTT:    (n / (blockDim × kPTE), 4, cols)
gridDim CRT acc: (ceil(n / 128), cols),  blockDim=128
```

**Test:** `dft_apply → idft_apply`, compare `Big32` words to `NTT120Ref`
`i128` output.

---

## Step 7 — `vec_znx_big_normalize`: `VecZnxBig → VecZnx`
**Files:** `cuda/ntt120_crt.cu` (append), `src/ntt120/hal_impl.rs`

```cpp
__global__ void ntt120_normalize(
    int64_t *dst, const uint32_t *src,
    int n, int base2k, int limb_idx)
{
    int i = blockIdx.x * 128 + threadIdx.x,  b = blockIdx.y;
    if (i >= n) return;
    // load Big32, right-shift by base2k * limb_idx in 32-bit sub-words,
    // extract low 64 bits as signed i64, write to dst[b * n + i]
}
// gridDim: (ceil(n/128), cols * size),  blockDim=128
```

`unsafe impl HalVecZnxBigImpl<CudaNtt120Backend>` — `vec_znx_big_normalize`.

**TEST: `dft_apply → idft_apply → normalize`.**
**Final `VecZnx` host bytes must equal `NTT120Ref` output.**
**This is the primary correctness gate. Do not proceed to Step 8 until it passes.**

---

## Steps 8–10 — Prepared-domain families
*(Start only after Step 7 passes.)*

### Step 8 — `SvpPPol`: prepare + apply
- Preparation: `dft_apply` on scalar poly → prime-major `i32`, then repack
  kernel: prime-major → coefficient-local `q120c`.
- Apply kernel: for each `(batch, coeff)`: load `q120c` + `i32[4]`, 4 Montgomery MACs.
- **Test:** parity with `NTT120Ref::svp_apply`.

### Step 9 — `VmpPMat`: prepare + apply
- Preparation: `dft_apply` → prime-major `i32` → convert to `q120c`, write
  block-major layout `[blk][out_vec][in_row][lane8]`.
- Apply kernel: `gridDim=(n/2, out_vecs)`, each CTA reduces over `in_rows`.
  Low register pressure — one `q120c` accumulator per prime.
  Switch to x2 output-pairing only if profiling shows L2 misses dominate.
- **Test:** parity with `NTT120Ref::vmp_apply`.

### Step 10 — Convolution: `CnvPVecL` / `CnvPVecR` prepare + apply
- `CnvPVecL`: `dft_apply` → prime-major `i32` → pack x2 adjacent coefficients →
  `q120x2b32` layout.
- `CnvPVecR`: `dft_apply` → prime-major `i32` → `q120c` → pack x2 → `q120x2c`
  layout. (Fall back to on-the-fly packing during development; switch to
  prepacked layout once the apply kernel is stable.)
- Apply kernel: stream x2 blocks, produce `VecZnxDft` output.
- **Test:** parity with `NTT120Ref::cnv_apply`.

---

## Milestone summary

| Steps | Milestone | Gate |
|-------|-----------|------|
| 1–3 | Backend type-system, handle, twiddles | `cargo test` (pure Rust) |
| 4 | Build infrastructure | `cargo build` clean |
| 5 | Forward NTT | Residues match CPU reference |
| 6 | Inverse NTT + CRT | `VecZnxBig` words match CPU reference |
| **7** | **Normalize** | **`VecZnx` output identical to `NTT120Ref`** |
| 8 | SVP | `svp_apply` parity |
| 9 | VMP | `vmp_apply` parity |
| 10 | Convolution | `cnv_apply` parity |
