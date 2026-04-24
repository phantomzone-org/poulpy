# `poulpy-gpu-cuda`

Minimal opt-in CUDA backend for `poulpy`.

This crate is not meant to be a full GPU backend yet. Its job is to document
the smallest useful pattern for integrating an external device backend into the
current `poulpy` architecture.

## Status

The crate currently owns:

- backend type: `CudaGpuBackend`
- backend buffer type: `CudaBuf`
- explicit CPU <-> CUDA transfer through `TransferFrom`
- module construction via the existing FFT64 handle path
- scratch allocation via the existing default scratch path
- backend-owned `VecZnx` copy
- backend-owned `VecZnx` addition

The crate is opt-in at the workspace level:

- plain root `cargo check` / `cargo test` skip it
- explicit `cargo check -p poulpy-gpu-cuda`
- explicit `cargo test -p poulpy-gpu-cuda`

## Architecture

### 1. Backend Type

`CudaGpuBackend` is a normal `Device` backend.

It is not a "hybrid backend". Mixed CPU/GPU execution is handled by explicit
transfer and delegation elsewhere in the stack. This crate itself is just the
CUDA backend.

### 2. Buffer Model

`CudaBuf` contains:

- `host: Vec<u8>`
- `device: Mutex<Option<CudaSlice<u8>>>`
- `device_stale: AtomicBool`

Why both host and device?

- The current HAL still has important host-readable surfaces.
- Backend-owned families in this crate execute against CUDA buffers directly.
- The host mirror keeps old surfaces working while we add new device-native
  families incrementally.

What `device_stale` means:

- host writes happened through `view_mut` / `region_mut`
- device memory must be refreshed before a CUDA op reads it

The helper `ensure_device_current()` performs that synchronization.

### 3. CUDA Runtime Layer

The crate uses `cudarc` rather than handwritten CUDA FFI.

Current usage:

- `CudaContext`
- `CudaStream`
- `CudaSlice<u8>`
- `memcpy_htod`
- `memcpy_dtoh`
- `memcpy_dtod`
- kernel launch through `LaunchConfig` + `launch_builder`

The current backend uses one shared context on device ordinal `0` and the
default stream. That keeps the prototype simple and deterministic.

### 4. Module / ScratchArena

This crate does not try to differentiate module construction yet.

It reuses:

- `HalModuleImpl` through `FFT64ModuleDefaults`

Scratch management follows the backend-native `ScratchArena` path directly.
That is deliberate. The backend-specific value in this crate is device-owned
`VecZnx` execution, not reintroducing deleted legacy scratch plumbing.

### 5. Backend-Native HAL Families

The important design point is that CUDA-owned buffers can still participate in
the modern `ScratchArena` flow even when higher-level arithmetic remains host
driven.

Current focus:

- backend-owned byte storage
- explicit CPU/CUDA transfer hooks
- `ScratchOwned<CudaGpuBackend>` borrowing into a CUDA-backed `ScratchArena`
- typed arena carving such as `take_vec_znx(...)`

That gives us a minimal, testable device slice without committing to a wider
backend-native arithmetic API yet.

## How To Add Another Device Family

Recommended sequence:

1. Decide whether the family really needs a backend-owned surface.
   If the old API only exposes host slices, add a new backend-owned trait
   instead of pretending the old one is device-native.

2. Add the safe API trait in `poulpy-hal/src/api/...`.

3. Add the backend OEP trait in `poulpy-hal/src/oep/hal_impl.rs`.

4. Add the blanket delegate in `poulpy-hal/src/delegates/...`.

5. Implement that one trait in `poulpy-gpu-cuda/src/hal_impl.rs`.

6. Keep host/device coherence explicit.
   Update the host mirror intentionally.
   Push host changes to device only when needed.

7. Add a focused parity test:
   upload inputs
   run CUDA op
   download output
   compare with a CPU reference result

## Testing

Current focused tests:

- transfer roundtrip preserves bytes
- backend-owned copy matches `FFT64Ref`
- backend-owned add matches `FFT64Ref`

Recommended commands:

```bash
cargo check -p poulpy-gpu-cuda
cargo test -p poulpy-gpu-cuda
```

## Current Limitations

- still keeps a host mirror
- fixed to CUDA device ordinal `0`
- uses the default stream only
- only a very small subset of backend-owned `VecZnx` families exist
- does not yet implement a full GPU-native core/schemes path

That is intentional. This crate is a documented minimal backend slice, not the
finished GPU backend.
