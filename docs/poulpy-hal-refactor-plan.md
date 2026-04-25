# poulpy-hal Refactor Plan — CPU/GPU Backend Support

## Objective

Refactor `poulpy-hal` so that adding a GPU backend requires only implementing the `Backend`
trait and the `_Backend`-suffix OEP traits — nothing in the hot path should assume host byte
access (`AsRef<[u8]>`).

---

## What stays

- `Backend` trait with `OwnedBuf / BufRef<'a> / BufMut<'a>` associated types
- `Module<B>` as the dispatch handle
- `ScratchArena<'a, B>` bump allocator with `borrow()` reset semantics
- Layout types (`VecZnx<D>`, `VecZnxBig<D,B>`, etc.) parameterized over `D`
- OEP unsafe extension-point layer
- `ScratchArenaTakeBasic` take-pattern for typed temporaries

---

## Host code policy

`poulpy-hal` contains no host-specific code except:

- The abstract host-transfer methods on `Backend` (`copy_to_host`, `from_host_bytes`, etc.)
- Serialization — routed through those transfer methods, no `AsRef<[u8]>` (see below)
- `Source` (ChaCha8) and sampling traits — backend-agnostic by design (see below)
- `HostDataRef`, `HostDataMut`, `HostBackend`, `DeviceBackend` — type-level vocabulary that
  belongs in the HAL as architectural concepts, not CPU-ref implementation details

The following move to `poulpy-cpu-ref`:

| Item | Notes |
|------|-------|
| `ScratchArenaTakeHost` | Removed; CPU defaults now use local typed-take helpers over `ScratchArena::take_region(...)` + `HostBufMut::into_bytes()` |
| Stats / debug layout printing | Require host byte visibility |

`poulpy-core` must use exclusively `BackendRef/Mut` variants so it does not depend on
`poulpy-cpu-ref` for host-byte conversions.

Transitional host wrapper traits such as `VecZnxToRef` / `VecZnxToMut` remain in `poulpy-hal`
only during the Phase 5 compatibility period. They are deleted in Phase 6 once the old
host-facing API family is gone; they are not moved permanently to `poulpy-cpu-ref`.

---

## Serialization

Serialization stays in `poulpy-hal` but routes through the abstract transfer interface on
`Backend` rather than `AsRef<[u8]>` on layout storage:

```rust
fn to_host_bytes(buf: &Self::OwnedBuf) -> Vec<u8>;
fn from_host_bytes(bytes: &[u8]) -> Self::OwnedBuf;
fn copy_to_host(buf: &Self::OwnedBuf, dst: &mut [u8]);
fn copy_from_host(buf: &mut Self::OwnedBuf, src: &[u8]);
```

For CPU this is a memcpy. For GPU it is a device→host transfer. No `HostDataRef` bound
required; works for any backend.

---

## Sampling

The API is **distribution-oriented**: each sampling OEP trait expresses *what* distribution
to produce and *where* to write it. *How* randomness is generated internally is the backend's
concern.

`Source` (ChaCha8) stays in `poulpy-hal` as the top-level entropy source, narrowed to seed
derivation: callers call `source.new_seed() -> [u8; 32]` and pass the seed to the trait.
No backend-specific sampler type threads through the call stack.

```rust
fn vec_znx_fill_uniform(
    &self,
    res: &mut VecZnxBackendMut<'_, B>,
    base2k: usize,
    res_col: usize,
    seed: [u8; 32],
);

fn vec_znx_fill_normal(
    &self,
    res: &mut VecZnxBackendMut<'_, B>,
    res_col: usize,
    noise: &NoiseInfos,
    seed: [u8; 32],
);
```

Each backend's OEP impl uses the seed however is optimal for that distribution:

- **CPU:** initializes a ChaCha8 from the seed, generates directly into the host buffer.
- **GPU:** passes the seed as a kernel constant (counter-based PRNG per thread), or
  initializes a cuRAND generator from the seed and discards it after the call.

This public API remains seed-oriented for simplicity. Backends are still free to implement that
seed contract using either:

- a stateless counter-based PRNG derived from `(seed, index, counter)`, or
- a reusable backend-local sampling context or stream initialized from the seed

So the surface stays simple while still leaving room for production-grade GPU implementations.

**Guarantees:**
- Correct distribution (uniform range, correct Gaussian variance).
- Intra-backend reproducibility: same backend + same seed → same output.
- Cross-backend bit parity is **not** guaranteed. FHE correctness depends on the
  distribution, not on specific bit values.

Sampling traits migrate to `_Backend` signatures in Phase 5. The existing
`source: &mut Source` parameter is replaced by `seed: [u8; 32]` throughout.

---

## Phase 1 — Fix scratch reuse in `poulpy-core`

### 1a — Add explicit reborrow traits for borrowed backend views

The original plan here was to add `VecZnxToBackendRef/Mut` impls for
`VecZnx<B::BufRef<'b>>` / `VecZnx<B::BufMut<'b>>`. That shape does **not**
work: those impls overlap the existing `VecZnx<B::OwnedBuf>` impls under Rust
coherence because `OwnedBuf`, `BufRef<'_>`, and `BufMut<'_>` are associated
types and the compiler cannot assume they are distinct.

Use a separate API family instead:

```rust
pub trait VecZnxReborrowBackendRef<B: Backend> {
    fn reborrow_backend_ref(&self) -> VecZnxBackendRef<'_, B>;
}

pub trait VecZnxReborrowBackendMut<B: Backend> {
    fn reborrow_backend_mut(&mut self) -> VecZnxBackendMut<'_, B>;
}
```

Implement those traits only for mutable backend-borrowed layouts:

- `VecZnx<B::BufMut<'b>>` → `reborrow_backend_ref` and `reborrow_backend_mut`

They are thin method wrappers over the existing `*_backend_*_from_*` helper
functions and do not overlap the owned-buffer `ToBackendRef/Mut` traits.

Do **not** add the same trait for both `BufRef<'_>` and `BufMut<'_>` under one
trait name. That overlaps for the same reason as the original `ToBackendRef`
plan: the compiler cannot assume `BufRef<'_>` and `BufMut<'_>` are distinct
associated types. Shared borrowed views can continue using the existing
free-function adapters or inline `B::view_ref(...)` construction.

### 1b — Refactor `glwe_encrypt_sk_internal`

Status on April 24, 2026: complete.

In `poulpy-core/src/encryption/glwe.rs`:

- Remove `ScratchOwned<BE>: ScratchOwnedAlloc<BE>` and `BE::OwnedBuf: HostDataMut` from impl bounds.
- Replace all `BE::alloc_bytes(...)` and `ScratchOwned::alloc(...)` locals with
  `scratch.take_vec_znx(...)` + `scratch.borrow()` in block scopes.
- Rename `_scratch` back to `scratch`.
- The `*_tmp_bytes` budget already accounts for all temporaries — no size change needed.

Follow the model of `glwe_encrypt_pk_internal`, which already uses the arena correctly.

### 1c — Fix `glwe_trace.rs`

Status on April 24, 2026: complete.

In `poulpy-core/src/glwe_trace.rs`: replace the remaining `BE::alloc_bytes` call with a
`scratch.take_*` carve.

---

## Phase 2 — Clean up the data trait hierarchy

### 2a — Remove backwards-compat aliases

In `poulpy-hal/src/layouts/mod.rs`, remove:

```rust
pub trait DataRef = HostDataRef;
pub trait DataMut = HostDataMut;
```

Replace all uses of `DataRef` → `HostDataRef` and `DataMut` → `HostDataMut` across the
codebase. Makes GPU incompatibility a compile error at the right location.

### 2b — Tighten bounds on host-only API traits

Replace scattered `BE::OwnedBuf: HostDataMut` bounds with more meaningful capability markers.

- Use `BE: HostBackend` when the requirement is only "this backend is host-resident".
- Use a stronger host-visible trait when the API actually needs byte access to borrowed views.

For example:

```rust
pub trait HostVisibleBackend: HostBackend
where
    for<'a> Self::BufRef<'a>: AsRef<[u8]>,
    for<'a> Self::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]>,
{}
```

This avoids giving `HostBackend` stronger guarantees than it actually expresses.

---

## Phase 3 — Complete borrowed backend reborrow coverage for all layout types

The same gap from Phase 1a exists for every layout type that can be carved out
of a `ScratchArena`. After `VecZnx`, apply the same `BufMut<'_>`-only
`ReborrowBackendRef/Mut` pattern to:

| Type | Missing impls |
|------|--------------|
| `VecZnxBig<D, B>` | `ReborrowBackendRef/Mut` for `BufMut<'b>` |
| `VecZnxDft<D, B>` | same |
| `SvpPPol<D, B>` | same |
| `VmpPMat<D, B>` | same |
| `CnvPVecL/R<D, B>` | same |

After this phase, any type carved from a scratch arena can be passed directly to any
backend-native API trait without reaching for manual free-function conversion.

---

## Phase 4 — Retire `Scratch<B>` DST

`Scratch<B>` (`#[repr(C)] struct { _phantom: PhantomData<B>, data: [u8] }`) encodes a
host-byte-slice model that cannot represent device memory. All usage must migrate to
`ScratchArena<'a, B>`.

1. Find all `ScratchFromBytes`, `ScratchAvailable`, `split_at_mut` call sites.
2. Replace each with the `ScratchArena` equivalent.
3. Remove `ScratchFromBytes` and `Scratch<B>` from the public API.
4. Keep `ScratchOwned<B>` — its `.arena()` method is the only entry point needed.

Do this after all hot-path users and OEP traits have migrated to `ScratchArena` (Phase 5).

---

## Phase 5 — Migrate OEP traits to `_Backend` signatures

The OEP file (`poulpy-hal/src/oep/hal_impl.rs`) contains traits that take
`VecZnxToRef/Mut`-style host-byte types. For GPU they must take `VecZnxBackendMut/Ref`.

Two generations already coexist — old style and `_Backend` suffix. The migration:

1. For each operation still in the old style, add or complete a `_Backend` variant.
2. Demote the old host-accessible variant to a blanket wrapper on `HostBackend`:
   ```rust
   impl<B: HostBackend, ...> VecZnxNormalize for Module<B> {
       fn vec_znx_normalize_inplace<A: VecZnxToMut>(&self, a: &mut A, ...) {
           let mut a_mut = a.to_backend_mut();
           self.vec_znx_normalize_inplace_backend(&mut a_mut, ...);
       }
   }
   ```
3. OEP implementors only implement the `_Backend` variant; host wrappers are free blanket
   impls requiring no OEP work.

This includes migrating sampling traits: `source: &mut Source` → `seed: [u8; 32]`.

The old host-facing variants remain in `poulpy-hal` only as transitional compatibility shims
while this migration is in progress. They are not the long-term API surface.

Can be done trait-by-trait, incrementally, without breaking anything else.

---

## Phase 6 — Move host utilities to `poulpy-cpu-ref`

Once `poulpy-core` uses only `BackendRef/Mut` variants (Phase 5 complete):

**Move to `poulpy-cpu-ref`:**
- `ScratchArenaTakeHost` is already removed; keep only local CPU-ref typed-take helpers where host scratch slices are needed
- `layouts/stats.rs` and debug layout printing
- CPU-specific host defaults and helper algorithms that still require raw slices

**Keep in `poulpy-hal` temporarily, then delete once migration is complete:**
- transitional host wrapper traits such as `VecZnxToRef` / `VecZnxToMut`
- blanket wrappers from old host-facing APIs to `_Backend` APIs

These traits stay only long enough to support the Phase 5 compatibility layer. Once the old
host-facing API family is gone, they should be removed entirely rather than moved permanently
into `poulpy-cpu-ref`.

**Rewrite in place in `poulpy-hal`** (remove `AsRef<[u8]>` bounds):
- `layouts/serialization.rs` — use `Backend::copy_to_host` / `from_host_bytes`

---

## Priority

Phases 1–3 are low-risk and unblock both the immediate correctness issue (hot-path
allocations in `glwe_encrypt_sk_internal`) and future GPU work (scratch-carved types usable
in backend-native APIs).

Phases 4–5 are the GPU-enabling work, done incrementally per trait.

Phase 6 can only happen after Phase 5 is complete.
