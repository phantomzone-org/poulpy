# ScratchArenaTakeHost Reduction Handoff

Current task:
- reduce `ScratchArenaTakeHost` usage in `poulpy-cpu-ref` defaults by replacing it with local `ScratchArena::take_region(...)` + `HostBufMut::into_bytes()` helpers
- user instruction still applies: do not change algorithms/control flow; if a refactor starts cascading unexpectedly, back out and reassess

## What has already landed and is green

- `Scratch<B>` was fully removed from live code earlier.
- post-`Scratch<B>` cleanup is done.
- `ScratchArenaTakeHost` was already removed from:
  - [poulpy-cpu-ref/src/hal_defaults/vec_znx_dft.rs](/home/pro7ech/gausslabs/poulpy_ckks/poulpy-cpu-ref/src/hal_defaults/vec_znx_dft.rs)
  - [poulpy-cpu-ref/src/hal_defaults/vmp_pmat.rs](/home/pro7ech/gausslabs/poulpy_ckks/poulpy-cpu-ref/src/hal_defaults/vmp_pmat.rs)
  - [poulpy-cpu-ref/src/hal_defaults/convolution.rs](/home/pro7ech/gausslabs/poulpy_ckks/poulpy-cpu-ref/src/hal_defaults/convolution.rs)
- those all use a local helper of the form:
  - `take_host_typed<'a, BE, T>(arena: ScratchArena<'a, BE>, len: usize) -> (&'a mut [T], ScratchArena<'a, BE>)`
  - bounds: `BE::BufMut<'a>: HostBufMut<'a>`
- validation after those slices:
  - `cargo check -p poulpy-cpu-ref --message-format short` passed
  - `cargo check --workspace --message-format short` passed

## What was just edited and is now validated

- [poulpy-cpu-ref/src/hal_defaults/vec_znx_big.rs](/home/pro7ech/gausslabs/poulpy_ckks/poulpy-cpu-ref/src/hal_defaults/vec_znx_big.rs)
  - replaced `api::ScratchArenaTakeHost` import with `api::HostBufMut`
  - added local `take_host_typed(...)` helper
  - replaced 4 `ScratchArenaTakeHost` bounds/usages:
    - FFT64 normalize tmp `i64`
    - FFT64 automorphism inplace tmp `i64`
    - NTT120 normalize tmp `i128`
    - NTT120 automorphism inplace tmp `i128`
- [poulpy-cpu-ref/src/hal_defaults/vec_znx.rs](/home/pro7ech/gausslabs/poulpy_ckks/poulpy-cpu-ref/src/hal_defaults/vec_znx.rs)
  - replaced `api::ScratchArenaTakeHost` import with `api::HostBufMut`
  - added local `take_host_typed(...)` helper
  - replaced all visible `ScratchArenaTakeHost`-based `take_i64(...)` usages/bounds with `BE::BufMut<'s>: HostBufMut<'s>` plus `take_host_typed::<BE, i64>(...)`
  - touched normalize / normalize_inplace / rsh / lsh / rotate_inplace / automorphism_inplace / mul_xp_minus_one_inplace / split_ring / merge_rings paths

## Validation result

- `cargo check -p poulpy-cpu-ref --message-format short` passed
- `cargo check --workspace --message-format short` passed
- `rg -n "ScratchArenaTakeHost" poulpy-cpu-ref/src -S` returned no matches
- `ScratchArenaTakeHost` is now gone from `poulpy-cpu-ref/src`

## Important

- the slice described in this handoff is complete
- `ScratchArenaTakeHost` has also been removed from `poulpy-hal/src/api/scratch.rs`
- `HostBufMut` remains in `poulpy-hal` as the narrow host-visible borrowed-buffer capability used by CPU defaults
- no algorithm/control-flow changes were made in this seam, only host scratch typed-take plumbing

## Expected outcome if current patch is correct

- `ScratchArenaTakeHost` should disappear entirely from `poulpy-cpu-ref/src`

## If it fails

- likely causes are lifetime/bound mismatches in `vec_znx.rs`, because that file has many `BE::BufMut<'r>` / `BE::BufMut<'s>` interactions
- safest fix style is:
  - keep the local helper
  - add `BE: 's` where already used
  - require `BE::BufMut<'s>: HostBufMut<'s>` only on the exact functions needing scratch conversion
- do not widen this into a broader HAL/API rewrite unless necessary

## Current modified files in working tree from this last slice

- none required for this slice anymore

## Known good pattern to copy from

- `poulpy-cpu-ref/src/hal_defaults/vmp_pmat.rs`
- `poulpy-cpu-ref/src/hal_defaults/convolution.rs`
- `poulpy-cpu-ref/src/hal_defaults/vec_znx_dft.rs`

## Overall project state before the last two unverified edits

- workspace was green
- no algorithm/control-flow changes were made in this seam, only host scratch typed-take plumbing
