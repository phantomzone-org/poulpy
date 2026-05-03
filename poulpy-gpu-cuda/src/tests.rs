use poulpy_hal::{
    api::{ScratchArenaTakeBasic, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, ScratchOwned, VecZnx, VecZnxReborrowBackendMut, VecZnxReborrowBackendRef},
};

use crate::{CudaBufRef, CudaGpuBackend};

fn panic_message(err: Box<dyn std::any::Any + Send>) -> Option<String> {
    if let Some(msg) = err.downcast_ref::<String>() {
        Some(msg.clone())
    } else {
        err.downcast_ref::<&'static str>().map(|msg| (*msg).to_string())
    }
}

fn is_cuda_unavailable(msg: &str) -> bool {
    msg.contains("failed to initialize CUDA context for device 0")
        || msg.contains("Unable to dynamically load the \"cuda\" shared library")
}

fn sized_sample_vec_znx(n: usize, cols: usize, size: usize, modulus: i64, bias: i64, scale: i64) -> VecZnx<Vec<u8>> {
    // Build a host-owned VecZnx filled with a deterministic coefficient pattern.
    let mut v = VecZnx::from_data(
        HostBytesBackend::alloc_bytes(VecZnx::<Vec<u8>>::bytes_of(n, cols, size)),
        n,
        cols,
        size,
    );
    for (i, coeff) in bytemuck::cast_slice_mut::<u8, i64>(&mut v.data).iter_mut().enumerate() {
        // Reinterpret the backing bytes as i64 coefficients and fill them.
        *coeff = ((i as i64 % modulus) - bias) * scale;
    }
    v
}

fn vec_range_cuda<'a>(vec: &VecZnx<crate::CudaBufMut<'a>>) -> std::ops::Range<usize> {
    // Recover the byte range carved out of the CUDA-owned scratch buffer.
    vec.data.offset..vec.data.offset + vec.data.len
}

fn vec_range_cuda_ref<'a>(vec: &VecZnx<CudaBufRef<'a>>) -> std::ops::Range<usize> {
    // Recover the byte range of a shared CUDA-native VecZnx view.
    vec.data.offset..vec.data.offset + vec.data.len
}

#[test]
fn cuda_scratch_arena_take_vec_znx_roundtrips_through_device() {
    // Verify that ScratchArena carving on the CUDA backend produces disjoint
    // regions in CUDA-owned storage and that explicit host/device transfer can
    // populate those regions without exposing the owned buffer as a host slice.
    let n = 64;
    let cols = 2;
    let size = 3;
    // Compute the byte footprint of one carved VecZnx temporary.
    let bytes = VecZnx::<Vec<u8>>::bytes_of(n, cols, size);

    // Prepare deterministic host-side payloads for the two scratch regions.
    let lhs_host = sized_sample_vec_znx(n, cols, size, 17, 8, 3);
    let rhs_host = sized_sample_vec_znx(n, cols, size, 19, 9, 7);

    // Allocate one CUDA-owned scratch buffer large enough for two temporaries.
    let mut scratch = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ScratchOwned::<CudaGpuBackend>::alloc(2 * bytes)
    })) {
        Ok(scratch) => scratch,
        Err(err) => {
            let msg = panic_message(err).unwrap_or_else(|| "unknown CUDA initialization failure".to_string());
            if is_cuda_unavailable(&msg) {
                eprintln!("skipping CUDA scratch test: {msg}");
                return;
            }
            panic!("{msg}");
        }
    };
    let (lhs_range, rhs_range) = {
        // Borrow the owned scratch as a backend-native ScratchArena.
        let arena = scratch.borrow();
        // Carve the first temporary VecZnx out of the CUDA scratch arena.
        let (lhs, arena) = arena.take_vec_znx_scratch(n, cols, size);
        // Carve the second temporary from the remaining arena.
        let (rhs, _) = arena.take_vec_znx_scratch(n, cols, size);

        // Both carved views should refer to the same owned CUDA buffer.
        assert_eq!(lhs.data.ptr, rhs.data.ptr);
        // Each carved view should have exactly the expected byte length.
        assert_eq!(lhs.data.len, lhs_host.data.len());
        assert_eq!(rhs.data.len, rhs_host.data.len());
        // The carved regions must be non-overlapping.
        assert!(vec_range_cuda(&lhs).end <= vec_range_cuda(&rhs).start);
        // Keep only the byte ranges so the mutable scratch views can drop.
        (vec_range_cuda(&lhs), vec_range_cuda(&rhs))
    };

    // Download the CUDA-owned scratch buffer into a temporary host Vec<u8>.
    let mut host = <CudaGpuBackend as Backend>::to_host_bytes(&scratch.data);
    // Fill the first carved region inside that host copy.
    host[lhs_range.clone()].copy_from_slice(&lhs_host.data);
    // Fill the second carved region inside that host copy.
    host[rhs_range.clone()].copy_from_slice(&rhs_host.data);
    // Upload the modified host bytes back into the CUDA-owned scratch buffer.
    <CudaGpuBackend as Backend>::copy_from_host(&mut scratch.data, &host);

    // Download again and verify that both carved regions survived the roundtrip.
    let roundtrip = <CudaGpuBackend as Backend>::to_host_bytes(&scratch.data);
    assert_eq!(&roundtrip[lhs_range.clone()], lhs_host.data.as_slice());
    assert_eq!(&roundtrip[rhs_range.clone()], rhs_host.data.as_slice());
}

#[test]
fn cuda_scratch_vec_znx_supports_backend_reborrows() {
    let n = 64;
    let cols = 2;
    let size = 3;
    let bytes = VecZnx::<Vec<u8>>::bytes_of(n, cols, size);

    let mut scratch =
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ScratchOwned::<CudaGpuBackend>::alloc(bytes))) {
            Ok(scratch) => scratch,
            Err(err) => {
                let msg = panic_message(err).unwrap_or_else(|| "unknown CUDA initialization failure".to_string());
                if is_cuda_unavailable(&msg) {
                    eprintln!("skipping CUDA backend reborrow test: {msg}");
                    return;
                }
                panic!("{msg}");
            }
        };

    let arena = scratch.borrow();
    let (mut vec, _) = arena.take_vec_znx_scratch(n, cols, size);
    let raw_range = vec_range_cuda(&vec);
    let raw_ptr = vec.data.ptr;
    let raw_len = vec.data.len;

    let backend_ref: VecZnx<CudaBufRef<'_>> =
        <VecZnx<crate::CudaBufMut<'_>> as VecZnxReborrowBackendRef<CudaGpuBackend>>::reborrow_backend_ref(&vec);
    assert_eq!(backend_ref.n(), n);
    assert_eq!(backend_ref.cols(), cols);
    assert_eq!(backend_ref.size(), size);
    assert_eq!(backend_ref.max_size(), size);
    assert_eq!(backend_ref.data.ptr, raw_ptr);
    assert_eq!(backend_ref.data.len, raw_len);
    assert_eq!(vec_range_cuda_ref(&backend_ref), raw_range);

    let backend_mut: VecZnx<crate::CudaBufMut<'_>> =
        <VecZnx<crate::CudaBufMut<'_>> as VecZnxReborrowBackendMut<CudaGpuBackend>>::reborrow_backend_mut(&mut vec);
    assert_eq!(backend_mut.n(), n);
    assert_eq!(backend_mut.cols(), cols);
    assert_eq!(backend_mut.size(), size);
    assert_eq!(backend_mut.max_size(), size);
    assert_eq!(backend_mut.data.ptr, raw_ptr);
    assert_eq!(backend_mut.data.len, raw_len);
    assert_eq!(vec_range_cuda(&backend_mut), raw_range);
}
