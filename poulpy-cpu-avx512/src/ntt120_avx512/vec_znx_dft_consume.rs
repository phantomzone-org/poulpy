//! AVX-512F override for the NTT120 `vec_znx_idft_apply_consume` hot path.
//!
//! The shared HAL defaults already cover the whole `vec_znx_dft` theme. This
//! module restores only the AVX-512F accelerated in-place q120b -> i128
//! compaction used by `NTT120Avx512`.

use bytemuck::cast_slice_mut;
use core::arch::x86_64::{__m256i, __m512i, _mm512_extracti64x4_epi64, _mm512_loadu_si512, _mm512_mul_epu32};
use poulpy_cpu_ref::reference::ntt120::{ntt::NttTableInv, primes::Primes30, vec_znx_dft::NttModuleHandle};
use poulpy_hal::layouts::{Data, Module, VecZnxBig, VecZnxDft, VecZnxDftToMut, ZnxInfos, ZnxViewMut};

use super::{
    NTT120Avx512,
    arithmetic_avx512::{
        BARRETT_MU, CRT_VEC, POW16_CRT, POW32_CRT, Q_VEC, QM_HI, QM_LO, QM_MID, TOTAL_Q, TOTAL_Q_MULT, bcast_quad,
        crt_accumulate_avx2, hadd64_pub, reduce_b_and_apply_crt, reduce_b_and_apply_crt_512,
    },
    ntt::intt_avx2,
};

/// AVX-512F accelerated in-place CRT compaction: q120b (32 bytes/coeff) -> i128 (16 bytes/coeff).
///
/// For each DFT block:
/// 1. apply the inverse NTT in place,
/// 2. reduce each 4-residue q120b coefficient against the CRT basis,
/// 3. accumulate the CRT combination directly into an `i128`.
///
/// The 512-bit pair-packed inner loop processes two coefficients per
/// iteration; a 256-bit tail covers the odd-coefficient case.
///
/// # Safety
///
/// - `u64_ptr` must cover `4 * n * n_blocks` `u64` values.
/// - AVX-512F support must be available at runtime.
/// - no aliased references to the same buffer may be live during the call.
#[target_feature(enable = "avx512f")]
unsafe fn compact_all_blocks_avx2(n: usize, n_blocks: usize, u64_ptr: *mut u64, table: &NttTableInv<Primes30>) {
    use core::arch::x86_64::_mm256_loadu_si256;

    let half_q: u128 = TOTAL_Q.div_ceil(2);

    // 256-bit constants for the odd-coefficient tail.
    let q_avx = unsafe { _mm256_loadu_si256(Q_VEC.as_ptr() as *const __m256i) };
    let mu_avx = unsafe { _mm256_loadu_si256(BARRETT_MU.as_ptr() as *const __m256i) };
    let pow32_crt_avx = unsafe { _mm256_loadu_si256(POW32_CRT.as_ptr() as *const __m256i) };
    let pow16_crt_avx = unsafe { _mm256_loadu_si256(POW16_CRT.as_ptr() as *const __m256i) };
    let crt_avx = unsafe { _mm256_loadu_si256(CRT_VEC.as_ptr() as *const __m256i) };
    let qm_hi_avx = unsafe { _mm256_loadu_si256(QM_HI.as_ptr() as *const __m256i) };
    let qm_mid_avx = unsafe { _mm256_loadu_si256(QM_MID.as_ptr() as *const __m256i) };
    let qm_lo_avx = unsafe { _mm256_loadu_si256(QM_LO.as_ptr() as *const __m256i) };

    // 512-bit broadcast constants for the pair-packed inner loop.
    let q_512 = unsafe { bcast_quad(Q_VEC.as_ptr()) };
    let mu_512 = unsafe { bcast_quad(BARRETT_MU.as_ptr()) };
    let pow32_crt_512 = unsafe { bcast_quad(POW32_CRT.as_ptr()) };
    let pow16_crt_512 = unsafe { bcast_quad(POW16_CRT.as_ptr()) };
    let crt_512 = unsafe { bcast_quad(CRT_VEC.as_ptr()) };
    let qm_hi_512 = unsafe { bcast_quad(QM_HI.as_ptr()) };
    let qm_mid_512 = unsafe { bcast_quad(QM_MID.as_ptr()) };
    let qm_lo_512 = unsafe { bcast_quad(QM_LO.as_ptr()) };

    for k in 0..n_blocks {
        let src_start = 4 * n * k;
        let dst_start = 2 * n * k;

        {
            let blk: &mut [u64] = unsafe { std::slice::from_raw_parts_mut(u64_ptr.add(src_start), 4 * n) };
            unsafe { intt_avx2::<Primes30>(table, blk) };
        }

        // Pair-packed compaction: read 2 q120b coefficients per iteration via __m512i;
        // write the 2 i128 results separately (each occupies 2 u64s; in-place compaction
        // is safe because the dst stride (2 u64) is half the src stride (4 u64), so iter c
        // never writes into a yet-to-be-read src slot).
        unsafe {
            let pairs = n / 2;
            let mut c = 0usize;
            for _ in 0..pairs {
                let xv: __m512i = _mm512_loadu_si512(u64_ptr.add(src_start + 4 * c) as *const __m512i);
                let t = reduce_b_and_apply_crt_512(xv, q_512, mu_512, pow32_crt_512, pow16_crt_512, crt_512);
                let p_hi = _mm512_mul_epu32(t, qm_hi_512);
                let p_mid = _mm512_mul_epu32(t, qm_mid_512);
                let p_lo = _mm512_mul_epu32(t, qm_lo_512);

                let pairs_iter = [
                    (
                        _mm512_extracti64x4_epi64::<0>(p_hi),
                        _mm512_extracti64x4_epi64::<0>(p_mid),
                        _mm512_extracti64x4_epi64::<0>(p_lo),
                    ),
                    (
                        _mm512_extracti64x4_epi64::<1>(p_hi),
                        _mm512_extracti64x4_epi64::<1>(p_mid),
                        _mm512_extracti64x4_epi64::<1>(p_lo),
                    ),
                ];
                for (j, (ph, pm, pl)) in pairs_iter.into_iter().enumerate() {
                    let s_hi = hadd64_pub(ph);
                    let s_mid = hadd64_pub(pm);
                    let s_lo = hadd64_pub(pl);
                    let mut v: u128 = ((s_hi as u128) << 64) + ((s_mid as u128) << 32) + (s_lo as u128);
                    let q_approx = (v >> 120) as usize;
                    v -= TOTAL_Q_MULT[q_approx];
                    if v >= TOTAL_Q {
                        v -= TOTAL_Q;
                    }
                    let val: i128 = if v >= half_q { v as i128 - TOTAL_Q as i128 } else { v as i128 };
                    (u64_ptr.add(dst_start + 2 * (c + j)) as *mut i128).write_unaligned(val);
                }
                c += 2;
            }

            // Tail (single 256-bit coefficient when n is odd).
            if n & 1 != 0 {
                let xv: __m256i = _mm256_loadu_si256(u64_ptr.add(src_start + 4 * c) as *const __m256i);
                let t = reduce_b_and_apply_crt(xv, q_avx, mu_avx, pow32_crt_avx, pow16_crt_avx, crt_avx);
                let mut v = crt_accumulate_avx2(t, qm_hi_avx, qm_mid_avx, qm_lo_avx);
                let q_approx = (v >> 120) as usize;
                v -= TOTAL_Q_MULT[q_approx];
                if v >= TOTAL_Q {
                    v -= TOTAL_Q;
                }
                let val: i128 = if v >= half_q { v as i128 - TOTAL_Q as i128 } else { v as i128 };
                (u64_ptr.add(dst_start + 2 * c) as *mut i128).write_unaligned(val);
            }
        }
    }
}

pub(crate) fn vec_znx_idft_apply_consume<D: Data>(
    module: &Module<NTT120Avx512>,
    mut a: VecZnxDft<D, NTT120Avx512>,
) -> VecZnxBig<D, NTT120Avx512>
where
    VecZnxDft<D, NTT120Avx512>: VecZnxDftToMut<NTT120Avx512>,
{
    let table = module.get_intt_table();

    let (n, n_blocks, u64_ptr) = {
        let mut a_mut: VecZnxDft<&mut [u8], NTT120Avx512> = a.to_mut();
        let n = a_mut.n();
        let n_blocks = a_mut.cols() * a_mut.size();
        let ptr: *mut u64 = {
            let s = a_mut.raw_mut();
            cast_slice_mut::<_, u64>(s).as_mut_ptr()
        };
        (n, n_blocks, ptr)
    };

    unsafe { compact_all_blocks_avx2(n, n_blocks, u64_ptr, table) };
    a.into_big()
}
