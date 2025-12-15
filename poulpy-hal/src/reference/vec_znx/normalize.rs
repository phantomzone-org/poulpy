use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use crate::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes},
    layouts::{Backend, FillUniform, Module, ScratchOwned, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::znx::{
        ZnxAddInplace, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulPowerOfTwoInplace, ZnxNormalizeDigit, ZnxNormalizeFinalStep,
        ZnxNormalizeFinalStepInplace, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeFirstStepInplace,
        ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepInplace, ZnxZero,
    },
    source::Source,
};

pub fn vec_znx_normalize_tmp_bytes(n: usize) -> usize {
    3 * n * size_of::<i64>()
}

#[allow(clippy::too_many_arguments)]
pub fn vec_znx_normalize<R, A, ZNXARI>(
    res_offset: i64,
    res_base2k: usize,
    res: &mut R,
    res_col: usize,
    a_base2k: usize,
    a: &A,
    a_col: usize,
    carry: &mut [i64],
) where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxZero
        + ZnxCopy
        + ZnxAddInplace
        + ZnxMulPowerOfTwoInplace
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFinalStep
        + ZnxNormalizeFirstStep
        + ZnxExtractDigitAddMul
        + ZnxNormalizeMiddleStepInplace
        + ZnxNormalizeFinalStepInplace
        + ZnxNormalizeDigit,
{
    match res_base2k == a_base2k {
        true => vec_znx_normalize_inter_base2k::<R, A, ZNXARI>(res_base2k, res, res_offset, res_col, a, a_col, carry),
        false => vec_znx_normalize_cross_base2k::<R, A, ZNXARI>(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry),
    }
}

fn vec_znx_normalize_inter_base2k<R, A, ZNXARI>(
    base2k: usize,
    res: &mut R,
    res_offset: i64,
    res_col: usize,
    a: &A,
    a_col: usize,
    carry: &mut [i64],
) where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxZero
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFinalStepInplace
        + ZnxNormalizeMiddleStepInplace,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert!(carry.len() >= vec_znx_normalize_tmp_bytes(res.n()) / size_of::<i64>());
        assert_eq!(res.n(), a.n());
    }

    let n: usize = res.n();
    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let (carry, _) = carry.split_at_mut(n);

    let mut lsh: i64 = res_offset % base2k as i64;
    let mut limbs_offset: i64 = res_offset / base2k as i64;

    // If res_offset is negative, makes it positive
    // and corrects by adding an additional offset
    // on the limbs.
    if res_offset < 0 && lsh != 0 {
        lsh = (lsh + base2k as i64) % (base2k as i64);
        limbs_offset -= 1;
    }

    let lsh_pos: usize = lsh as usize;

    let res_end: usize = (-limbs_offset).clamp(0, res_size as i64) as usize;
    let res_start: usize = (a_size as i64 - limbs_offset).clamp(0, res_size as i64) as usize;
    let a_end: usize = limbs_offset.clamp(0, a_size as i64) as usize;
    let a_start: usize = (res_size as i64 + limbs_offset).clamp(0, a_size as i64) as usize;

    let a_out_range: usize = a_size.saturating_sub(a_start);

    // Computes the carry over the discarded limbs of a
    for j in 0..a_out_range {
        if j == 0 {
            ZNXARI::znx_normalize_first_step_carry_only(base2k, lsh_pos, a.at(a_col, a_size - j - 1), carry);
        } else {
            ZNXARI::znx_normalize_middle_step_carry_only(base2k, lsh_pos, a.at(a_col, a_size - j - 1), carry);
        }
    }

    // If no limbs were discarded, initialize carry to zero
    if a_out_range == 0 {
        ZNXARI::znx_zero(carry);
    }

    // Zeroes bottom limbs that will not be interacted with
    for j in res_start..res_size {
        ZNXARI::znx_zero(res.at_mut(res_col, j));
    }

    let mid_range: usize = a_start.saturating_sub(a_end);

    // Regular normalization over the overlapping limbs of res and a.
    for j in 0..mid_range {
        ZNXARI::znx_normalize_middle_step(
            base2k,
            lsh_pos,
            res.at_mut(res_col, res_start - j - 1),
            a.at(a_col, a_start - j - 1),
            carry,
        );
    }

    // Propagates the carry over the non-overlapping limbs between res and a
    for j in 0..res_end {
        if j == res_end - 1 {
            ZNXARI::znx_normalize_final_step_inplace(base2k, lsh_pos, res.at_mut(res_col, res_end - j - 1), carry);
        } else {
            ZNXARI::znx_normalize_middle_step_inplace(base2k, lsh_pos, res.at_mut(res_col, res_end - j - 1), carry);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn vec_znx_normalize_cross_base2k<R, A, ZNXARI>(
    res: &mut R,
    res_base2k: usize,
    res_offset: i64,
    res_col: usize,
    a: &A,
    a_base2k: usize,
    a_col: usize,
    carry: &mut [i64],
) where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxZero
        + ZnxCopy
        + ZnxAddInplace
        + ZnxMulPowerOfTwoInplace
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFinalStep
        + ZnxNormalizeFirstStep
        + ZnxExtractDigitAddMul
        + ZnxNormalizeMiddleStepInplace
        + ZnxNormalizeFinalStepInplace
        + ZnxNormalizeDigit,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert!(carry.len() >= vec_znx_normalize_tmp_bytes(res.n()) / size_of::<i64>());
        assert_eq!(res.n(), a.n());
    }

    let n: usize = res.n();
    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let (a_norm, carry) = carry.split_at_mut(n);
    let (res_carry, a_carry) = carry[..2 * n].split_at_mut(n);
    ZNXARI::znx_zero(res_carry);

    // Total precision (in bits) that `a` and `res` can represent.
    let a_tot_bits: usize = a_size * a_base2k;
    let res_tot_bits: usize = res_size * res_base2k;

    // Derive intra-limb shift and cross-limb offset.
    let mut lsh: i64 = res_offset % a_base2k as i64;
    let mut limbs_offset: i64 = res_offset / a_base2k as i64;

    // If res_offset is negative, ensures it is positive
    // and corrects by incrementing the cross-limb offset.
    if res_offset < 0 && lsh != 0 {
        lsh = (lsh + a_base2k as i64) % (a_base2k as i64);
        limbs_offset -= 1;
    }

    let lsh_pos: usize = lsh as usize;

    // Derive start/stop bit indexes of the overlap between `a` and `res` (after taking into account the offset)..
    let res_end_bit: usize = (-limbs_offset * a_base2k as i64).clamp(0, res_tot_bits as i64) as usize; // Stop bit
    let res_start_bit: usize = (a_tot_bits as i64 - limbs_offset * a_base2k as i64).clamp(0, res_tot_bits as i64) as usize; // Start bit
    let a_end_bit: usize = (limbs_offset * a_base2k as i64).clamp(0, a_tot_bits as i64) as usize; // Stop bit
    let a_start_bit: usize = (res_tot_bits as i64 + limbs_offset * a_base2k as i64).clamp(0, a_tot_bits as i64) as usize; // Start bit

    // Convert bits to limb indexes.
    let res_end: usize = res_end_bit / res_base2k;
    let res_start: usize = res_start_bit.div_ceil(res_base2k);
    let a_end: usize = a_end_bit / a_base2k;
    let a_start: usize = a_start_bit.div_ceil(a_base2k);

    // Zero all limbs of `res`. Unlike the simple case
    // where `res_base2k` is equal to `a_base2k`, we also
    // need to ensure that the limbs starting from `res_end`
    // are zero.
    for j in 0..res_size {
        ZNXARI::znx_zero(res.at_mut(res_col, j));
    }

    // Case where offset is positive and greater or equal
    // to the precision of a.
    if res_start == 0 {
        return;
    }

    // Limbs of `a` that have a greater precision than `res`.
    let a_out_range: usize = a_size.saturating_sub(a_start);

    for j in 0..a_out_range {
        if j == 0 {
            ZNXARI::znx_normalize_first_step_carry_only(a_base2k, lsh_pos, a.at(a_col, a_size - j - 1), a_carry);
        } else {
            ZNXARI::znx_normalize_middle_step_carry_only(a_base2k, lsh_pos, a.at(a_col, a_size - j - 1), a_carry);
        }
    }

    // Zero carry if the above loop didn't trigger.
    if a_out_range == 0 {
        ZNXARI::znx_zero(a_carry);
    }

    // How much is left to accumulate to fill a limb of `res`.
    let mut res_acc_left: usize = res_base2k;

    // Starting limb of `res`.
    let mut res_limb: usize = res_start - 1;

    // How many limbs of `a` overlap with `res` (after taking into account the offset).
    let mid_range: usize = a_start.saturating_sub(a_end);

    // Regular normalization over the overlapping limbs of res and a.
    'outer: for j in 0..mid_range {
        let a_limb: usize = a_start - j - 1;

        // Current res & a limbs
        let a_slice: &[i64] = a.at(a_col, a_limb);

        // Trackers: wow much of a_norm is left to
        // be flushed on res.
        let mut a_take_left: usize = a_base2k;

        // Normalizes the j-th limb of a and store the results into `a_norm``.
        // This step is required to avoid overflow in the next step,
        // which assumes that |a| is bounded by 2^{a_base2k -1} (i.e. normalized).
        ZNXARI::znx_normalize_middle_step(a_base2k, lsh_pos, a_norm, a_slice, a_carry);

        // In the first iteration we need to match the precision `res` and `a`.
        if j == 0 {
            // Case where `a` has more precision than `res` (after taking into account the offset)
            //
            // For example:
            //
            // a:      [x  x  x  x  x][x  x  x  x  x][x  x  x  x  x][x  x  x  x  x]
            // res: [x  x  x  x  x  x][x  x  x  x  x  x][x  x  x  x  x  x]
            if !(a_tot_bits - a_start_bit).is_multiple_of(a_base2k) {
                let take: usize = (a_tot_bits - a_start_bit) % a_base2k;
                ZNXARI::znx_mul_power_of_two_inplace(-(take as i64), a_norm);
                a_take_left -= take;
            // Case where `res` has more precision than `a` (after taking into account the offset)
            //
            // For example:
            //
            // a:    [x  x  x  x  x][x  x  x  x  x][x  x  x  x  x][x  x  x  x  x]
            // res:           [x  x  x  x  x  x][x  x  x  x  x  x][x  x  x  x  x  x]
            } else if !(res_tot_bits - res_start_bit).is_multiple_of(res_base2k) {
                res_acc_left -= (res_tot_bits - res_start_bit) % res_base2k;
            }
        }

        // Extract bits of `a_norm` and accumulates them on res[res_limb] until
        // res_base2k bits have been accumulated or until all bits of `a` are
        // extracted.
        'inner: loop {
            // Current limb of res
            let res_slice: &mut [i64] = res.at_mut(res_col, res_limb);

            // We can take at most a_base2k bits
            // but not more than what is left on a_norm or what is left to
            // fully populate the current limb of res.
            let a_take: usize = a_base2k.min(a_take_left).min(res_acc_left);

            if a_take != 0 {
                // Extract `a_take` bits from a_norm and accumulates them on `res_slice`.
                let scale: usize = res_base2k - res_acc_left;
                ZNXARI::znx_extract_digit_addmul(a_take, scale, res_slice, a_norm);
                a_take_left -= a_take;
                res_acc_left -= a_take;
            }

            // If either:
            //  * At least `res_base2k` bits have been accumulated
            //  * We have reached the last limb of a
            // Then: Flushes them onto res
            if res_acc_left == 0 || a_limb == 0 {
                // This case happens only if `res_offset` is negative.
                // If `res_offset` is negative, we need to apply the offset BEFORE
                // the normalization to ensure the `res-offset` overflowing bits of `a`
                // are in the MSB of `res` instead of being discarded.
                if a_limb == 0 && a_take_left == 0 {
                    // TODO: prove no overflow can happen here (should not intuitively)
                    ZNXARI::znx_add_inplace(a_carry, a_norm);

                    // Usual case where for example
                    // a:   [     overflow     ][x  x  x  x  x][x  x  x  x  x][x  x  x  x  x][x  x  x  x  x]
                    // res:      [x  x  x  x  x  x][x  x  x  x  x  x][x  x  x  x  x  x][x  x  x  x  x  x]
                    //
                    // where [overflow] are the overflowing bits of `a` (note that they are not a limb, but
                    // stored in a[0] & carry from a[1]) that are moved into the MSB of `res` due to the
                    // negative offset.
                    //
                    // In this case we populate what is left of `res_acc_left` using `a_carry`
                    //
                    // TODO: see if this can be simplified (e.g. just add).
                    if res_acc_left != 0 {
                        let scale: usize = res_base2k - res_acc_left;
                        ZNXARI::znx_extract_digit_addmul(res_acc_left, scale, res_slice, a_carry);
                    }

                    ZNXARI::znx_normalize_middle_step_inplace(res_base2k, 0, res_slice, res_carry);

                    // Previous step might not consume all bits of a_carry
                    // TODO: prove no overflow can happen here
                    ZNXARI::znx_add_inplace(res_carry, a_carry);

                    // We are done, so breaks out of the loop (yes we are at a[0], but
                    // this avoids possible over/under flows of tracking variables)
                    break 'outer;
                }

                // This block is not needed since exactly `res_base2k` bits are accumulated on `res_slice`.
                // else{
                //ZNXARI::znx_normalize_middle_step_inplace(res_base2k, 0, res_slice, res_carry);
                //}

                if res_limb == 0 {
                    break 'outer;
                }

                res_acc_left += res_base2k;
                res_limb -= 1;
            }

            // If a_norm is exhausted, breaks the inner loop.
            if a_take_left == 0 {
                ZNXARI::znx_add_inplace(a_carry, a_norm);
                break 'inner;
            }
        }
    }

    // This case will happen if offset is negative.
    if res_end != 0 {
        // If there are no overlapping limbs between `res` and `a`
        // (can happen if offset is negative), then we propagate the
        // carry of `a` on res. Note that the carry of `a` can be
        // greater than the precision of res.
        //
        // For example with offset = -8:
        //             a carry           a[0]     a[1]     a[2]     a[3]
        // a: [---------------------- ][x  x  x][x  x  x][x  x  x][x  x  x]
        // b: [x  x  x  x][x  x  x  x ]
        //        res[0]       res[1]
        //
        // If there are overlapping limbs between `res` and `a`,
        // we can use `res_carry`, which contains the carry of propagating
        // the shifted reconstruction of `a` in `res_base2k` along with
        // the carry of a[0].
        let carry_to_use = if a_start == a_end { a_carry } else { res_carry };

        for j in 0..res_end {
            if j == res_end - 1 {
                ZNXARI::znx_normalize_final_step_inplace(res_base2k, 0, res.at_mut(res_col, res_end - j - 1), carry_to_use);
            } else {
                ZNXARI::znx_normalize_middle_step_inplace(res_base2k, 0, res.at_mut(res_col, res_end - j - 1), carry_to_use);
            }
        }
    }
}

pub fn vec_znx_normalize_inplace<R: VecZnxToMut, ZNXARI>(base2k: usize, res: &mut R, res_col: usize, carry: &mut [i64])
where
    ZNXARI: ZnxNormalizeFirstStepInplace + ZnxNormalizeMiddleStepInplace + ZnxNormalizeFinalStepInplace,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert!(carry.len() >= res.n());
    }

    let res_size: usize = res.size();

    for j in (0..res_size).rev() {
        if j == res_size - 1 {
            ZNXARI::znx_normalize_first_step_inplace(base2k, 0, res.at_mut(res_col, j), carry);
        } else if j == 0 {
            ZNXARI::znx_normalize_final_step_inplace(base2k, 0, res.at_mut(res_col, j), carry);
        } else {
            ZNXARI::znx_normalize_middle_step_inplace(base2k, 0, res.at_mut(res_col, j), carry);
        }
    }
}

#[test]
fn test_vec_znx_normalize_cross_base2k() {
    let n: usize = 8;

    let mut carry: Vec<i64> = vec![0i64; vec_znx_normalize_tmp_bytes(n) / size_of::<i64>()];

    use crate::reference::znx::ZnxRef;
    use rug::ops::SubAssignRound;
    use rug::{Float, float::Round};

    let prec: usize = 128;

    for base2k_in in 1..=51 {
        for base2k_out in 1..=51 {
            for offset in -(prec as i64)..=(prec as i64) {
                let mut source: Source = Source::new([1u8; 32]);

                //-(base2k_in as i64)..(base2k_in as i64 + 1) {

                // for base2k_out in 1..2_usize {
                println!("offset: {offset} base2k_in: {base2k_in} base2k_out: {base2k_out}");

                let in_size: usize = prec.div_ceil(base2k_in);
                let in_prec: u32 = (in_size * base2k_in) as u32;

                // Ensures no loss of precision (mostly for testing purpose)
                let out_size: usize = (in_prec as usize).div_ceil(base2k_out);

                let out_prec: u32 = (out_size * base2k_out) as u32;
                let min_prec: u32 = (in_size * base2k_in).min(out_size * base2k_out) as u32;
                let mut want: VecZnx<Vec<u8>> = VecZnx::alloc(n, 1, in_size);
                want.fill_uniform(60, &mut source);

                let mut have: VecZnx<Vec<u8>> = VecZnx::alloc(n, 1, out_size);
                vec_znx_normalize_cross_base2k::<_, _, ZnxRef>(&mut have, base2k_out, offset, 0, &want, base2k_in, 0, &mut carry);

                let mut data_have: Vec<Float> = (0..n).map(|_| Float::with_val(out_prec + 60, 0)).collect();
                //let mut data_have_2: Vec<Float> = (0..n).map(|_| Float::with_val(out_prec + 60, 0)).collect();
                let mut data_want: Vec<Float> = (0..n).map(|_| Float::with_val(in_prec + 60, 0)).collect();

                have.decode_vec_float(base2k_out, 0, &mut data_have);

                want.decode_vec_float(base2k_in, 0, &mut data_want);

                //let mut have2: VecZnx<Vec<u8>> = VecZnx::alloc(n, 1, out_size);
                //if offset < 0{
                //    vec_znx_rsh_inplace::<_, ZnxRef>(base2k_in, offset.unsigned_abs() as usize, &mut want, 0, &mut carry);
                //    vec_znx_normalize::<_, _, ZnxRef>(0, base2k_out, &mut have2, 0, base2k_in, &want, 0, &mut carry);
                //}else{
                //vec_znx_normalize::<_, _, ZnxRef>(0, base2k_out, &mut have2, 0, base2k_in, &want, 0, &mut carry);
                //}

                //have2.decode_vec_float(base2k_out, 0, &mut data_have_2);

                println!("data_want: {:?}", data_want);
                println!("data_have: {:?}", data_have);
                //println!("data_have: {:?}", data_have_2);

                println!("have: {have}");
                println!("want: {want}");
                //println!("have2: {have2}");

                let scale: Float = Float::with_val(out_prec + 60, Float::u_pow_u(2, offset.unsigned_abs() as u32));

                if offset > 0 {
                    for x in &mut data_want {
                        *x *= &scale;
                        *x %= 1;
                    }
                } else if offset < 0 {
                    for x in &mut data_want {
                        *x /= &scale;
                        *x %= 1;
                    }
                } else {
                    for x in &mut data_want {
                        *x %= 1;
                    }
                }

                for x in &mut data_have {
                    if *x >= 0.5 {
                        *x -= 1;
                    } else if *x < -0.5 {
                        *x += 1;
                    }
                }

                for x in &mut data_want {
                    if *x >= 0.5 {
                        *x -= 1;
                    } else if *x < -0.5 {
                        *x += 1;
                    }
                }

                for i in 0..n {
                    println!("i:{i:02} {} {}", data_want[i], data_have[i]);

                    let mut err: Float = data_have[i].clone();
                    err.sub_assign_round(&data_want[i], Round::Nearest);
                    err = err.abs();

                    let err_log2: f64 = err.clone().max(&Float::with_val(prec as u32, 1e-60)).log2().to_f64();

                    assert!(err_log2 <= -(min_prec as f64) + 1.0, "{} {}", err_log2, -(min_prec as f64))
                }
            }
        }
    }
}

#[test]
fn test_vec_znx_normalize_inter_base2k() {
    let n: usize = 8;

    let mut carry: Vec<i64> = vec![0i64; vec_znx_normalize_tmp_bytes(n) / size_of::<i64>()];

    use crate::reference::znx::ZnxRef;
    use rug::ops::SubAssignRound;
    use rug::{Float, float::Round};

    let mut source: Source = Source::new([1u8; 32]);

    let prec: usize = 128;
    let offset_range: i64 = prec as i64;

    for base2k in 1..=51 {
        for offset in -offset_range..=offset_range {
            let size: usize = prec.div_ceil(base2k);
            let out_prec: u32 = (size * base2k) as u32;

            // Fills "want" with uniform values
            let mut want: VecZnx<Vec<u8>> = VecZnx::alloc(n, 1, size);
            want.fill_uniform(60, &mut source);

            // Fills "have" with the shifted normalization of "want"
            let mut have: VecZnx<Vec<u8>> = VecZnx::alloc(n, 1, size);

            vec_znx_normalize_inter_base2k::<_, _, ZnxRef>(base2k, &mut have, offset, 0, &want, 0, &mut carry);

            let mut data_have: Vec<Float> = (0..n).map(|_| Float::with_val(out_prec + 60, 0)).collect();
            let mut data_want: Vec<Float> = (0..n).map(|_| Float::with_val(out_prec + 60, 0)).collect();

            have.decode_vec_float(base2k, 0, &mut data_have);
            want.decode_vec_float(base2k, 0, &mut data_want);

            let scale: Float = Float::with_val(out_prec + 60, Float::u_pow_u(2, offset.unsigned_abs() as u32));

            if offset > 0 {
                for x in &mut data_want {
                    *x *= &scale;
                    *x %= 1;
                }
            } else if offset < 0 {
                for x in &mut data_want {
                    *x /= &scale;
                    *x %= 1;
                }
            } else {
                for x in &mut data_want {
                    *x %= 1;
                }
            }

            for x in &mut data_have {
                if *x >= 0.5 {
                    *x -= 1;
                } else if *x < -0.5 {
                    *x += 1;
                }
            }

            for x in &mut data_want {
                if *x >= 0.5 {
                    *x -= 1;
                } else if *x < -0.5 {
                    *x += 1;
                }
            }

            for i in 0..n {
                //println!("i:{i:02} {} {}", data_want[i], data_have[i]);

                let mut err: Float = data_have[i].clone();
                err.sub_assign_round(&data_want[i], Round::Nearest);
                err = err.abs();

                let err_log2: f64 = err.clone().max(&Float::with_val(prec as u32, 1e-60)).log2().to_f64();

                assert!(err_log2 <= -(out_prec as f64), "{} {}", err_log2, -(out_prec as f64))
            }
        }
    }
}
pub fn bench_vec_znx_normalize<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxNormalize<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_normalize::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxNormalize<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let base2k: usize = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        // Fill a with random i64
        a.fill_uniform(50, &mut source);
        res.fill_uniform(50, &mut source);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());

        move || {
            for i in 0..cols {
                module.vec_znx_normalize(base2k, &mut res, i, base2k, &a, i, scratch.borrow());
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2],));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_normalize_inplace<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxNormalizeInplace<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_normalize_inplace::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxNormalizeInplace<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let base2k: usize = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        // Fill a with random i64
        a.fill_uniform(50, &mut source);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());

        move || {
            for i in 0..cols {
                module.vec_znx_normalize_inplace(base2k, &mut a, i, scratch.borrow());
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2],));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
