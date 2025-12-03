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
    2 * n * size_of::<i64>()
}

pub fn vec_znx_normalize<R, A, ZNXARI>(
    offset: i64,
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
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert!(carry.len() >= 2 * res.n());
        assert_eq!(res.n(), a.n());
    }

    let n: usize = res.n();
    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let (a_norm, carry) = carry[..2 * n].split_at_mut(n);

    let mut res_carry: Vec<i64> = vec![0i64; n];

    let offset_abs: usize = offset.abs() as usize;

    let mut steps: usize = offset_abs / res_base2k;

    if res_base2k == a_base2k {
        let (lsh, res_end, res_start, a_start) = if offset < 0 {
            let lsh = if !offset_abs.is_multiple_of(res_base2k) {
                steps += 1;
                res_base2k - (offset_abs % res_base2k)
            } else {
                0
            };

            (
                lsh,
                res_size.min(steps),                        // res_end
                res_size.min(a_size + steps),               // res_start
                a_size.min(res_size.saturating_sub(steps)), // a_start
            )
        } else {
            (
                offset_abs % res_base2k,
                0,                                          // res_end
                res_size.min(a_size.saturating_sub(steps)), // res_start
                a_size.min(res_size + steps),               // a_start
            )
        };

        let a_out_range: usize = a_size.saturating_sub(a_start);

        // Computes the carry over the discarded limbs of a
        for j in 0..a_out_range {
            if j == 0 {
                ZNXARI::znx_normalize_first_step_carry_only(res_base2k, lsh, a.at(a_col, a_size - j - 1), carry);
            } else {
                ZNXARI::znx_normalize_middle_step_carry_only(res_base2k, lsh, a.at(a_col, a_size - j - 1), carry);
            }
        }

        if a_out_range == 0 {
            ZNXARI::znx_zero(carry);
        }

        for j in res_start..res_size {
            ZNXARI::znx_zero(res.at_mut(res_col, j));
        }

        let mid_range: usize = res_start.saturating_sub(res_end);

        // Regular normalization over the overlapping limbs of res and a.
        for j in 0..mid_range {
            ZNXARI::znx_normalize_middle_step(
                res_base2k,
                lsh,
                res.at_mut(res_col, res_start - j - 1),
                a.at(a_col, a_start - j - 1),
                carry,
            );
        }

        // Propagates the carry over the last limbs of res
        for j in 0..res_end {
            if j == res_end - 1 {
                ZNXARI::znx_normalize_final_step_inplace(res_base2k, lsh, res.at_mut(res_col, res_end - j - 1), carry);
            } else {
                ZNXARI::znx_normalize_middle_step_inplace(res_base2k, lsh, res.at_mut(res_col, res_end - j - 1), carry);
            }
        }
    } else {
        // Relevant limbs of res
        let res_min_size: usize = (a_size * a_base2k).div_ceil(res_base2k).min(res_size);

        // Relevant limbs of a
        let a_min_size: usize = (res_size * res_base2k).div_ceil(a_base2k).min(a_size);

        // Get carry for limbs of a that have higher precision than res
        for j in (a_min_size..a_size).rev() {
            if j == a_size - 1 {
                ZNXARI::znx_normalize_first_step_carry_only(a_base2k, 0, a.at(a_col, j), carry);
            } else {
                ZNXARI::znx_normalize_middle_step_carry_only(a_base2k, 0, a.at(a_col, j), carry);
            }
        }

        if a_min_size == a_size {
            ZNXARI::znx_zero(carry);
        }

        // Maximum relevant precision of a
        let a_prec: usize = a_min_size * a_base2k;

        // Maximum relevant precision of res
        let res_prec: usize = res_min_size * res_base2k;

        // Res limb index
        let mut res_idx: usize = res_min_size - 1;

        // Trackers: wow much of res is left to be populated
        // for the current limb.
        let mut res_left: usize = res_base2k;

        for j in 0..res_size {
            ZNXARI::znx_zero(res.at_mut(res_col, j));
        }

        for j in 0..a_min_size {
            // Trackers: wow much of a_norm is left to
            // be flushed on res.
            let mut a_left: usize = a_base2k;

            // Normalizes the j-th limb of a and store the results into a_norm.
            // This step is required to avoid overflow in the next step,
            // which assumes that |a| is bounded by 2^{a_base2k -1}.
            if j != a_min_size - 1 {
                ZNXARI::znx_normalize_middle_step(a_base2k, 0, a_norm, a.at(a_col, a_min_size - j - 1), carry);
            } else {
                ZNXARI::znx_normalize_final_step(a_base2k, 0, a_norm, a.at(a_col, a_min_size - j - 1), carry);
            }

            // In the first iteration we need to match the precision of the input/output.
            // If a_min_size * a_base2k > res_min_size * res_base2k
            // then divround a_norm by the difference of precision and
            // acts like if a_norm has already been partially consummed.
            // Else acts like if res has been already populated
            // by the difference.
            if j == 0 {
                if a_prec > res_prec {
                    ZNXARI::znx_mul_power_of_two_inplace(res_prec as i64 - a_prec as i64, a_norm);
                    a_left -= a_prec - res_prec;
                } else if res_prec > a_prec {
                    res_left -= res_prec - a_prec;
                }
            }

            // Flushes a into res
            loop {
                // Selects the maximum amount of a that can be flushed
                let a_take: usize = a_base2k.min(a_left).min(res_left);

                // Output limb
                let res_slice: &mut [i64] = res.at_mut(res_col, res_idx);

                // Scaling of the value to flush
                let scale: usize = res_base2k - res_left;

                // Extract the bits to flush on the output and updates
                // a_norm accordingly.
                ZNXARI::znx_extract_digit_addmul(a_take, scale, res_slice, a_norm);

                // Updates the trackers
                a_left -= a_take;
                res_left -= a_take;

                // If the current limb of res is full,
                // then normalizes this limb and adds
                // the carry on a_norm.
                if res_left == 0 {
                    // Updates tracker
                    res_left += res_base2k;

                    // Normalizes res and propagates it's own carry.
                    ZNXARI::znx_normalize_middle_step_inplace(res_base2k, 0, res_slice, &mut res_carry);

                    // If we reached the last limb of res breaks,
                    // but we might rerun the above loop if the
                    // base2k of a is much smaller than the base2k
                    // of res.
                    if res_idx == 0 {
                        ZNXARI::znx_add_inplace(carry, a_norm);
                        break;
                    }

                    // Else updates the limb index of res.
                    res_idx -= 1
                }

                // If a_norm is exhausted, breaks the loop.
                if a_left == 0 {
                    ZNXARI::znx_add_inplace(carry, a_norm);
                    break;
                }
            }
        }

        /*
        ZNXARI::znx_add_inplace(carry, &res_carry);

        // Propagates the carry over the last limbs of res
        for j in 0..res_end {
            if j == res_end - 1 {
                ZNXARI::znx_normalize_final_step_inplace(res_base2k, 0, res.at_mut(res_col, res_end - j - 1), carry);
            } else {
                ZNXARI::znx_normalize_middle_step_inplace(res_base2k, 0, res.at_mut(res_col, res_end - j - 1), carry);
            }
        }
         */
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

#[test]
fn test_vec_znx_normalize_base2k_in_equal_base2k_out() {
    let n: usize = 8;

    let mut carry: Vec<i64> = vec![0i64; 2 * n];

    use crate::reference::znx::ZnxRef;
    use rug::ops::SubAssignRound;
    use rug::{Float, float::Round};

    use crate::reference::vec_znx::{vec_znx_lsh_inplace, vec_znx_rsh_inplace};

    let mut source: Source = Source::new([1u8; 32]);

    let prec: usize = 128;

    for base2k in 1..50_usize {
        for offset in -(base2k as i64)..(base2k as i64 + 1) {
            println!("offset: {offset} base2k: {base2k}");

            let size: usize = prec.div_ceil(base2k);
            let out_prec: u32 = (size * base2k) as u32;

            // Fills "want" with uniform values
            let mut want: VecZnx<Vec<u8>> = VecZnx::alloc(n, 1, size);
            want.fill_uniform(60, &mut source);

            // Fills "have" with the shifted normalization of "want"
            let mut have: VecZnx<Vec<u8>> = VecZnx::alloc(n, 1, size);
            vec_znx_normalize::<_, _, ZnxRef>(offset, base2k, &mut have, 0, base2k, &want, 0, &mut carry);

            // Shifts THEN ONLY normalizes "want"
            if offset > 0 {
                vec_znx_lsh_inplace::<_, ZnxRef>(base2k, offset as usize, &mut want, 0, &mut carry);
            } else if offset < 0 {
                vec_znx_rsh_inplace::<_, ZnxRef>(base2k, offset.abs() as usize, &mut want, 0, &mut carry);
            } else {
                vec_znx_normalize_inplace::<_, ZnxRef>(base2k, &mut want, 0, &mut carry);
            }

            let mut data_have: Vec<Float> = (0..n).map(|_| Float::with_val(out_prec, 0)).collect();
            let mut data_want: Vec<Float> = (0..n).map(|_| Float::with_val(out_prec, 0)).collect();

            have.decode_vec_float(base2k, 0, &mut data_have);
            want.decode_vec_float(base2k, 0, &mut data_want);

            println!("have: {have}");
            println!("want: {want}");

            for i in 0..n {
                println!("i:{i:02} {} {}", data_want[i], data_have[i]);

                let mut err: Float = data_have[i].clone();
                err.sub_assign_round(&data_want[i], Round::Nearest);
                err = err.abs();

                let err_log2: f64 = err
                    .clone()
                    .max(&Float::with_val(prec as u32, 1e-60))
                    .log2()
                    .to_f64();

                assert!(
                    err_log2 <= -(out_prec as f64) + 1.,
                    "{} {}",
                    err_log2,
                    -(out_prec as f64) + 1.
                )
            }
        }
    }
}

#[test]
fn test_vec_znx_normalize_base2k_not_equal_base2k_out() {
    let n: usize = 8;

    let mut carry: Vec<i64> = vec![0i64; 2 * n];

    use crate::reference::znx::ZnxRef;
    use rug::ops::SubAssignRound;
    use rug::{Float, float::Round};

    use crate::reference::vec_znx::{vec_znx_lsh_inplace, vec_znx_rsh_inplace};

    let mut source: Source = Source::new([1u8; 32]);

    let prec: usize = 128;

    for base2k_in in 1..50_usize {
        for base2k_out in 1..50_usize {
            for offset in 0..1 {//-(base2k_in as i64)..(base2k_in as i64 + 1) {
            
                // for base2k_out in 1..2_usize {
                println!("offset: {offset} base2k_in: {base2k_in} base2k_out: {base2k_out}");

                let in_size: usize = prec.div_ceil(base2k_in);
                let out_size: usize = prec.div_ceil(base2k_out);
                let out_prec: u32 = (out_size * base2k_out) as u32;

                let mut want: VecZnx<Vec<u8>> = VecZnx::alloc(n, 1, in_size);
                want.fill_uniform(base2k_in, &mut source);

                let mut have: VecZnx<Vec<u8>> = VecZnx::alloc(n, 1, out_size);
                let mut have2: VecZnx<Vec<u8>> = VecZnx::alloc(n, 1, out_size);
                vec_znx_normalize::<_, _, ZnxRef>(
                    offset, base2k_out, &mut have, 0, base2k_in, &want, 0, &mut carry,
                );

                vec_znx_normalize::<_, _, ZnxRef>(
                    0, base2k_out, &mut have2, 0, base2k_in, &want, 0, &mut carry,
                );

                vec_znx_normalize_inplace::<_, ZnxRef>(base2k_in, &mut want, 0, &mut carry);

                if offset > 0 {
                    vec_znx_lsh_inplace::<_, ZnxRef>(base2k_in, offset as usize, &mut want, 0, &mut carry);
                    vec_znx_lsh_inplace::<_, ZnxRef>(base2k_out, offset as usize, &mut have2, 0, &mut carry);
                } else if offset < 0 {
                    vec_znx_rsh_inplace::<_, ZnxRef>(base2k_in, offset.abs() as usize, &mut want, 0, &mut carry);
                    vec_znx_rsh_inplace::<_, ZnxRef>(base2k_out, offset.abs() as usize, &mut have2, 0, &mut carry);
                }

                let mut data_have: Vec<Float> = (0..n).map(|_| Float::with_val(out_prec, 0)).collect();
                let mut data_have2: Vec<Float> = (0..n).map(|_| Float::with_val(out_prec, 0)).collect();

                let mut data_want: Vec<Float> = (0..n).map(|_| Float::with_val(out_prec, 0)).collect();

                have.decode_vec_float(base2k_out, 0, &mut data_have);
                want.decode_vec_float(base2k_in, 0, &mut data_want);
                have2.decode_vec_float(base2k_out, 0, &mut data_have2);

                //println!("have2: {have2}");
                //println!("have: {have}");
                //println!("want: {want}");

                for i in 0..n {
                    if data_want[i] >= 0.5 {
                        data_want[i] -= 1;
                    }

                    if data_want[i] <= -0.5 {
                        data_want[i] += 1;
                    }

                    if data_have[i] >= 0.5 {
                        data_have[i] -= 1;
                    }

                    if data_have[i] <= -0.5 {
                        data_have[i] += 1;
                    }

                    if data_have2[i] >= 0.5 {
                        data_have2[i] -= 1;
                    }

                    if data_have2[i] <= -0.5 {
                        data_have2[i] += 1;
                    }

                    println!(
                        "i:{i:02} want: {} have_1: {} have_2: {}",
                        data_want[i], data_have[i], data_have2[i]
                    );

                    let mut err: Float = data_have[i].clone();
                    err.sub_assign_round(&data_want[i], Round::Nearest);
                    err = err.abs();

                    let err_log2: f64 = err
                        .clone()
                        .max(&Float::with_val(prec as u32, 1e-60))
                        .log2()
                        .to_f64();

                    assert!(
                        err_log2 <= -(out_prec as f64) + 1.,
                        "{} {}",
                        err_log2,
                        -(out_prec as f64) + 1.
                    )
                }
            }
        }
    }
}
