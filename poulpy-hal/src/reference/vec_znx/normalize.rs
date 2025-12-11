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
        assert!(carry.len() >= vec_znx_normalize_tmp_bytes(res.n()) / size_of::<i64>());
        assert_eq!(res.n(), a.n());
    }

    let n: usize = res.n();
    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let (a_norm, carry) = carry.split_at_mut(n);
    let (res_carry, a_carry) = carry[..2 * n].split_at_mut(n);
    ZNXARI::znx_zero(res_carry);

    // Relevant limbs of res
    let res_min_size: usize = (a_size * a_base2k).div_ceil(res_base2k).min(res_size);

    // Relevant limbs of a
    let a_min_size: usize = (res_size * res_base2k).div_ceil(a_base2k).min(a_size);

    // Maximum relevant precision of a
    let a_prec: usize = a_min_size * a_base2k;

    // Maximum relevant precision of res
    let res_prec: usize = res_min_size * res_base2k;

    let offset_abs: usize = offset.unsigned_abs() as usize;
    let mut steps: usize = offset_abs / res_base2k;

    let (lsh, res_end_bit, res_start_bit, a_start_bit) = if offset < 0 {
        if !offset_abs.is_multiple_of(res_base2k) {
            steps += 1;
        }
        (
            (res_base2k - (offset_abs % res_base2k)) % res_base2k,
            res_prec.min(steps * res_base2k),                        // res_end
            res_prec.min(a_prec + steps * res_base2k),               // res_start
            a_prec.min(res_prec.saturating_sub(steps * res_base2k)), // a_start
        )
    } else {
        (
            offset_abs % res_base2k,
            0,                                                       // res_end
            res_prec.min(a_prec.saturating_sub(steps * res_base2k)), // res_start
            a_prec.min(res_prec + steps * res_base2k),               // a_start
        )
    };

    let a_start: usize = a_start_bit.div_ceil(a_base2k);
    let res_start: usize = res_start_bit.div_ceil(res_base2k);
    let res_end: usize = res_end_bit.div_ceil(res_base2k);

    let a_out_range: usize = a_size.saturating_sub(a_start);

    if res_base2k == a_base2k {
        // Computes the carry over the discarded limbs of a
        for j in 0..a_out_range {
            if j == 0 {
                ZNXARI::znx_normalize_first_step_carry_only(a_base2k, lsh, a.at(a_col, a_size - j - 1), a_carry);
            } else {
                ZNXARI::znx_normalize_middle_step_carry_only(a_base2k, lsh, a.at(a_col, a_size - j - 1), a_carry);
            }
        }

        if a_out_range == 0 {
            ZNXARI::znx_zero(a_carry);
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
                a_carry,
            );
        }

        // Propagates the carry over the last limbs of res
        for j in 0..res_end {
            if j == res_end - 1 {
                ZNXARI::znx_normalize_final_step_inplace(res_base2k, lsh, res.at_mut(res_col, res_end - j - 1), a_carry);
            } else {
                ZNXARI::znx_normalize_middle_step_inplace(res_base2k, lsh, res.at_mut(res_col, res_end - j - 1), a_carry);
            }
        }
    } else {
        for j in 0..a_out_range {
            if j == 0 {
                ZNXARI::znx_normalize_first_step_carry_only(a_base2k, 0, a.at(a_col, a_size - j - 1), a_carry);
            } else {
                ZNXARI::znx_normalize_middle_step_carry_only(a_base2k, 0, a.at(a_col, a_size - j - 1), a_carry);
            }
        }

        if a_out_range == 0 {
            ZNXARI::znx_zero(a_carry);
        }

        // Maximum relevant precision of a
        let mut a_bits: usize = a_min_size * a_base2k;

        // Maximum relevant precision of res
        let res_bits: usize = res_min_size * res_base2k;

        for j in 0..res_size {
            ZNXARI::znx_zero(res.at_mut(res_col, j));
        }

        let mut res_start_bit: usize = res_bits.saturating_sub(steps * res_base2k);

        // Res limb index
        if res_start_bit == 0 {
            return;
        }

        //println!("res_start_bit: {res_start_bit} lsh: {lsh}");

        let mut res_acc_left: usize = res_base2k;

        'outer: loop {
            if a_bits == 0 {
                break;
            }

            // Current res & a limbs
            let a_limb: usize = a_bits.div_ceil(a_base2k) - 1;
            let a_slice: &[i64] = a.at(a_col, a_limb);

            //println!("a_prec: {a_prec} a_limb: {a_limb}");

            // Trackers: wow much of a_norm is left to
            // be flushed on res.
            let mut a_take_left: usize = a_base2k;

            // Normalizes the j-th limb of a and store the results into a_norm.
            // This step is required to avoid overflow in the next step,
            // which assumes that |a| is bounded by 2^{a_base2k -1}.
            if a_limb != 0 {
                ZNXARI::znx_normalize_middle_step(a_base2k, 0, a_norm, a_slice, a_carry);
            } else {
                ZNXARI::znx_normalize_final_step(a_base2k, 0, a_norm, a_slice, a_carry);
            }

            // In the first iteration we need to match the precision of the input/output.
            // If a_min_size * a_base2k > res_min_size * res_base2k
            // then divround a_norm by the difference of precision and
            // acts like if a_norm has already been partially consummed.
            // Else acts like if res has been already populated
            // by the difference.
            if a_limb == a_min_size - 1 {
                if a_bits > res_bits {
                    let take: usize = a_bits - res_bits;
                    ZNXARI::znx_mul_power_of_two_inplace(-(take as i64), a_norm);
                    a_take_left -= take;
                    a_bits -= take;
                } else if res_bits > a_bits {
                    res_acc_left -= res_bits - a_bits;
                }
            }

            // Accumulates until a & Flushes
            loop {
                let res_limb: usize = res_start_bit.div_ceil(res_base2k) - 1;
                let res_slice: &mut [i64] = res.at_mut(res_col, res_limb);
                //println!("res_prec: {res_start_bit} res_limb: {res_limb}");

                // We can take at most a_base2k bits
                // but not more than what is left on on a_norm
                let a_take: usize = a_base2k.min(a_take_left).min(res_acc_left);

                if a_take != 0 {
                    // Extract `a_take` bits from a_norm and accumulates them on `res_slice`.
                    let scale: usize = res_base2k - res_acc_left;

                    ZNXARI::znx_extract_digit_addmul(a_take, scale, res_slice, a_norm);

                    a_take_left -= a_take;
                    res_acc_left -= a_take;
                    a_bits -= a_take;
                }

                // If at least `res_base2k` bits have been accumulated flushes them onto res
                // We can accomodate for more than `res_base2k` bits.
                if res_acc_left == 0 || a_bits == 0 {
                    //println!("res_slice: {:?} a_norm: {:?}", res_slice, a_norm);

                    ZNXARI::znx_normalize_middle_step_inplace(res_base2k, lsh, res_slice, res_carry);

                    res_acc_left += res_base2k;
                    res_start_bit = res_start_bit.saturating_sub(res_base2k);

                    if res_start_bit == 0 {
                        break 'outer;
                    }
                }

                // If a_norm is exhausted, breaks the loop.
                if a_take_left == 0 {
                    ZNXARI::znx_add_inplace(a_carry, a_norm);
                    break;
                }
            }

            //println!();
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

#[test]
fn test_vec_znx_normalize_base2k_not_equal_base2k_out() {
    let n: usize = 8;

    let mut carry: Vec<i64> = vec![0i64; vec_znx_normalize_tmp_bytes(n) / size_of::<i64>()];

    use crate::reference::znx::ZnxRef;
    use rug::ops::SubAssignRound;
    use rug::{Float, float::Round};

    let prec: usize = 32;

    for base2k_in in (1..52_usize).step_by(2) {
        for base2k_out in (1..52_usize).step_by(2) {
            for offset in 0..(prec as i64 + 1) {
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
                vec_znx_normalize::<_, _, ZnxRef>(offset, base2k_out, &mut have, 0, base2k_in, &want, 0, &mut carry);

                let mut data_have: Vec<Float> = (0..n).map(|_| Float::with_val(out_prec + 60, 0)).collect();
                let mut data_want: Vec<Float> = (0..n).map(|_| Float::with_val(in_prec + 60, 0)).collect();

                have.decode_vec_float(base2k_out, 0, &mut data_have);
                want.decode_vec_float(base2k_in, 0, &mut data_want);

                println!("data_want: {:?}", data_want);
                println!("data_have: {:?}", data_have);

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

                    assert!(err_log2 <= -(min_prec as f64), "{} {}", err_log2, -(min_prec as f64))
                }
            }
        }
    }
}

#[test]
fn test_vec_znx_normalize_base2k_in_equal_base2k_out() {
    let n: usize = 8;

    let mut carry: Vec<i64> = vec![0i64; vec_znx_normalize_tmp_bytes(n) / size_of::<i64>()];

    use crate::reference::znx::ZnxRef;
    use rug::ops::SubAssignRound;
    use rug::{Float, float::Round};

    let mut source: Source = Source::new([1u8; 32]);

    let prec: usize = 128;
    let offset_range: i64 = prec as i64;

    for base2k in 1..=55_usize {
        for offset in -offset_range..=offset_range {
            //println!("offset: {offset} base2k: {base2k}");

            let size: usize = prec.div_ceil(base2k);
            let out_prec: u32 = (size * base2k) as u32;

            // Fills "want" with uniform values
            let mut want: VecZnx<Vec<u8>> = VecZnx::alloc(n, 1, size);
            want.fill_uniform(60, &mut source);

            // Fills "have" with the shifted normalization of "want"
            let mut have: VecZnx<Vec<u8>> = VecZnx::alloc(n, 1, size);
            vec_znx_normalize::<_, _, ZnxRef>(offset, base2k, &mut have, 0, base2k, &want, 0, &mut carry);

            let mut data_have: Vec<Float> = (0..n).map(|_| Float::with_val(out_prec + 60, 0)).collect();
            let mut data_want: Vec<Float> = (0..n).map(|_| Float::with_val(out_prec + 60, 0)).collect();

            have.decode_vec_float(base2k, 0, &mut data_have);
            want.decode_vec_float(base2k, 0, &mut data_want);

            //println!("data_want: {:?}", &data_want);
            //println!("data_have: {:?}", &data_have);

            let scale: Float = Float::with_val(out_prec + 60, Float::u_pow_u(2, offset.unsigned_abs() as u32));

            //println!("scale: {}", scale);

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
