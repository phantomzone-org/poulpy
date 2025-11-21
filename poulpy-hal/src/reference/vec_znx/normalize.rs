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

    let carry: &mut [i64] = &mut carry[..2 * n];

    if res_base2k == a_base2k {
        if a_size > res_size {
            for j in (res_size..a_size).rev() {
                if j == a_size - 1 {
                    ZNXARI::znx_normalize_first_step_carry_only(res_base2k, 0, a.at(a_col, j), carry);
                } else {
                    ZNXARI::znx_normalize_middle_step_carry_only(res_base2k, 0, a.at(a_col, j), carry);
                }
            }

            for j in (1..res_size).rev() {
                ZNXARI::znx_normalize_middle_step(res_base2k, 0, res.at_mut(res_col, j), a.at(a_col, j), carry);
            }

            ZNXARI::znx_normalize_final_step(res_base2k, 0, res.at_mut(res_col, 0), a.at(a_col, 0), carry);
        } else {
            for j in (0..a_size).rev() {
                if j == a_size - 1 {
                    ZNXARI::znx_normalize_first_step(res_base2k, 0, res.at_mut(res_col, j), a.at(a_col, j), carry);
                } else if j == 0 {
                    ZNXARI::znx_normalize_final_step(res_base2k, 0, res.at_mut(res_col, j), a.at(a_col, j), carry);
                } else {
                    ZNXARI::znx_normalize_middle_step(res_base2k, 0, res.at_mut(res_col, j), a.at(a_col, j), carry);
                }
            }

            for j in a_size..res_size {
                ZNXARI::znx_zero(res.at_mut(res_col, j));
            }
        }
    } else {
        let (a_norm, carry) = carry.split_at_mut(n);

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

        for j in (0..a_min_size).rev() {
            // Trackers: wow much of a_norm is left to
            // be flushed on res.
            let mut a_left: usize = a_base2k;

            // Normalizes the j-th limb of a and store the results into a_norm.
            // This step is required to avoid overflow in the next step,
            // which assumes that |a| is bounded by 2^{a_base2k -1}.
            if j != 0 {
                ZNXARI::znx_normalize_middle_step(a_base2k, 0, a_norm, a.at(a_col, j), carry);
            } else {
                ZNXARI::znx_normalize_final_step(a_base2k, 0, a_norm, a.at(a_col, j), carry);
            }

            // In the first iteration we need to match the precision of the input/output.
            // If a_min_size * a_base2k > res_min_size * res_base2k
            // then divround a_norm by the difference of precision and
            // acts like if a_norm has already been partially consummed.
            // Else acts like if res has been already populated
            // by the difference.
            if j == a_min_size - 1 {
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
                let lsh: usize = res_base2k - res_left;

                // Extract the bits to flush on the output and updates
                // a_norm accordingly.
                ZNXARI::znx_extract_digit_addmul(a_take, lsh, res_slice, a_norm);

                // Updates the trackers
                a_left -= a_take;
                res_left -= a_take;

                // If the current limb of res is full,
                // then normalizes this limb and adds
                // the carry on a_norm.
                if res_left == 0 {
                    // Updates tracker
                    res_left += res_base2k;

                    // Normalizes res and propagates the carry on a.
                    ZNXARI::znx_normalize_digit(res_base2k, res_slice, a_norm);

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
fn test_vec_znx_normalize_conv() {
    let n: usize = 8;

    let mut carry: Vec<i64> = vec![0i64; 2 * n];

    use crate::reference::znx::ZnxRef;
    use rug::ops::SubAssignRound;
    use rug::{Float, float::Round};

    let mut source: Source = Source::new([1u8; 32]);

    let prec: usize = 128;

    let mut data: Vec<i128> = vec![0i128; n];

    data.iter_mut().for_each(|x| *x = source.next_i128());

    for start_base2k in 1..50 {
        for end_base2k in 1..50 {
            let end_size: usize = prec.div_ceil(end_base2k);

            let mut want: VecZnx<Vec<u8>> = VecZnx::alloc(n, 1, end_size);
            want.encode_vec_i128(end_base2k, 0, prec, &data);
            vec_znx_normalize_inplace::<_, ZnxRef>(end_base2k, &mut want, 0, &mut carry);

            // Creates a temporary poly where encoding is in start_base2k
            let mut tmp: VecZnx<Vec<u8>> = VecZnx::alloc(n, 1, prec.div_ceil(start_base2k));
            tmp.encode_vec_i128(start_base2k, 0, prec, &data);

            vec_znx_normalize_inplace::<_, ZnxRef>(start_base2k, &mut tmp, 0, &mut carry);

            let mut data_tmp: Vec<Float> = (0..n).map(|_| Float::with_val(prec as u32, 0)).collect();
            tmp.decode_vec_float(start_base2k, 0, &mut data_tmp);

            let mut have: VecZnx<Vec<u8>> = VecZnx::alloc(n, 1, end_size);
            vec_znx_normalize::<_, _, ZnxRef>(end_base2k, &mut have, 0, start_base2k, &tmp, 0, &mut carry);

            let out_prec: u32 = (end_size * end_base2k) as u32;

            let mut data_want: Vec<Float> = (0..n).map(|_| Float::with_val(out_prec, 0)).collect();
            let mut data_res: Vec<Float> = (0..n).map(|_| Float::with_val(out_prec, 0)).collect();

            have.decode_vec_float(end_base2k, 0, &mut data_want);
            want.decode_vec_float(end_base2k, 0, &mut data_res);

            for i in 0..n {
                let mut err: Float = data_want[i].clone();
                err.sub_assign_round(&data_res[i], Round::Nearest);
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
