use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use crate::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes},
    layouts::{Backend, FillUniform, Module, ScratchOwned, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::znx::{
        ZnxAddInplace, ZnxCopy, ZnxMulAddPowerOfTwo, ZnxMulPowerOfTwoInplace, ZnxNormalizeFinalStep,
        ZnxNormalizeFinalStepInplace, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeFirstStepInplace,
        ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepInplace, ZnxZero,
    },
    source::Source,
};

pub fn vec_znx_normalize_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_normalize<R, A, ZNXARI>(
    res_basek: usize,
    res: &mut R,
    res_col: usize,
    a_basek: usize,
    a: &A,
    a_col: usize,
    carry: &mut [i64],
) where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxZero
        + ZnxCopy
        + ZnxMulAddPowerOfTwo
        + ZnxAddInplace
        + ZnxMulPowerOfTwoInplace
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStepInplace
        + ZnxNormalizeFinalStepInplace
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFinalStep
        + ZnxNormalizeFirstStep,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert!(carry.len() >= res.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    if res_basek == a_basek {
        if a_size > res_size {
            for j in (res_size..a_size).rev() {
                if j == a_size - 1 {
                    ZNXARI::znx_normalize_first_step_carry_only(res_basek, 0, a.at(a_col, j), carry);
                } else {
                    ZNXARI::znx_normalize_middle_step_carry_only(res_basek, 0, a.at(a_col, j), carry);
                }
            }

            for j in (1..res_size).rev() {
                ZNXARI::znx_normalize_middle_step(res_basek, 0, res.at_mut(res_col, j), a.at(a_col, j), carry);
            }

            ZNXARI::znx_normalize_final_step(res_basek, 0, res.at_mut(res_col, 0), a.at(a_col, 0), carry);
        } else {
            for j in (0..a_size).rev() {
                if j == a_size - 1 {
                    ZNXARI::znx_normalize_first_step(res_basek, 0, res.at_mut(res_col, j), a.at(a_col, j), carry);
                } else if j == 0 {
                    ZNXARI::znx_normalize_final_step(res_basek, 0, res.at_mut(res_col, j), a.at(a_col, j), carry);
                } else {
                    ZNXARI::znx_normalize_middle_step(res_basek, 0, res.at_mut(res_col, j), a.at(a_col, j), carry);
                }
            }

            for j in a_size..res_size {
                ZNXARI::znx_zero(res.at_mut(res_col, j));
            }
        }
    } else {
        // Relevant limbs of res
        let res_min_size: usize = (a_size * a_basek).div_ceil(res_basek).min(res_size);

        // Relevant limbs of a
        let a_min_size: usize = (res_size * res_basek).div_ceil(a_basek).min(a_size);

        // Get carry for limbs of a that have higher precision than res
        for j in (a_min_size..a_size).rev() {
            if j == a_size - 1 {
                ZNXARI::znx_normalize_first_step_carry_only(res_basek, 0, a.at(a_col, j), carry);
            } else {
                ZNXARI::znx_normalize_middle_step_carry_only(res_basek, 0, a.at(a_col, j), carry);
            }
        }

        if a_min_size == a_size {
            ZNXARI::znx_copy(carry, a.at(a_col, a_min_size - 1));
        } else {
            ZNXARI::znx_add_inplace(carry, a.at(a_col, a_min_size - 1));
        }

        ZNXARI::znx_mul_power_of_two_inplace(
            (res_min_size * res_basek) as i64 - (a_min_size * a_basek) as i64,
            carry,
        );

        let mut j: usize = a_min_size - 1;
        for j_tild in (0..res_min_size).rev() {
            // Accumulate carry until precision is greater than receiver base2k
            while j * a_basek + res_basek > (j_tild + 1) * res_basek {
                ZNXARI::znx_muladd_power_of_two(
                    ((j_tild + 1) * res_basek - j * a_basek) as i64,
                    carry,
                    a.at(a_col, j - 1),
                );
                j -= 1;
            }

            // Flushes the carry on the receiver and updates the carry
            if j_tild != 0 {
                ZNXARI::znx_normalize_middle_step_inplace::<true>(res_basek, 0, res.at_mut(res_col, j_tild), carry);
            } else {
                ZNXARI::znx_normalize_final_step_inplace::<true>(res_basek, 0, res.at_mut(res_col, j_tild), carry);
            }
        }

        for j in res_min_size..res_size {
            ZNXARI::znx_zero(res.at_mut(res_col, j));
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
            ZNXARI::znx_normalize_final_step_inplace::<false>(base2k, 0, res.at_mut(res_col, j), carry);
        } else {
            ZNXARI::znx_normalize_middle_step_inplace::<false>(base2k, 0, res.at_mut(res_col, j), carry);
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
    let n: usize = 32;

    let mut carry = vec![0i64; n];

    use crate::reference::znx::ZnxRef;
    use rug::ops::SubAssignRound;
    use rug::{Float, float::Round};

    let mut source: Source = Source::new([1u8; 32]);

    let prec: usize = 100;

    for a_basek in 1..51 {
        for res_basek in 1..51 {
            let a_size: usize = prec.div_ceil(a_basek);
            let res_size: usize = prec.div_ceil(res_basek);

            let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, 1, a_size);

            a.fill_uniform(a_basek, &mut source);

            let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, 1, res_size);

            vec_znx_normalize::<_, _, ZnxRef>(res_basek, &mut res, 0, a_basek, &a, 0, &mut carry);

            let a_prec: u32 = (a_size * a_basek) as u32;
            let mut data_a: Vec<Float> = (0..n).map(|_| Float::with_val(a_prec as u32, 0)).collect();
            a.decode_vec_float(a_basek, 0, &mut data_a);

            let res_prec: u32 = (res_size * res_basek) as u32;
            let mut data_res: Vec<Float> = (0..n)
                .map(|_| Float::with_val(res_prec as u32, 0))
                .collect();
            res.decode_vec_float(res_basek, 0, &mut data_res);

            for i in 0..n {
                let mut err: Float = data_res[i].clone();
                err.sub_assign_round(&data_res[i], Round::Nearest);

                let err_log2: f64 = err
                    .clone()
                    .max(&Float::with_val(prec as u32, 1e-60))
                    .log2()
                    .to_f64();

                assert!(
                    err_log2 <= -(res_prec as f64) + 1.,
                    "{} {}",
                    err_log2,
                    -(res_prec as f64) + 1.
                )
            }
        }
    }
}
