use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use crate::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxLsh, VecZnxLshInplace, VecZnxRsh, VecZnxRshInplace},
    layouts::{Backend, FillUniform, Module, ScratchOwned, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::znx::{
        ZnxCopy, ZnxNormalizeFinalStep, ZnxNormalizeFinalStepInplace, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepCarryOnly,
        ZnxNormalizeFirstStepInplace, ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepInplace,
        ZnxZero,
    },
    source::Source,
};

pub fn vec_znx_lsh_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_lsh_inplace<R, ZNXARI>(base2k: usize, k: usize, res: &mut R, res_col: usize, carry: &mut [i64])
where
    R: VecZnxToMut,
    ZNXARI: ZnxZero
        + ZnxCopy
        + ZnxNormalizeFirstStepInplace
        + ZnxNormalizeMiddleStepInplace
        + ZnxNormalizeFirstStepInplace
        + ZnxNormalizeFinalStepInplace,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    let n: usize = res.n();
    let cols: usize = res.cols();
    let size: usize = res.size();
    let steps: usize = k / base2k;
    let k_rem: usize = k % base2k;

    if steps >= size {
        for j in 0..size {
            ZNXARI::znx_zero(res.at_mut(res_col, j));
        }
        return;
    }

    // Inplace shift of limbs by a k/base2k
    if steps > 0 {
        let start: usize = n * res_col;
        let end: usize = start + n;
        let slice_size: usize = n * cols;
        let res_raw: &mut [i64] = res.raw_mut();

        (0..size - steps).for_each(|j| {
            let (lhs, rhs) = res_raw.split_at_mut(slice_size * (j + steps));
            ZNXARI::znx_copy(&mut lhs[start + j * slice_size..end + j * slice_size], &rhs[start..end]);
        });

        for j in size - steps..size {
            ZNXARI::znx_zero(res.at_mut(res_col, j));
        }
    }

    for j in (0..size - steps).rev() {
        if j == size - steps - 1 {
            ZNXARI::znx_normalize_first_step_inplace(base2k, k_rem, res.at_mut(res_col, j), carry);
        } else if j == 0 {
            ZNXARI::znx_normalize_final_step_inplace(base2k, k_rem, res.at_mut(res_col, j), carry);
        } else {
            ZNXARI::znx_normalize_middle_step_inplace(base2k, k_rem, res.at_mut(res_col, j), carry);
        }
    }
}

pub fn vec_znx_lsh<R, A, ZNXARI>(base2k: usize, k: usize, res: &mut R, res_col: usize, a: &A, a_col: usize, carry: &mut [i64])
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxZero + ZnxNormalizeFirstStep + ZnxNormalizeMiddleStep + ZnxNormalizeFirstStep + ZnxCopy + ZnxNormalizeFinalStep,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    let res_size: usize = res.size();
    let a_size = a.size();
    let steps: usize = k / base2k;
    let k_rem: usize = k % base2k;

    if steps >= res_size.min(a_size) {
        for j in 0..res_size {
            ZNXARI::znx_zero(res.at_mut(res_col, j));
        }
        return;
    }

    let min_size: usize = a_size.min(res_size) - steps;

    // Simply a left shifted normalization of limbs
    // by k/base2k and intra-limb by base2k - k%base2k
    for j in (0..min_size).rev() {
        if j == min_size - 1 {
            ZNXARI::znx_normalize_first_step(base2k, k_rem, res.at_mut(res_col, j), a.at(a_col, j + steps), carry);
        } else if j == 0 {
            ZNXARI::znx_normalize_final_step(base2k, k_rem, res.at_mut(res_col, j), a.at(a_col, j + steps), carry);
        } else {
            ZNXARI::znx_normalize_middle_step(base2k, k_rem, res.at_mut(res_col, j), a.at(a_col, j + steps), carry);
        }
    }

    // Zeroes bottom
    for j in min_size..res_size {
        ZNXARI::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_rsh_tmp_bytes(n: usize) -> usize {
    2 * n * size_of::<i64>()
}

pub fn vec_znx_rsh_inplace<R, ZNXARI>(base2k: usize, k: usize, res: &mut R, res_col: usize, tmp: &mut [i64])
where
    R: VecZnxToMut,
    ZNXARI: ZnxZero
        + ZnxCopy
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeMiddleStepInplace
        + ZnxNormalizeFirstStepInplace
        + ZnxNormalizeFinalStepInplace,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let n: usize = res.n();

    let size: usize = res.size();

    let mut steps: usize = k / base2k;
    let k_rem: usize = k % base2k;

    if !k.is_multiple_of(base2k) {
        // We rsh by an additional base2k and then lsh by base2k-k
        // Allows to re-use efficient normalization code, avoids
        // avoids overflows & produce output that is normalized
        steps += 1;
    }

    let (carry, tmp) = tmp[..2 * n].split_at_mut(n);

    let lsh: usize = (base2k - k_rem) % base2k;

    // All limbs of a that would fall outside of the limbs of res are discarded,
    // but the carry still need to be computed.
    for j in 0..steps {
        if j == 0 {
            ZNXARI::znx_normalize_first_step_carry_only(base2k, lsh, res.at(res_col, size - j - 1), carry);
        } else {
            ZNXARI::znx_normalize_middle_step_carry_only(base2k, lsh, res.at(res_col, size - j - 1), carry);
        }
    }

    // Continues with shifted normalization
    for j in 0..size - steps {
        ZNXARI::znx_copy(tmp, res.at(res_col, size - steps - j - 1));
        ZNXARI::znx_normalize_middle_step_inplace(base2k, lsh, tmp, carry);
        ZNXARI::znx_copy(res.at_mut(res_col, size - j - 1), tmp);
    }

    // Propagates carry on the rest of the limbs of res
    for j in 0..steps {
        ZNXARI::znx_zero(res.at_mut(res_col, j));
        if j == 0 {
            ZNXARI::znx_normalize_final_step_inplace(base2k, lsh, res.at_mut(res_col, steps - j - 1), carry);
        } else {
            ZNXARI::znx_normalize_middle_step_inplace(base2k, lsh, res.at_mut(res_col, steps - j - 1), carry);
        }
    }
}

pub fn vec_znx_rsh<R, A, ZNXARI>(base2k: usize, k: usize, res: &mut R, res_col: usize, a: &A, a_col: usize, carry: &mut [i64])
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxZero
        + ZnxCopy
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeFirstStep
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeMiddleStepInplace
        + ZnxNormalizeFirstStepInplace
        + ZnxNormalizeFinalStepInplace,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let mut steps: usize = k / base2k;
    let k_rem: usize = k % base2k;

    if !k.is_multiple_of(base2k) {
        // We rsh by an additional base2k and then lsh by base2k-k
        // Allows to re-use efficient normalization code, avoids
        // avoids overflows & produce output that is normalized
        steps += 1;
    }

    let lsh: usize = (base2k - k_rem) % base2k; // 0 if k | base2k
    let res_end: usize = res_size.min(steps);
    let res_start: usize = res_size.min(a_size + steps);
    let a_start: usize = a_size.min(res_size.saturating_sub(steps));

    // All limbs of a that are moved outside of the limbs of res are discarded,
    // but the carry still need to be computed.
    let a_out_range: usize = a_size.saturating_sub(a_start);

    for j in 0..a_out_range {
        if j == 0 {
            ZNXARI::znx_normalize_first_step_carry_only(base2k, lsh, a.at(a_col, a_size - j - 1), carry);
        } else {
            ZNXARI::znx_normalize_middle_step_carry_only(base2k, lsh, a.at(a_col, a_size - j - 1), carry);
        }
    }

    if a_out_range == 0 {
        ZNXARI::znx_zero(carry);
    }

    // Zeroes lower limbs of res if a_size + steps < res_size
    for j in 0..res_size {
        ZNXARI::znx_zero(res.at_mut(res_col, j));
    }

    // Continues with shifted normalization
    let mid_range: usize = res_start.saturating_sub(res_end);

    for j in 0..mid_range {
        ZNXARI::znx_normalize_middle_step(
            base2k,
            lsh,
            res.at_mut(res_col, res_start - j - 1),
            a.at(a_col, a_start - j - 1),
            carry,
        );
    }

    // Propagates carry on the rest of the limbs of res
    for j in 0..res_end {
        if j == res_end - 1 {
            ZNXARI::znx_normalize_final_step_inplace(base2k, lsh, res.at_mut(res_col, res_end - j - 1), carry);
        } else {
            ZNXARI::znx_normalize_middle_step_inplace(base2k, lsh, res.at_mut(res_col, res_end - j - 1), carry);
        }
    }
}

pub fn bench_vec_znx_lsh_inplace<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B> + VecZnxLshInplace<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_lsh_inplace::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxLshInplace<B> + ModuleNew<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let base2k: usize = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(vec_znx_lsh_tmp_bytes(n));

        // Fill a with random i64
        a.fill_uniform(50, &mut source);
        b.fill_uniform(50, &mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_lsh_inplace(base2k, base2k - 1, &mut b, i, scratch.borrow());
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2]));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_lsh<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxLsh<B> + ModuleNew<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_lsh::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxLsh<B> + ModuleNew<B>,
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

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(vec_znx_lsh_tmp_bytes(n));

        // Fill a with random i64
        a.fill_uniform(50, &mut source);
        res.fill_uniform(50, &mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_lsh(base2k, base2k - 1, &mut res, i, &a, i, scratch.borrow());
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2]));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_rsh_inplace<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxRshInplace<B> + ModuleNew<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_rsh_inplace::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxRshInplace<B> + ModuleNew<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let base2k: usize = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(vec_znx_rsh_tmp_bytes(n));

        // Fill a with random i64
        a.fill_uniform(50, &mut source);
        b.fill_uniform(50, &mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_rsh_inplace(base2k, base2k - 1, &mut b, i, scratch.borrow());
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2]));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_rsh<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxRsh<B> + ModuleNew<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_rsh::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxRsh<B> + ModuleNew<B>,
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

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(vec_znx_rsh_tmp_bytes(n));

        // Fill a with random i64
        a.fill_uniform(50, &mut source);
        res.fill_uniform(50, &mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_rsh(base2k, base2k - 1, &mut res, i, &a, i, scratch.borrow());
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2]));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

#[cfg(test)]
mod tests {
    use crate::{
        layouts::{FillUniform, VecZnx, ZnxView},
        reference::{
            vec_znx::{
                vec_znx_copy, vec_znx_lsh, vec_znx_lsh_inplace, vec_znx_lsh_tmp_bytes, vec_znx_normalize_inplace, vec_znx_rsh,
                vec_znx_rsh_inplace, vec_znx_rsh_tmp_bytes, vec_znx_sub_inplace,
            },
            znx::ZnxRef,
        },
        source::Source,
    };

    #[test]
    fn test_vec_znx_lsh() {
        let n: usize = 8;
        let cols: usize = 2;
        let size: usize = 7;

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        let mut source: Source = Source::new([0u8; 32]);

        let mut carry: Vec<i64> = vec![0i64; vec_znx_lsh_tmp_bytes(n) / size_of::<i64>()];

        let base2k: usize = 50;

        for k in 0..256 {
            a.fill_uniform(50, &mut source);

            for i in 0..cols {
                vec_znx_normalize_inplace::<_, ZnxRef>(base2k, &mut a, i, &mut carry);
                vec_znx_copy::<_, _, ZnxRef>(&mut res_ref, i, &a, i);
            }

            for i in 0..cols {
                vec_znx_lsh_inplace::<_, ZnxRef>(base2k, k, &mut res_ref, i, &mut carry);
                vec_znx_lsh::<_, _, ZnxRef>(base2k, k, &mut res_test, i, &a, i, &mut carry);
                vec_znx_normalize_inplace::<_, ZnxRef>(base2k, &mut res_test, i, &mut carry);
            }

            assert_eq!(res_ref, res_test);
        }
    }

    #[test]
    fn test_vec_znx_rsh() {
        let n: usize = 8;
        let cols: usize = 2;

        let res_size: usize = 7;

        let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
        let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

        let mut carry: Vec<i64> = vec![0i64; vec_znx_rsh_tmp_bytes(n) / size_of::<i64>()];

        let base2k: usize = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let zero: Vec<i64> = vec![0i64; n];

        for a_size in [res_size - 1, res_size, res_size + 1] {
            let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);

            for k in 0..res_size * base2k {
                a.fill_uniform(50, &mut source);

                for i in 0..cols {
                    vec_znx_normalize_inplace::<_, ZnxRef>(base2k, &mut a, i, &mut carry);
                    vec_znx_copy::<_, _, ZnxRef>(&mut res_ref, i, &a, i);
                }

                res_test.fill_uniform(50, &mut source);

                for j in 0..cols {
                    vec_znx_rsh_inplace::<_, ZnxRef>(base2k, k, &mut res_ref, j, &mut carry);
                    vec_znx_rsh::<_, _, ZnxRef>(base2k, k, &mut res_test, j, &a, j, &mut carry);
                }

                for j in 0..cols {
                    vec_znx_lsh_inplace::<_, ZnxRef>(base2k, k, &mut res_ref, j, &mut carry);
                    vec_znx_lsh_inplace::<_, ZnxRef>(base2k, k, &mut res_test, j, &mut carry);
                }

                // Case where res has enough to fully store a right shifted without any loss
                // In this case we can check exact equality.
                if a_size + k.div_ceil(base2k) <= res_size {
                    assert_eq!(res_ref, res_test);

                    for i in 0..cols {
                        for j in 0..a_size {
                            assert_eq!(res_ref.at(i, j), a.at(i, j), "r0 {} {}", i, j);
                            assert_eq!(res_test.at(i, j), a.at(i, j), "r1 {} {}", i, j);
                        }

                        for j in a_size..res_size {
                            assert_eq!(res_ref.at(i, j), zero, "r0 {} {}", i, j);
                            assert_eq!(res_test.at(i, j), zero, "r1 {} {}", i, j);
                        }
                    }
                // Some loss occures, either because a initially has more precision than res
                // or because the storage of the right shift of a requires more precision than
                // res.
                } else {
                    for j in 0..cols {
                        vec_znx_sub_inplace::<_, _, ZnxRef>(&mut res_ref, j, &a, j);
                        vec_znx_sub_inplace::<_, _, ZnxRef>(&mut res_test, j, &a, j);

                        vec_znx_normalize_inplace::<_, ZnxRef>(base2k, &mut res_ref, j, &mut carry);
                        vec_znx_normalize_inplace::<_, ZnxRef>(base2k, &mut res_test, j, &mut carry);

                        assert!(res_ref.stats(base2k, j).std().log2() - (k as f64) <= (k * base2k) as f64);
                        assert!(res_test.stats(base2k, j).std().log2() - (k as f64) <= (k * base2k) as f64);
                    }
                }
            }
        }
    }
}
