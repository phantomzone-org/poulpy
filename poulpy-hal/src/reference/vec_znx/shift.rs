use crate::{
    layouts::{VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut, ZnxZero},
    reference::{
        vec_znx::vec_znx_copy_ref,
        znx::{
            znx_copy_ref, znx_normalize_beg_ref, znx_normalize_carry_only_beg_ref, znx_normalize_carry_only_mid_ref,
            znx_normalize_end_ref, znx_normalize_inplace_beg_ref, znx_normalize_inplace_end_ref, znx_normalize_inplace_mid_ref,
            znx_normalize_mid_ref,
        },
    },
};

pub fn vec_znx_lsh_inplace_ref<R>(basek: usize, k: usize, res: &mut R, res_col: usize, carry: &mut [i64])
where
    R: VecZnxToMut,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    let n: usize = res.n();
    let cols: usize = res.cols();
    let size: usize = res.size();
    let steps: usize = k / basek;
    let k_rem: usize = k % basek;

    if steps >= size {
        res.zero();
        return;
    }

    // Inplace shift of limbs by a k/basek
    if steps > 0 {
        let start: usize = n * res_col;
        let end: usize = start + n;
        let slice_size: usize = n * cols;
        let res_raw: &mut [i64] = res.raw_mut();

        (0..size - steps).for_each(|j| {
            let (lhs, rhs) = res_raw.split_at_mut(slice_size * (j + steps));
            znx_copy_ref(
                &mut lhs[start + j * slice_size..end + j * slice_size],
                &rhs[start..end],
            );
        });

        (size - steps..size).for_each(|j| {
            res.zero_at(res_col, j);
        });
    }

    // Inplace normalization with left shift of k % basek
    if !k.is_multiple_of(basek) {
        for j in (0..size - steps).rev() {
            if j == size - steps - 1 {
                znx_normalize_inplace_beg_ref(basek, k_rem, res.at_mut(res_col, j), carry);
            } else if j == 0 {
                znx_normalize_inplace_end_ref(basek, k_rem, res.at_mut(res_col, j), carry);
            } else {
                znx_normalize_inplace_mid_ref(basek, k_rem, res.at_mut(res_col, j), carry);
            }
        }
    }
}

pub fn vec_znx_lsh_ref<R, A>(basek: usize, k: usize, res: &mut R, res_col: usize, a: &A, a_col: usize, carry: &mut [i64])
where
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    let res_size: usize = res.size();
    let a_size = a.size();
    let steps: usize = k / basek;
    let k_rem: usize = k % basek;

    if steps >= res_size {
        res.zero();
        return;
    }

    let min_size: usize = a_size.min(res_size - steps);

    // Simply a left shifted normalization of limbs
    // by k/basek and intra-limb by basek - k%basek
    if !k.is_multiple_of(basek) {
        for j in (0..min_size).rev() {
            if j == min_size - 1 {
                znx_normalize_beg_ref(
                    basek,
                    k_rem,
                    res.at_mut(res_col, j),
                    a.at(a_col, j + steps),
                    carry,
                );
            } else if j == 0 {
                znx_normalize_end_ref(
                    basek,
                    k_rem,
                    res.at_mut(res_col, j),
                    a.at(a_col, j + steps),
                    carry,
                );
            } else {
                znx_normalize_mid_ref(
                    basek,
                    k_rem,
                    res.at_mut(res_col, j),
                    a.at(a_col, j + steps),
                    carry,
                );
            }
        }
    } else {
        // If k % basek = 0, then this is simply a copy.
        for j in (0..min_size).rev() {
            znx_copy_ref(res.at_mut(res_col, j), a.at(a_col, j + steps));
        }
    }

    // Zeroes bottom
    (min_size..res_size).for_each(|j| {
        res.zero_at(res_col, j);
    });
}

pub fn vec_znx_rsh_inplace_ref<R>(basek: usize, k: usize, res: &mut R, res_col: usize, carry: &mut [i64])
where
    R: VecZnxToMut,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let n: usize = res.n();
    let cols: usize = res.cols();
    let size: usize = res.size();

    let mut steps: usize = k / basek;
    let k_rem: usize = k % basek;

    if k == 0 {
        return;
    }

    if steps >= size {
        res.zero();
        return;
    }

    let start: usize = n * res_col;
    let end: usize = start + n;
    let slice_size: usize = n * cols;

    if !k.is_multiple_of(basek) {
        // We rsh by an additional basek and then lsh by basek-k
        // Allows to re-use efficient normalization code, avoids
        // avoids overflows & produce output that is normalized
        steps += 1;

        // All limbs of a that would fall outside of the limbs of res are discarded,
        // but the carry still need to be computed.
        (size - steps..size).rev().for_each(|j| {
            if j == size - 1 {
                znx_normalize_carry_only_beg_ref(basek, basek - k_rem, res.at(res_col, j), carry);
            } else {
                znx_normalize_carry_only_mid_ref(basek, basek - k_rem, res.at(res_col, j), carry);
            }
        });

        // Continues with shifted normalization
        let res_raw: &mut [i64] = res.raw_mut();
        (steps..size).rev().for_each(|j| {
            let (lhs, rhs) = res_raw.split_at_mut(slice_size * j);
            let rhs_slice: &mut [i64] = &mut rhs[start..end];
            let lhs_slice: &[i64] = &lhs[(j - steps) * slice_size + start..(j - steps) * slice_size + end];
            znx_normalize_mid_ref(basek, basek - k_rem, rhs_slice, lhs_slice, carry);
        });

        // Propagates carry on the rest of the limbs of res
        for j in (0..steps).rev() {
            res.zero_at(res_col, j);
            if j == 0 {
                znx_normalize_inplace_end_ref(basek, basek - k_rem, res.at_mut(res_col, j), carry);
            } else {
                znx_normalize_inplace_mid_ref(basek, basek - k_rem, res.at_mut(res_col, j), carry);
            }
        }
    } else {
        // Shift by multiples of basek
        let res_raw: &mut [i64] = res.raw_mut();
        (steps..size).rev().for_each(|j| {
            let (lhs, rhs) = res_raw.split_at_mut(slice_size * j);
            znx_copy_ref(
                &mut rhs[start..end],
                &lhs[(j - steps) * slice_size + start..(j - steps) * slice_size + end],
            );
        });

        // Zeroes the top
        (0..steps).for_each(|j| {
            res.zero_at(res_col, j);
        });
    }
}

pub fn vec_znx_rsh_ref<R, A>(basek: usize, k: usize, res: &mut R, res_col: usize, a: &A, a_col: usize, carry: &mut [i64])
where
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let mut steps: usize = k / basek;
    let k_rem: usize = k % basek;

    if k == 0 {
        vec_znx_copy_ref(&mut res, res_col, &a, a_col);
        return;
    }

    if steps >= res_size {
        res.zero();
        return;
    }

    if !k.is_multiple_of(basek) {
        // We rsh by an additional basek and then lsh by basek-k
        // Allows to re-use efficient normalization code, avoids
        // avoids overflows & produce output that is normalized
        steps += 1;

        // All limbs of a that are moved outside of the limbs of res are discarded,
        // but the carry still need to be computed.
        for j in (res_size..a_size + steps).rev() {
            if j == a_size + steps - 1 {
                znx_normalize_carry_only_beg_ref(basek, basek - k_rem, a.at(a_col, j - steps), carry);
            } else {
                znx_normalize_carry_only_mid_ref(basek, basek - k_rem, a.at(a_col, j - steps), carry);
            }
        }

        // Avoids over flow of limbs of res
        let min_size: usize = res_size.min(a_size + steps);

        // Zeroes lower limbs of res if a_size + steps < res_size
        (min_size..res_size).for_each(|j| {
            res.zero_at(res_col, j);
        });

        // Continues with shifted normalization
        for j in (steps..min_size).rev() {
            // Case if no limb of a was previously discarded
            if res_size.saturating_sub(steps) >= a_size && j == min_size - 1 {
                znx_normalize_beg_ref(
                    basek,
                    basek - k_rem,
                    res.at_mut(res_col, j),
                    a.at(a_col, j - steps),
                    carry,
                );
            } else {
                znx_normalize_mid_ref(
                    basek,
                    basek - k_rem,
                    res.at_mut(res_col, j),
                    a.at(a_col, j - steps),
                    carry,
                );
            }
        }

        // Propagates carry on the rest of the limbs of res
        for j in (0..steps).rev() {
            res.zero_at(res_col, j);
            if j == 0 {
                znx_normalize_inplace_end_ref(basek, basek - k_rem, res.at_mut(res_col, j), carry);
            } else {
                znx_normalize_inplace_mid_ref(basek, basek - k_rem, res.at_mut(res_col, j), carry);
            }
        }
    } else {
        let min_size: usize = res_size.min(a_size + steps);

        // Zeroes the top
        (0..steps).for_each(|j| {
            res.zero_at(res_col, j);
        });

        // Shift a into res, up to the maximum
        for j in (steps..min_size).rev() {
            znx_copy_ref(res.at_mut(res_col, j), a.at(a_col, j - steps));
        }

        // Zeroes bottom if a_size + steps < res_size
        (min_size..res_size).for_each(|j| {
            res.zero_at(res_col, j);
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        layouts::{FillUniform, VecZnx, ZnxView},
        reference::vec_znx::{
            vec_znx_copy_ref, vec_znx_lsh_inplace_ref, vec_znx_lsh_ref, vec_znx_normalize_inplace_ref, vec_znx_rsh_inplace_ref,
            vec_znx_rsh_ref, vec_znx_sub_ab_inplace_ref,
        },
        source::Source,
    };

    #[test]
    fn test_vec_znx_lsh() {
        let n: usize = 8;
        let cols: usize = 2;
        let size: usize = 7;

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        let mut res_0: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        let mut res_1: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        let mut source: Source = Source::new([0u8; 32]);

        let mut carry: Vec<i64> = vec![0i64; n];

        let basek: usize = 50;

        for k in 0..256 {
            a.fill_uniform(&mut source);

            for i in 0..cols {
                vec_znx_normalize_inplace_ref(basek, &mut a, i, &mut carry);
                vec_znx_copy_ref(&mut res_0, i, &a, i);
            }

            for i in 0..cols {
                vec_znx_lsh_inplace_ref(basek, k, &mut res_0, i, &mut carry);
                vec_znx_lsh_ref(basek, k, &mut res_1, i, &a, i, &mut carry);
                vec_znx_normalize_inplace_ref(basek, &mut res_1, i, &mut carry);
            }

            assert_eq!(res_0, res_1);
        }
    }

    #[test]
    fn test_vec_znx_rsh() {
        let n: usize = 8;
        let cols: usize = 2;

        let res_size: usize = 7;

        let mut res_0: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);
        let mut res_1: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, res_size);

        let mut carry: Vec<i64> = vec![0i64; n];

        let basek: usize = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let zero: Vec<i64> = vec![0i64; n];

        for a_size in [res_size - 1, res_size, res_size + 1] {
            let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, a_size);

            for k in 0..res_size * basek {
                a.fill_uniform(&mut source);

                for i in 0..cols {
                    vec_znx_normalize_inplace_ref(basek, &mut a, i, &mut carry);
                    vec_znx_copy_ref(&mut res_0, i, &a, i);
                }

                res_1.fill_uniform(&mut source);

                for j in 0..cols {
                    vec_znx_rsh_inplace_ref(basek, k, &mut res_0, j, &mut carry);
                    vec_znx_rsh_ref(basek, k, &mut res_1, j, &a, j, &mut carry);
                }

                for j in 0..cols {
                    vec_znx_lsh_inplace_ref(basek, k, &mut res_0, j, &mut carry);
                    vec_znx_lsh_inplace_ref(basek, k, &mut res_1, j, &mut carry);
                }

                // Case where res has enough to fully store a right shifted without any loss
                // In this case we can check exact equality.
                if a_size + k.div_ceil(basek) <= res_size {
                    assert_eq!(res_0, res_1);

                    for i in 0..cols {
                        for j in 0..a_size {
                            assert_eq!(res_0.at(i, j), a.at(i, j), "r0 {} {}", i, j);
                            assert_eq!(res_1.at(i, j), a.at(i, j), "r1 {} {}", i, j);
                        }

                        for j in a_size..res_size {
                            assert_eq!(res_0.at(i, j), zero, "r0 {} {}", i, j);
                            assert_eq!(res_1.at(i, j), zero, "r1 {} {}", i, j);
                        }
                    }
                // Some loss occures, either because a initially has more precision than res
                // or because the storage of the right shift of a requires more precision than
                // res.
                } else {
                    for j in 0..cols {
                        vec_znx_sub_ab_inplace_ref(&mut res_0, j, &a, j);
                        vec_znx_sub_ab_inplace_ref(&mut res_1, j, &a, j);

                        vec_znx_normalize_inplace_ref(basek, &mut res_0, j, &mut carry);
                        vec_znx_normalize_inplace_ref(basek, &mut res_1, j, &mut carry);

                        assert!(res_0.std(basek, j).log2() - (k as f64) <= (k * basek) as f64);
                        assert!(res_1.std(basek, j).log2() - (k as f64) <= (k * basek) as f64);
                    }
                }
            }
        }
    }
}
