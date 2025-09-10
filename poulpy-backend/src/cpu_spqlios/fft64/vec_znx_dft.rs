use poulpy_hal::{
    api::{TakeSlice, VecZnxIdftApplyTmpBytes},
    layouts::{
        Backend, Data, Module, Scratch, VecZnx, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftOwned, VecZnxDftToMut,
        VecZnxDftToRef, VecZnxToRef, ZnxInfos, ZnxSliceSize, ZnxView, ZnxViewMut,
    },
    oep::{
        VecZnxDftAddImpl, VecZnxDftAddInplaceImpl, VecZnxDftAllocBytesImpl, VecZnxDftAllocImpl, VecZnxDftApplyImpl,
        VecZnxDftCopyImpl, VecZnxDftFromBytesImpl, VecZnxDftSubABInplaceImpl, VecZnxDftSubBAInplaceImpl, VecZnxDftSubImpl,
        VecZnxDftZeroImpl, VecZnxIdftApplyConsumeImpl, VecZnxIdftApplyImpl, VecZnxIdftApplyTmpAImpl, VecZnxIdftApplyTmpBytesImpl,
    },
    reference::{
        reim::{ReimArithmeticRef, reim_copy_ref, reim_negate_inplace_ref, reim_negate_ref, reim_zero_ref},
        vec_znx_dft::fft64::vec_znx_dft_copy,
        znx::{ZnxArithmetic, ZnxArithmeticRef},
    },
};

use crate::cpu_spqlios::{
    FFT64,
    ffi::{vec_znx_big, vec_znx_dft},
};

unsafe impl VecZnxDftFromBytesImpl<Self> for FFT64 {
    fn vec_znx_dft_from_bytes_impl(n: usize, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxDftOwned<Self> {
        VecZnxDft::<Vec<u8>, FFT64>::from_bytes(n, cols, size, bytes)
    }
}

unsafe impl VecZnxDftAllocBytesImpl<Self> for FFT64 {
    fn vec_znx_dft_alloc_bytes_impl(n: usize, cols: usize, size: usize) -> usize {
        FFT64::layout_prep_word_count() * n * cols * size * size_of::<<FFT64 as Backend>::ScalarPrep>()
    }
}

unsafe impl VecZnxDftAllocImpl<Self> for FFT64 {
    fn vec_znx_dft_alloc_impl(n: usize, cols: usize, size: usize) -> VecZnxDftOwned<Self> {
        VecZnxDftOwned::alloc(n, cols, size)
    }
}

unsafe impl VecZnxIdftApplyTmpBytesImpl<Self> for FFT64 {
    fn vec_znx_idft_apply_tmp_bytes_impl(module: &Module<Self>) -> usize {
        unsafe { vec_znx_dft::vec_znx_idft_tmp_bytes(module.ptr()) as usize }
    }
}

unsafe impl VecZnxIdftApplyImpl<Self> for FFT64 {
    fn vec_znx_idft_apply_impl<R, A>(
        module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxBigToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();
        let a: VecZnxDft<&[u8], FFT64> = a.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), a.n())
        }

        let (tmp_bytes, _) = scratch.take_slice(module.vec_znx_idft_apply_tmp_bytes());

        let min_size: usize = res.size().min(a.size());

        unsafe {
            (0..min_size).for_each(|j| {
                vec_znx_dft::vec_znx_idft(
                    module.ptr(),
                    res.at_mut_ptr(res_col, j) as *mut vec_znx_big::vec_znx_big_t,
                    1_u64,
                    a.at_ptr(a_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                    1_u64,
                    tmp_bytes.as_mut_ptr(),
                )
            });
            (min_size..res.size()).for_each(|j| ZnxArithmeticRef::znx_zero(res.at_mut(res_col, j)));
        }
    }
}

unsafe impl VecZnxIdftApplyTmpAImpl<Self> for FFT64 {
    fn vec_znx_idft_apply_tmpa_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxDftToMut<Self>,
    {
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();
        let mut a_mut: VecZnxDft<&mut [u8], FFT64> = a.to_mut();

        let min_size: usize = res.size().min(a_mut.size());

        unsafe {
            (0..min_size).for_each(|j| {
                vec_znx_dft::vec_znx_idft_tmp_a(
                    module.ptr(),
                    res.at_mut_ptr(res_col, j) as *mut vec_znx_big::vec_znx_big_t,
                    1_u64,
                    a_mut.at_mut_ptr(a_col, j) as *mut vec_znx_dft::vec_znx_dft_t,
                    1_u64,
                )
            });
            (min_size..res.size()).for_each(|j| ZnxArithmeticRef::znx_zero(res.at_mut(res_col, j)))
        }
    }
}

unsafe impl VecZnxIdftApplyConsumeImpl<Self> for FFT64 {
    fn vec_znx_idft_apply_consume_impl<D: Data>(module: &Module<Self>, mut a: VecZnxDft<D, FFT64>) -> VecZnxBig<D, FFT64>
    where
        VecZnxDft<D, FFT64>: VecZnxDftToMut<Self>,
    {
        let mut a_mut: VecZnxDft<&mut [u8], FFT64> = a.to_mut();

        unsafe {
            // Rev col and rows because ZnxDft.sl() >= ZnxBig.sl()
            (0..a_mut.size()).for_each(|j| {
                (0..a_mut.cols()).for_each(|i| {
                    vec_znx_dft::vec_znx_idft_tmp_a(
                        module.ptr(),
                        a_mut.at_mut_ptr(i, j) as *mut vec_znx_big::vec_znx_big_t,
                        1_u64,
                        a_mut.at_mut_ptr(i, j) as *mut vec_znx_dft::vec_znx_dft_t,
                        1_u64,
                    )
                });
            });
        }

        a.into_big()
    }
}

unsafe impl VecZnxDftApplyImpl<Self> for FFT64 {
    fn vec_znx_dft_apply_impl<R, A>(
        module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        R: VecZnxDftToMut<Self>,
        A: VecZnxToRef,
    {
        let mut res: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a: VecZnx<&[u8]> = a.to_ref();
        let steps: usize = a.size().div_ceil(step);
        let min_steps: usize = res.size().min(steps);
        unsafe {
            (0..min_steps).for_each(|j| {
                let limb: usize = offset + j * step;
                if limb < a.size() {
                    vec_znx_dft::vec_znx_dft(
                        module.ptr(),
                        res.at_mut_ptr(res_col, j) as *mut vec_znx_dft::vec_znx_dft_t,
                        1_u64,
                        a.at_ptr(a_col, limb),
                        1_u64,
                        a.sl() as u64,
                    )
                }
            });
            (min_steps..res.size()).for_each(|j| reim_zero_ref(res.at_mut(res_col, j)));
        }
    }
}

unsafe impl VecZnxDftAddImpl<Self> for FFT64 {
    fn vec_znx_dft_add_impl<R, A, D>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
        D: VecZnxDftToRef<Self>,
    {
        let mut res: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a: VecZnxDft<&[u8], FFT64> = a.to_ref();
        let b: VecZnxDft<&[u8], FFT64> = b.to_ref();

        let res_size: usize = res.size();
        let a_size: usize = a.size();
        let b_size: usize = b.size();

        unsafe {
            if a_size <= b_size {
                let sum_size: usize = a_size.min(res_size);
                let cpy_size: usize = b_size.min(res_size);

                (0..sum_size).for_each(|j| {
                    vec_znx_dft::vec_dft_add(
                        module.ptr(),
                        res.at_mut_ptr(res_col, j) as *mut vec_znx_dft::vec_znx_dft_t,
                        1,
                        a.at_ptr(a_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                        1,
                        b.at_ptr(b_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                        1,
                    );
                });

                for j in sum_size..cpy_size {
                    reim_copy_ref(res.at_mut(res_col, j), b.at(b_col, j));
                }

                for j in cpy_size..res_size {
                    reim_zero_ref(res.at_mut(res_col, j));
                }
            } else {
                let sum_size: usize = b_size.min(res_size);
                let cpy_size: usize = a_size.min(res_size);

                (0..sum_size).for_each(|j| {
                    vec_znx_dft::vec_dft_add(
                        module.ptr(),
                        res.at_mut_ptr(res_col, j) as *mut vec_znx_dft::vec_znx_dft_t,
                        1,
                        a.at_ptr(a_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                        1,
                        b.at_ptr(b_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                        1,
                    );
                });

                for j in sum_size..cpy_size {
                    reim_copy_ref(res.at_mut(res_col, j), a.at(b_col, j));
                }

                for j in cpy_size..res_size {
                    reim_zero_ref(res.at_mut(res_col, j));
                }
            }
        }
    }
}

unsafe impl VecZnxDftAddInplaceImpl<Self> for FFT64 {
    fn vec_znx_dft_add_inplace_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        let mut res: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a: VecZnxDft<&[u8], FFT64> = a.to_ref();

        let min_size: usize = res.size().min(a.size());

        unsafe {
            (0..min_size).for_each(|j| {
                vec_znx_dft::vec_dft_add(
                    module.ptr(),
                    res.at_mut_ptr(res_col, j) as *mut vec_znx_dft::vec_znx_dft_t,
                    1,
                    res.at_ptr(res_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                    1,
                    a.at_ptr(a_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                    1,
                );
            });
        }
    }
}

unsafe impl VecZnxDftSubImpl<Self> for FFT64 {
    fn vec_znx_dft_sub_impl<R, A, D>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
        D: VecZnxDftToRef<Self>,
    {
        let mut res: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a: VecZnxDft<&[u8], FFT64> = a.to_ref();
        let b: VecZnxDft<&[u8], FFT64> = b.to_ref();

        unsafe {
            let res_size: usize = res.size();
            let a_size: usize = a.size();
            let b_size: usize = b.size();

            if a_size <= b_size {
                let sum_size: usize = a_size.min(res_size);
                let cpy_size: usize = b_size.min(res_size);

                (0..sum_size).for_each(|j| {
                    vec_znx_dft::vec_dft_sub(
                        module.ptr(),
                        res.at_mut_ptr(res_col, j) as *mut vec_znx_dft::vec_znx_dft_t,
                        1,
                        a.at_ptr(a_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                        1,
                        b.at_ptr(b_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                        1,
                    );
                });

                for j in sum_size..cpy_size {
                    reim_negate_ref(res.at_mut(res_col, j), b.at(b_col, j));
                }

                for j in cpy_size..res_size {
                    reim_zero_ref(res.at_mut(res_col, j));
                }
            } else {
                let sum_size: usize = b_size.min(res_size);
                let cpy_size: usize = a_size.min(res_size);

                (0..sum_size).for_each(|j| {
                    vec_znx_dft::vec_dft_sub(
                        module.ptr(),
                        res.at_mut_ptr(res_col, j) as *mut vec_znx_dft::vec_znx_dft_t,
                        1,
                        a.at_ptr(a_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                        1,
                        b.at_ptr(b_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                        1,
                    );
                });

                for j in sum_size..cpy_size {
                    reim_copy_ref(res.at_mut(res_col, j), a.at(a_col, j));
                }

                for j in cpy_size..res_size {
                    reim_zero_ref(res.at_mut(res_col, j));
                }
            }
        }
    }
}

unsafe impl VecZnxDftSubABInplaceImpl<Self> for FFT64 {
    fn vec_znx_dft_sub_ab_inplace_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        let mut res: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a: VecZnxDft<&[u8], FFT64> = a.to_ref();

        let min_size: usize = res.size().min(a.size());

        unsafe {
            (0..min_size).for_each(|j| {
                vec_znx_dft::vec_dft_sub(
                    module.ptr(),
                    res.at_mut_ptr(res_col, j) as *mut vec_znx_dft::vec_znx_dft_t,
                    1,
                    res.at_ptr(res_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                    1,
                    a.at_ptr(a_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                    1,
                );
            });
        }
    }
}

unsafe impl VecZnxDftSubBAInplaceImpl<Self> for FFT64 {
    fn vec_znx_dft_sub_ba_inplace_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        let mut res: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a: VecZnxDft<&[u8], FFT64> = a.to_ref();

        let min_size: usize = res.size().min(a.size());

        unsafe {
            (0..min_size).for_each(|j| {
                vec_znx_dft::vec_dft_sub(
                    module.ptr(),
                    res.at_mut_ptr(res_col, j) as *mut vec_znx_dft::vec_znx_dft_t,
                    1,
                    a.at_ptr(a_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                    1,
                    res.at_ptr(res_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                    1,
                );
            });

            for j in min_size..res.size() {
                reim_negate_inplace_ref(res.at_mut(res_col, j));
            }
        }
    }
}

unsafe impl VecZnxDftCopyImpl<Self> for FFT64 {
    fn vec_znx_dft_copy_impl<R, A>(
        _module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        vec_znx_dft_copy::<_, _, _, ReimArithmeticRef>(step, offset, res, res_col, a, a_col);
    }
}

unsafe impl VecZnxDftZeroImpl<Self> for FFT64 {
    fn vec_znx_dft_zero_impl<R>(_module: &Module<Self>, res: &mut R)
    where
        R: VecZnxDftToMut<Self>,
    {
        res.to_mut().data.fill(0);
    }
}
