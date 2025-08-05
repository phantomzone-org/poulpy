use std::fmt;

use crate::{
    hal::{
        api::{TakeSlice, VecZnxDftToVecZnxBigTmpBytes, ZnxInfos, ZnxSliceSize, ZnxView, ZnxViewMut, ZnxZero},
        layouts::{
            Module, Scratch, VecZnx, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftBytesOf, VecZnxDftOwned, VecZnxDftToMut,
            VecZnxDftToRef, VecZnxToRef,
        },
        oep::{
            VecZnxDftAddImpl, VecZnxDftAddInplaceImpl, VecZnxDftAllocBytesImpl, VecZnxDftAllocImpl, VecZnxDftCopyImpl,
            VecZnxDftFromBytesImpl, VecZnxDftFromVecZnxImpl, VecZnxDftSubABInplaceImpl, VecZnxDftSubBAInplaceImpl,
            VecZnxDftSubImpl, VecZnxDftToVecZnxBigConsumeImpl, VecZnxDftToVecZnxBigImpl, VecZnxDftToVecZnxBigTmpAImpl,
            VecZnxDftToVecZnxBigTmpBytesImpl, VecZnxDftZeroImpl,
        },
    },
    implementation::cpu_spqlios::{
        ffi::{vec_znx_big, vec_znx_dft},
        module_fft64::FFT64,
    },
};

const VEC_ZNX_DFT_FFT64_WORDSIZE: usize = 1;

impl<D> ZnxSliceSize for VecZnxDft<D, FFT64> {
    fn sl(&self) -> usize {
        VEC_ZNX_DFT_FFT64_WORDSIZE * self.n() * self.cols()
    }
}

impl<D: AsRef<[u8]>> VecZnxDftBytesOf for VecZnxDft<D, FFT64> {
    fn bytes_of(n: usize, cols: usize, size: usize) -> usize {
        VEC_ZNX_DFT_FFT64_WORDSIZE * n * cols * size * size_of::<f64>()
    }
}

impl<D: AsRef<[u8]>> ZnxView for VecZnxDft<D, FFT64> {
    type Scalar = f64;
}

unsafe impl VecZnxDftFromBytesImpl<FFT64> for FFT64 {
    fn vec_znx_dft_from_bytes_impl(n: usize, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxDftOwned<FFT64> {
        VecZnxDft::<Vec<u8>, FFT64>::from_bytes(n, cols, size, bytes)
    }
}

unsafe impl VecZnxDftAllocBytesImpl<FFT64> for FFT64 {
    fn vec_znx_dft_alloc_bytes_impl(n: usize, cols: usize, size: usize) -> usize {
        VecZnxDft::<Vec<u8>, FFT64>::bytes_of(n, cols, size)
    }
}

unsafe impl VecZnxDftAllocImpl<FFT64> for FFT64 {
    fn vec_znx_dft_alloc_impl(n: usize, cols: usize, size: usize) -> VecZnxDftOwned<FFT64> {
        VecZnxDftOwned::alloc(n, cols, size)
    }
}

unsafe impl VecZnxDftToVecZnxBigTmpBytesImpl<FFT64> for FFT64 {
    fn vec_znx_dft_to_vec_znx_big_tmp_bytes_impl(module: &Module<FFT64>) -> usize {
        unsafe { vec_znx_dft::vec_znx_idft_tmp_bytes(module.ptr()) as usize }
    }
}

unsafe impl VecZnxDftToVecZnxBigImpl<FFT64> for FFT64 {
    fn vec_znx_dft_to_vec_znx_big_impl<R, A>(
        module: &Module<FFT64>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<FFT64>,
    ) where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxDftToRef<FFT64>,
    {
        let mut res_mut: VecZnxBig<&mut [u8], FFT64> = res.to_mut();
        let a_ref: VecZnxDft<&[u8], FFT64> = a.to_ref();

        let (tmp_bytes, _) = scratch.take_slice(module.vec_znx_dft_to_vec_znx_big_tmp_bytes());

        let min_size: usize = res_mut.size().min(a_ref.size());

        unsafe {
            (0..min_size).for_each(|j| {
                vec_znx_dft::vec_znx_idft(
                    module.ptr(),
                    res_mut.at_mut_ptr(res_col, j) as *mut vec_znx_big::vec_znx_big_t,
                    1 as u64,
                    a_ref.at_ptr(a_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                    1 as u64,
                    tmp_bytes.as_mut_ptr(),
                )
            });
            (min_size..res_mut.size()).for_each(|j| {
                res_mut.zero_at(res_col, j);
            });
        }
    }
}

unsafe impl VecZnxDftToVecZnxBigTmpAImpl<FFT64> for FFT64 {
    fn vec_znx_dft_to_vec_znx_big_tmp_a_impl<R, A>(module: &Module<FFT64>, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxDftToMut<FFT64>,
    {
        let mut res_mut: VecZnxBig<&mut [u8], FFT64> = res.to_mut();
        let mut a_mut: VecZnxDft<&mut [u8], FFT64> = a.to_mut();

        let min_size: usize = res_mut.size().min(a_mut.size());

        unsafe {
            (0..min_size).for_each(|j| {
                vec_znx_dft::vec_znx_idft_tmp_a(
                    module.ptr(),
                    res_mut.at_mut_ptr(res_col, j) as *mut vec_znx_big::vec_znx_big_t,
                    1 as u64,
                    a_mut.at_mut_ptr(a_col, j) as *mut vec_znx_dft::vec_znx_dft_t,
                    1 as u64,
                )
            });
            (min_size..res_mut.size()).for_each(|j| {
                res_mut.zero_at(res_col, j);
            })
        }
    }
}

unsafe impl VecZnxDftToVecZnxBigConsumeImpl<FFT64> for FFT64 {
    fn vec_znx_dft_to_vec_znx_big_consume_impl<D>(module: &Module<FFT64>, mut a: VecZnxDft<D, FFT64>) -> VecZnxBig<D, FFT64>
    where
        VecZnxDft<D, FFT64>: VecZnxDftToMut<FFT64>,
    {
        let mut a_mut: VecZnxDft<&mut [u8], FFT64> = a.to_mut();

        unsafe {
            // Rev col and rows because ZnxDft.sl() >= ZnxBig.sl()
            (0..a_mut.size()).for_each(|j| {
                (0..a_mut.cols()).for_each(|i| {
                    vec_znx_dft::vec_znx_idft_tmp_a(
                        module.ptr(),
                        a_mut.at_mut_ptr(i, j) as *mut vec_znx_big::vec_znx_big_t,
                        1 as u64,
                        a_mut.at_mut_ptr(i, j) as *mut vec_znx_dft::vec_znx_dft_t,
                        1 as u64,
                    )
                });
            });
        }

        a.into_big()
    }
}

unsafe impl VecZnxDftFromVecZnxImpl<FFT64> for FFT64 {
    fn vec_znx_dft_from_vec_znx_impl<R, A>(
        module: &Module<FFT64>,
        step: usize,
        offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        R: VecZnxDftToMut<FFT64>,
        A: VecZnxToRef,
    {
        let mut res_mut: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a_ref: VecZnx<&[u8]> = a.to_ref();
        let steps: usize = a_ref.size().div_ceil(step);
        let min_steps: usize = res_mut.size().min(steps);
        unsafe {
            (0..min_steps).for_each(|j| {
                let limb: usize = offset + j * step;
                if limb < a_ref.size() {
                    vec_znx_dft::vec_znx_dft(
                        module.ptr(),
                        res_mut.at_mut_ptr(res_col, j) as *mut vec_znx_dft::vec_znx_dft_t,
                        1 as u64,
                        a_ref.at_ptr(a_col, limb),
                        1 as u64,
                        a_ref.sl() as u64,
                    )
                }
            });
            (min_steps..res_mut.size()).for_each(|j| {
                res_mut.zero_at(res_col, j);
            });
        }
    }
}

unsafe impl VecZnxDftAddImpl<FFT64> for FFT64 {
    fn vec_znx_dft_add_impl<R, A, D>(
        module: &Module<FFT64>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &D,
        b_col: usize,
    ) where
        R: VecZnxDftToMut<FFT64>,
        A: VecZnxDftToRef<FFT64>,
        D: VecZnxDftToRef<FFT64>,
    {
        let mut res_mut: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a_ref: VecZnxDft<&[u8], FFT64> = a.to_ref();
        let b_ref: VecZnxDft<&[u8], FFT64> = b.to_ref();

        let min_size: usize = res_mut.size().min(a_ref.size()).min(b_ref.size());

        unsafe {
            (0..min_size).for_each(|j| {
                vec_znx_dft::vec_dft_add(
                    module.ptr(),
                    res_mut.at_mut_ptr(res_col, j) as *mut vec_znx_dft::vec_znx_dft_t,
                    1,
                    a_ref.at_ptr(a_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                    1,
                    b_ref.at_ptr(b_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                    1,
                );
            });
        }
        (min_size..res_mut.size()).for_each(|j| {
            res_mut.zero_at(res_col, j);
        })
    }
}

unsafe impl VecZnxDftAddInplaceImpl<FFT64> for FFT64 {
    fn vec_znx_dft_add_inplace_impl<R, A>(module: &Module<FFT64>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<FFT64>,
        A: VecZnxDftToRef<FFT64>,
    {
        let mut res_mut: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a_ref: VecZnxDft<&[u8], FFT64> = a.to_ref();

        let min_size: usize = res_mut.size().min(a_ref.size());

        unsafe {
            (0..min_size).for_each(|j| {
                vec_znx_dft::vec_dft_add(
                    module.ptr(),
                    res_mut.at_mut_ptr(res_col, j) as *mut vec_znx_dft::vec_znx_dft_t,
                    1,
                    res_mut.at_ptr(res_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                    1,
                    a_ref.at_ptr(a_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                    1,
                );
            });
        }
    }
}

unsafe impl VecZnxDftSubImpl<FFT64> for FFT64 {
    fn vec_znx_dft_sub_impl<R, A, D>(
        module: &Module<FFT64>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &D,
        b_col: usize,
    ) where
        R: VecZnxDftToMut<FFT64>,
        A: VecZnxDftToRef<FFT64>,
        D: VecZnxDftToRef<FFT64>,
    {
        let mut res_mut: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a_ref: VecZnxDft<&[u8], FFT64> = a.to_ref();
        let b_ref: VecZnxDft<&[u8], FFT64> = b.to_ref();

        let min_size: usize = res_mut.size().min(a_ref.size()).min(b_ref.size());

        unsafe {
            (0..min_size).for_each(|j| {
                vec_znx_dft::vec_dft_sub(
                    module.ptr(),
                    res_mut.at_mut_ptr(res_col, j) as *mut vec_znx_dft::vec_znx_dft_t,
                    1,
                    a_ref.at_ptr(a_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                    1,
                    b_ref.at_ptr(b_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                    1,
                );
            });
        }
        (min_size..res_mut.size()).for_each(|j| {
            res_mut.zero_at(res_col, j);
        })
    }
}

unsafe impl VecZnxDftSubABInplaceImpl<FFT64> for FFT64 {
    fn vec_znx_dft_sub_ab_inplace_impl<R, A>(module: &Module<FFT64>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<FFT64>,
        A: VecZnxDftToRef<FFT64>,
    {
        let mut res_mut: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a_ref: VecZnxDft<&[u8], FFT64> = a.to_ref();

        let min_size: usize = res_mut.size().min(a_ref.size());

        unsafe {
            (0..min_size).for_each(|j| {
                vec_znx_dft::vec_dft_sub(
                    module.ptr(),
                    res_mut.at_mut_ptr(res_col, j) as *mut vec_znx_dft::vec_znx_dft_t,
                    1,
                    res_mut.at_ptr(res_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                    1,
                    a_ref.at_ptr(a_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                    1,
                );
            });
        }
    }
}

unsafe impl VecZnxDftSubBAInplaceImpl<FFT64> for FFT64 {
    fn vec_znx_dft_sub_ba_inplace_impl<R, A>(module: &Module<FFT64>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<FFT64>,
        A: VecZnxDftToRef<FFT64>,
    {
        let mut res_mut: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a_ref: VecZnxDft<&[u8], FFT64> = a.to_ref();

        let min_size: usize = res_mut.size().min(a_ref.size());

        unsafe {
            (0..min_size).for_each(|j| {
                vec_znx_dft::vec_dft_sub(
                    module.ptr(),
                    res_mut.at_mut_ptr(res_col, j) as *mut vec_znx_dft::vec_znx_dft_t,
                    1,
                    a_ref.at_ptr(a_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                    1,
                    res_mut.at_ptr(res_col, j) as *const vec_znx_dft::vec_znx_dft_t,
                    1,
                );
            });
        }
    }
}

unsafe impl VecZnxDftCopyImpl<FFT64> for FFT64 {
    fn vec_znx_dft_copy_impl<R, A>(
        _module: &Module<FFT64>,
        step: usize,
        offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        R: VecZnxDftToMut<FFT64>,
        A: VecZnxDftToRef<FFT64>,
    {
        let mut res_mut: VecZnxDft<&mut [u8], FFT64> = res.to_mut();
        let a_ref: VecZnxDft<&[u8], FFT64> = a.to_ref();

        let steps: usize = a_ref.size().div_ceil(step);
        let min_steps: usize = res_mut.size().min(steps);

        (0..min_steps).for_each(|j| {
            let limb: usize = offset + j * step;
            if limb < a_ref.size() {
                res_mut
                    .at_mut(res_col, j)
                    .copy_from_slice(a_ref.at(a_col, limb));
            }
        });
        (min_steps..res_mut.size()).for_each(|j| {
            res_mut.zero_at(res_col, j);
        })
    }
}

unsafe impl VecZnxDftZeroImpl<FFT64> for FFT64 {
    fn vec_znx_dft_zero_impl<R>(_module: &Module<FFT64>, res: &mut R)
    where
        R: VecZnxDftToMut<FFT64>,
    {
        res.to_mut().data.fill(0);
    }
}

impl<D: AsRef<[u8]>> fmt::Display for VecZnxDft<D, FFT64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "VecZnxDft(n={}, cols={}, size={})",
            self.n, self.cols, self.size
        )?;

        for col in 0..self.cols {
            writeln!(f, "Column {}:", col)?;
            for size in 0..self.size {
                let coeffs = self.at(col, size);
                write!(f, "  Size {}: [", size)?;

                let max_show = 100;
                let show_count = coeffs.len().min(max_show);

                for (i, &coeff) in coeffs.iter().take(show_count).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", coeff)?;
                }

                if coeffs.len() > max_show {
                    write!(f, ", ... ({} more)", coeffs.len() - max_show)?;
                }

                writeln!(f, "]")?;
            }
        }
        Ok(())
    }
}
