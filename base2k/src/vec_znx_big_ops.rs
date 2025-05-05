use crate::ffi::vec_znx;
use crate::znx_base::{ZnxInfos, ZnxView, ZnxViewMut};
use crate::{
    Backend, FFT64, Module, Scratch, VecZnx, VecZnxBig, VecZnxBigOwned, VecZnxBigToMut, VecZnxBigToRef, VecZnxScratch,
    VecZnxToMut, VecZnxToRef, ZnxSliceSize, bytes_of_vec_znx_big,
};

pub trait VecZnxBigAlloc<B: Backend> {
    /// Allocates a vector Z[X]/(X^N+1) that stores not normalized values.
    fn new_vec_znx_big(&self, cols: usize, size: usize) -> VecZnxBigOwned<B>;

    /// Returns a new [VecZnxBig] with the provided bytes array as backing array.
    ///
    /// Behavior: takes ownership of the backing array.
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of polynomials..
    /// * `size`: the number of polynomials per column.
    /// * `bytes`: a byte array of size at least [Module::bytes_of_vec_znx_big].
    ///
    /// # Panics
    /// If `bytes.len()` < [Module::bytes_of_vec_znx_big].
    fn new_vec_znx_big_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxBigOwned<B>;

    // /// Returns a new [VecZnxBig] with the provided bytes array as backing array.
    // ///
    // /// Behavior: the backing array is only borrowed.
    // ///
    // /// # Arguments
    // ///
    // /// * `cols`: the number of polynomials..
    // /// * `size`: the number of polynomials per column.
    // /// * `bytes`: a byte array of size at least [Module::bytes_of_vec_znx_big].
    // ///
    // /// # Panics
    // /// If `bytes.len()` < [Module::bytes_of_vec_znx_big].
    // fn new_vec_znx_big_from_bytes_borrow(&self, cols: usize, size: usize, tmp_bytes: &mut [u8]) -> VecZnxBig<B>;

    /// Returns the minimum number of bytes necessary to allocate
    /// a new [VecZnxBig] through [VecZnxBig::from_bytes].
    fn bytes_of_vec_znx_big(&self, cols: usize, size: usize) -> usize;
}

pub trait VecZnxBigOps<BACKEND: Backend> {
    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxBigToRef<BACKEND>,
        B: VecZnxBigToRef<BACKEND>;

    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxBigToRef<BACKEND>;

    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add_small<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxBigToRef<BACKEND>,
        B: VecZnxToRef;

    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_small_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxToRef;

    /// Subtracts `a` to `b` and stores the result on `c`.
    fn vec_znx_big_sub<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxBigToRef<BACKEND>,
        B: VecZnxBigToRef<BACKEND>;

    /// Subtracts `a` from `b` and stores the result on `b`.
    fn vec_znx_big_sub_ab_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxBigToRef<BACKEND>;

    /// Subtracts `b` from `a` and stores the result on `b`.
    fn vec_znx_big_sub_ba_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxBigToRef<BACKEND>;

    /// Subtracts `b` from `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_a<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxToRef,
        B: VecZnxBigToRef<BACKEND>;

    /// Subtracts `a` from `res` and stores the result on `res`.
    fn vec_znx_big_sub_small_a_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxToRef;

    /// Subtracts `b` from `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_b<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxBigToRef<BACKEND>,
        B: VecZnxToRef;

    /// Subtracts `res` from `a` and stores the result on `res`.
    fn vec_znx_big_sub_small_b_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxToRef;

    /// Normalizes `a` and stores the result on `b`.
    ///
    /// # Arguments
    ///
    /// * `log_base2k`: normalization basis.
    /// * `tmp_bytes`: scratch space of size at least [VecZnxBigOps::vec_znx_big_normalize].
    fn vec_znx_big_normalize<R, A>(
        &self,
        log_base2k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch,
    ) where
        R: VecZnxToMut,
        A: VecZnxBigToRef<BACKEND>;

    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `b`.
    fn vec_znx_big_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<BACKEND>,
        A: VecZnxBigToRef<BACKEND>;

    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `a`.
    fn vec_znx_big_automorphism_inplace<A>(&self, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxBigToMut<BACKEND>;
}

pub trait VecZnxBigScratch {
    /// Returns the minimum number of bytes to apply [VecZnxBigOps::vec_znx_big_normalize].
    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize;
}

impl VecZnxBigAlloc<FFT64> for Module<FFT64> {
    fn new_vec_znx_big(&self, cols: usize, size: usize) -> VecZnxBigOwned<FFT64> {
        VecZnxBig::new(self, cols, size)
    }

    fn new_vec_znx_big_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxBigOwned<FFT64> {
        VecZnxBig::new_from_bytes(self, cols, size, bytes)
    }

    fn bytes_of_vec_znx_big(&self, cols: usize, size: usize) -> usize {
        bytes_of_vec_znx_big(self, cols, size)
    }
}

impl VecZnxBigOps<FFT64> for Module<FFT64> {
    fn vec_znx_big_add<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
        B: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let b: VecZnxBig<&[u8], FFT64> = b.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(res.n(), self.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_add(
                self.ptr,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                b.at_ptr(b_col, 0),
                b.size() as u64,
                b.sl() as u64,
            )
        }
    }

    fn vec_znx_big_add_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_add(
                self.ptr,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                res.at_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
            )
        }
    }

    fn vec_znx_big_sub<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
        B: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let b: VecZnxBig<&[u8], FFT64> = b.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(res.n(), self.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                b.at_ptr(b_col, 0),
                b.size() as u64,
                b.sl() as u64,
            )
        }
    }

    fn vec_znx_big_sub_ab_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                res.at_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }

    fn vec_znx_big_sub_ba_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                res.at_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
            )
        }
    }

    fn vec_znx_big_sub_small_b<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
        B: VecZnxToRef,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let b: VecZnx<&[u8]> = b.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(res.n(), self.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                b.at_ptr(b_col, 0),
                b.size() as u64,
                b.sl() as u64,
            )
        }
    }

    fn vec_znx_big_sub_small_b_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                res.at_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
            )
        }
    }

    fn vec_znx_big_sub_small_a<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxToRef,
        B: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let b: VecZnxBig<&[u8], FFT64> = b.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(res.n(), self.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                b.at_ptr(b_col, 0),
                b.size() as u64,
                b.sl() as u64,
            )
        }
    }

    fn vec_znx_big_sub_small_a_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                res.at_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }

    fn vec_znx_big_add_small<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
        B: VecZnxToRef,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let b: VecZnx<&[u8]> = b.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(res.n(), self.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_add(
                self.ptr,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                b.at_ptr(b_col, 0),
                b.size() as u64,
                b.sl() as u64,
            )
        }
    }

    fn vec_znx_big_add_small_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_add(
                self.ptr,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                res.at_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }

    fn vec_znx_big_normalize<R, A>(
        &self,
        log_base2k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch,
    ) where
        R: VecZnxToMut,
        A: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
            //(Jay)Note: This is calling VezZnxOps::vec_znx_normalize_tmp_bytes and not VecZnxBigOps::vec_znx_big_normalize_tmp_bytes.
            // In the FFT backend the tmp sizes are same but will be different in the NTT backend
            // assert!(tmp_bytes.len() >= <Self as VecZnxOps<&mut [u8], & [u8]>>::vec_znx_normalize_tmp_bytes(&self));
            // assert_alignement(tmp_bytes.as_ptr());
        }

        let (tmp_bytes, _) = scratch.tmp_scalar_slice(<Self as VecZnxBigScratch>::vec_znx_big_normalize_tmp_bytes(
            &self,
        ));
        unsafe {
            vec_znx::vec_znx_normalize_base2k(
                self.ptr,
                log_base2k as u64,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                tmp_bytes.as_mut_ptr(),
            );
        }
    }

    fn vec_znx_big_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<FFT64>,
        A: VecZnxBigToRef<FFT64>,
    {
        let a: VecZnxBig<&[u8], FFT64> = a.to_ref();
        let mut res: VecZnxBig<&mut [u8], FFT64> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_automorphism(
                self.ptr,
                k,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }

    fn vec_znx_big_automorphism_inplace<A>(&self, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxBigToMut<FFT64>,
    {
        let mut a: VecZnxBig<&mut [u8], FFT64> = a.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_automorphism(
                self.ptr,
                k,
                a.at_mut_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }
}

impl<B: Backend> VecZnxBigScratch for Module<B> {
    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize {
        <Self as VecZnxScratch>::vec_znx_normalize_tmp_bytes(self)
    }
}
