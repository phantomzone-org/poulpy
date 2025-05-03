use crate::ffi::vec_znx;
use crate::znx_base::{ZnxInfos, ZnxView, ZnxViewMut};
use crate::{Backend, DataView, FFT64, Module, ScratchSpace, VecZnx, VecZnxBig, VecZnxBigOwned, VecZnxOps, assert_alignement};

pub trait VecZnxBigAlloc<B> {
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

pub trait VecZnxBigOps<DataMut, Data, B> {
    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add(
        &self,
        res: &mut VecZnxBig<DataMut, B>,
        res_col: usize,
        a: &VecZnxBig<Data, B>,
        a_col: usize,
        b: &VecZnxBig<Data, B>,
        b_col: usize,
    );

    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_inplace(&self, res: &mut VecZnxBig<DataMut, B>, res_col: usize, a: &VecZnxBig<Data, B>, a_col: usize);

    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add_small(
        &self,
        res: &mut VecZnxBig<DataMut, B>,
        res_col: usize,
        a: &VecZnxBig<Data, B>,
        a_col: usize,
        b: &VecZnx<Data>,
        b_col: usize,
    );

    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_small_inplace(&self, res: &mut VecZnxBig<DataMut, B>, res_col: usize, a: &VecZnx<Data>, a_col: usize);

    /// Subtracts `a` to `b` and stores the result on `c`.
    fn vec_znx_big_sub(
        &self,
        res: &mut VecZnxBig<DataMut, B>,
        res_col: usize,
        a: &VecZnxBig<Data, B>,
        a_col: usize,
        b: &VecZnxBig<Data, B>,
        b_col: usize,
    );

    /// Subtracts `a` from `b` and stores the result on `b`.
    fn vec_znx_big_sub_ab_inplace(&self, res: &mut VecZnxBig<DataMut, B>, res_col: usize, a: &VecZnxBig<Data, B>, a_col: usize);

    /// Subtracts `b` from `a` and stores the result on `b`.
    fn vec_znx_big_sub_ba_inplace(&self, res: &mut VecZnxBig<DataMut, B>, res_col: usize, a: &VecZnxBig<Data, B>, a_col: usize);

    /// Subtracts `b` from `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_a(
        &self,
        res: &mut VecZnxBig<DataMut, B>,
        res_col: usize,
        a: &VecZnx<Data>,
        a_col: usize,
        b: &VecZnxBig<Data, B>,
        b_col: usize,
    );

    /// Subtracts `a` from `res` and stores the result on `res`.
    fn vec_znx_big_sub_small_a_inplace(&self, res: &mut VecZnxBig<DataMut, B>, res_col: usize, a: &VecZnx<Data>, a_col: usize);

    /// Subtracts `b` from `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_b(
        &self,
        res: &mut VecZnxBig<DataMut, B>,
        res_col: usize,
        a: &VecZnxBig<Data, B>,
        a_col: usize,
        b: &VecZnx<Data>,
        b_col: usize,
    );

    /// Subtracts `res` from `a` and stores the result on `res`.
    fn vec_znx_big_sub_small_b_inplace(&self, res: &mut VecZnxBig<DataMut, B>, res_col: usize, a: &VecZnx<Data>, a_col: usize);

    /// Returns the minimum number of bytes to apply [VecZnxBigOps::vec_znx_big_normalize].
    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize;

    /// Normalizes `a` and stores the result on `b`.
    ///
    /// # Arguments
    ///
    /// * `log_base2k`: normalization basis.
    /// * `tmp_bytes`: scratch space of size at least [VecZnxBigOps::vec_znx_big_normalize].
    fn vec_znx_big_normalize(
        &self,
        log_base2k: usize,
        res: &mut VecZnx<DataMut>,
        res_col: usize,
        a: &VecZnxBig<Data, B>,
        a_col: usize,
        scratch: &mut ScratchSpace,
    );

    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `b`.
    fn vec_znx_big_automorphism(
        &self,
        k: i64,
        res: &mut VecZnxBig<DataMut, B>,
        res_col: usize,
        a: &VecZnxBig<Data, B>,
        a_col: usize,
    );

    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `a`.
    fn vec_znx_big_automorphism_inplace(&self, k: i64, a: &mut VecZnxBig<DataMut, B>, a_col: usize);
}

impl VecZnxBigAlloc<FFT64> for Module<FFT64> {
    fn new_vec_znx_big(&self, cols: usize, size: usize) -> VecZnxBigOwned<FFT64> {
        VecZnxBig::new(self, cols, size)
    }

    fn new_vec_znx_big_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxBigOwned<FFT64> {
        VecZnxBig::new_from_bytes(self, cols, size, bytes)
    }

    // fn new_vec_znx_big_from_bytes_borrow(&self, cols: usize, size: usize, tmp_bytes: &mut [u8]) -> VecZnxBig<FFT64> {
    // VecZnxBig::from_bytes_borrow(self, 1, cols, size, tmp_bytes)
    // }

    fn bytes_of_vec_znx_big(&self, cols: usize, size: usize) -> usize {
        VecZnxBigOwned::bytes_of(self, cols, size)
    }
}

impl<DataMut, Data> VecZnxBigOps<DataMut, Data, FFT64> for Module<FFT64>
where
    DataMut: AsMut<[u8]> + AsRef<[u8]>,
    Data: AsRef<[u8]>,
{
    fn vec_znx_big_add(
        &self,
        res: &mut VecZnxBig<DataMut, FFT64>,
        res_col: usize,
        a: &VecZnxBig<Data, FFT64>,
        a_col: usize,
        b: &VecZnxBig<Data, FFT64>,
        b_col: usize,
    ) {
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

    fn vec_znx_big_add_inplace(
        &self,
        res: &mut VecZnxBig<DataMut, FFT64>,
        res_col: usize,
        a: &VecZnxBig<Data, FFT64>,
        a_col: usize,
    ) {
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

    fn vec_znx_big_sub(
        &self,
        res: &mut VecZnxBig<DataMut, FFT64>,
        res_col: usize,
        a: &VecZnxBig<Data, FFT64>,
        a_col: usize,
        b: &VecZnxBig<Data, FFT64>,
        b_col: usize,
    ) {
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

    fn vec_znx_big_sub_ab_inplace(
        &self,
        res: &mut VecZnxBig<DataMut, FFT64>,
        res_col: usize,
        a: &VecZnxBig<Data, FFT64>,
        a_col: usize,
    ) {
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

    fn vec_znx_big_sub_ba_inplace(
        &self,
        res: &mut VecZnxBig<DataMut, FFT64>,
        res_col: usize,
        a: &VecZnxBig<Data, FFT64>,
        a_col: usize,
    ) {
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

    fn vec_znx_big_sub_small_b(
        &self,
        res: &mut VecZnxBig<DataMut, FFT64>,
        res_col: usize,
        a: &VecZnxBig<Data, FFT64>,
        a_col: usize,
        b: &VecZnx<Data>,
        b_col: usize,
    ) {
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

    fn vec_znx_big_sub_small_b_inplace(
        &self,
        res: &mut VecZnxBig<DataMut, FFT64>,
        res_col: usize,
        a: &VecZnx<Data>,
        a_col: usize,
    ) {
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

    fn vec_znx_big_sub_small_a(
        &self,
        res: &mut VecZnxBig<DataMut, FFT64>,
        res_col: usize,
        a: &VecZnx<Data>,
        a_col: usize,
        b: &VecZnxBig<Data, FFT64>,
        b_col: usize,
    ) {
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

    fn vec_znx_big_sub_small_a_inplace(
        &self,
        res: &mut VecZnxBig<DataMut, FFT64>,
        res_col: usize,
        a: &VecZnx<Data>,
        a_col: usize,
    ) {
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

    fn vec_znx_big_add_small(
        &self,
        res: &mut VecZnxBig<DataMut, FFT64>,
        res_col: usize,
        a: &VecZnxBig<Data, FFT64>,
        a_col: usize,
        b: &VecZnx<Data>,
        b_col: usize,
    ) {
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

    fn vec_znx_big_add_small_inplace(&self, res: &mut VecZnxBig<DataMut, FFT64>, res_col: usize, a: &VecZnx<Data>, a_col: usize) {
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

    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize {
        <Self as VecZnxOps<DataMut, Data>>::vec_znx_normalize_tmp_bytes(self)
    }

    fn vec_znx_big_normalize(
        &self,
        log_base2k: usize,
        res: &mut VecZnx<DataMut>,
        res_col: usize,
        a: &VecZnxBig<Data, FFT64>,
        a_col: usize,
        scratch: &mut ScratchSpace,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
            //(Jay)Note: This is calling VezZnxOps::vec_znx_normalize_tmp_bytes and not VecZnxBigOps::vec_znx_big_normalize_tmp_bytes.
            // In the FFT backend the tmp sizes are same but will be different in the NTT backend
            // assert!(tmp_bytes.len() >= <Self as VecZnxOps<DataMut, Data>>::vec_znx_normalize_tmp_bytes(&self));
            // assert_alignement(tmp_bytes.as_ptr());
        }
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
                scratch.vec_znx_big_normalize_tmp_bytes(self).as_mut_ptr(),
            );
        }
    }

    fn vec_znx_big_automorphism(
        &self,
        k: i64,
        res: &mut VecZnxBig<DataMut, FFT64>,
        res_col: usize,
        a: &VecZnxBig<Data, FFT64>,
        a_col: usize,
    ) {
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

    fn vec_znx_big_automorphism_inplace(&self, k: i64, a: &mut VecZnxBig<DataMut, FFT64>, a_col: usize) {
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
