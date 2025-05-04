use crate::ffi::vec_znx_dft::vec_znx_dft_t;
use crate::ffi::vmp;
use crate::znx_base::{ZnxInfos, ZnxView, ZnxViewMut};
use crate::{
    Backend, FFT64, MatZnxDft, MatZnxDftAllocOwned, Module, ScratchBorr, VecZnx, VecZnxBigOps, VecZnxBigScratch, VecZnxDft,
    VecZnxDftAlloc, VecZnxDftOps,
};

pub trait MatZnxDftAlloc<B> {
    /// Allocates a new [MatZnxDft] with the given number of rows and columns.
    ///
    /// # Arguments
    ///
    /// * `rows`: number of rows (number of [VecZnxDft]).
    /// * `size`: number of size (number of size of each [VecZnxDft]).
    fn new_mat_znx_dft(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnxDftAllocOwned<B>;

    fn bytes_of_mat_znx_dft(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;

    fn new_mat_znx_dft_from_bytes(
        &self,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: Vec<u8>,
    ) -> MatZnxDftAllocOwned<B>;

    // fn new_mat_znx_dft_from_bytes_borrow(
    //     &self,
    //     rows: usize,
    //     cols_in: usize,
    //     cols_out: usize,
    //     size: usize,
    //     bytes: &mut [u8],
    // ) -> MatZnxDft<FFT64>;
}

pub trait MatZnxDftScratch {
    /// Returns the of bytes needed as scratch space for [MatZnxDftOps::vmp_prepare_row]
    fn vmp_prepare_row_tmp_bytes(&self, cols_out: usize, size: usize) -> usize;

    /// Returns the of bytes needed as scratch space for [MatZnxDftOps::vmp_extract_row]
    fn vmp_extract_row_tmp_bytes(&self, cols_out: usize, size: usize) -> usize;

    /// Returns the size of the stratch space necessary for [MatZnxDftOps::vmp_apply_dft].
    ///
    /// # Arguments
    ///
    /// * `c_size`: number of size of the output [VecZnxDft].
    /// * `a_size`: number of size of the input [VecZnx].
    /// * `rows`: number of rows of the input [MatZnxDft].
    /// * `size`: number of size of the input [MatZnxDft].
    fn vmp_apply_dft_tmp_bytes(
        &self,
        c_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;

    /// Returns the size of the stratch space necessary for [MatZnxDftOps::vmp_apply_dft_to_dft].
    ///
    /// # Arguments
    ///
    /// * `c_size`: number of size of the output [VecZnxDft].
    /// * `a_size`: number of size of the input [VecZnxDft].
    /// * `rows`: number of rows of the input [MatZnxDft].
    /// * `size`: number of size of the input [MatZnxDft].
    fn vmp_apply_dft_to_dft_tmp_bytes(
        &self,
        c_cols: usize,
        c_size: usize,
        a_cols: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;
}

/// This trait implements methods for vector matrix product,
/// that is, multiplying a [VecZnx] with a [MatZnxDft].
pub trait MatZnxDftOps<DataMut, Data, B: Backend> {
    /// Prepares the ith-row of [MatZnxDft] from a [VecZnx].
    ///
    /// # Arguments
    ///
    /// * `b`: [MatZnxDft] on which the values are encoded.
    /// * `row_i`: the row of the [MatZnxDft] to prepare.
    /// * `a`: the [VecZnx] to encode on the i-th row of the [MatZnxDft].
    /// * `buf`: scratch space, the size of buf can be obtained with [MatZnxDftOps::vmp_prepare_tmp_bytes].
    ///
    /// The size of buf can be obtained with [MatZnxDftOps::vmp_prepare_tmp_bytes].
    fn vmp_prepare_row(
        &self,
        b: &mut MatZnxDft<DataMut, B>,
        b_row: usize,
        b_col_in: usize,
        a: &VecZnx<Data>,
        scratch: &mut ScratchBorr,
    );

    /// Extracts the ith-row of [MatZnxDft] into a [VecZnxBig].
    ///
    /// # Arguments
    ///
    /// * `b`: the [VecZnxBig] to on which to extract the row of the [MatZnxDft].
    /// * `a`: [MatZnxDft] on which the values are encoded.
    /// * `row_i`: the index of the row to extract.
    fn vmp_extract_row(
        &self,
        log_base2k: usize,
        b: &mut VecZnx<DataMut>,
        a: &MatZnxDft<Data, B>,
        b_row: usize,
        b_col_in: usize,
        scratch: &mut ScratchBorr,
    );

    /// Prepares the ith-row of [MatZnxDft] from a [VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `b`: [MatZnxDft] on which the values are encoded.
    /// * `a`: the [VecZnxDft] to encode on the [MatZnxDft].
    /// * `row_i`: the index of the row to prepare.
    ///
    /// The size of buf can be obtained with [MatZnxDftOps::vmp_prepare_tmp_bytes].
    fn vmp_prepare_row_dft(&self, b: &mut MatZnxDft<DataMut, B>, b_row: usize, b_col_in: usize, a: &VecZnxDft<Data, B>);

    /// Extracts the ith-row of [MatZnxDft] into a [VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `b`: the [VecZnxDft] to on which to extract the row of the [MatZnxDft].
    /// * `a`: [MatZnxDft] on which the values are encoded.
    /// * `row_i`: the index of the row to extract.
    fn vmp_extract_row_dft(&self, b: &mut VecZnxDft<DataMut, B>, a: &MatZnxDft<Data, B>, a_row: usize, a_col_in: usize);

    /// Applies the vector matrix product [VecZnxDft] x [MatZnxDft].
    ///
    /// A vector matrix product is equivalent to a sum of [crate::SvpPPolOps::svp_apply_dft]
    /// where each [crate::Scalar] is a limb of the input [VecZnxDft] (equivalent to an [crate::SvpPPol])
    /// and each vector a [VecZnxDft] (row) of the [MatZnxDft].
    ///
    /// As such, given an input [VecZnx] of `i` size and a [MatZnxDft] of `i` rows and
    /// `j` size, the output is a [VecZnx] of `j` size.
    ///
    /// If there is a mismatch between the dimensions the largest valid ones are used.
    ///
    /// ```text
    /// |a b c d| x |e f g| = (a * |e f g| + b * |h i j| + c * |k l m|) = |n o p|
    ///             |h i j|
    ///             |k l m|
    /// ```
    /// where each element is a [VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `c`: the output of the vector matrix product, as a [VecZnxDft].
    /// * `a`: the left operand [VecZnx] of the vector matrix product.
    /// * `b`: the right operand [MatZnxDft] of the vector matrix product.
    /// * `buf`: scratch space, the size can be obtained with [MatZnxDftOps::vmp_apply_dft_tmp_bytes].
    fn vmp_apply_dft(&self, c: &mut VecZnxDft<DataMut, B>, a: &VecZnx<Data>, b: &MatZnxDft<Data, B>, scratch: &mut ScratchBorr);

    /// Applies the vector matrix product [VecZnxDft] x [MatZnxDft].
    /// The size of `buf` is given by [MatZnxDftOps::vmp_apply_dft_to_dft_tmp_bytes].
    ///
    /// A vector matrix product is equivalent to a sum of [crate::SvpPPolOps::svp_apply_dft]
    /// where each [crate::Scalar] is a limb of the input [VecZnxDft] (equivalent to an [crate::SvpPPol])
    /// and each vector a [VecZnxDft] (row) of the [MatZnxDft].
    ///
    /// As such, given an input [VecZnx] of `i` size and a [MatZnxDft] of `i` rows and
    /// `j` size, the output is a [VecZnx] of `j` size.
    ///
    /// If there is a mismatch between the dimensions the largest valid ones are used.
    ///
    /// ```text
    /// |a b c d| x |e f g| = (a * |e f g| + b * |h i j| + c * |k l m|) = |n o p|
    ///             |h i j|
    ///             |k l m|
    /// ```
    /// where each element is a [VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `c`: the output of the vector matrix product, as a [VecZnxDft].
    /// * `a`: the left operand [VecZnxDft] of the vector matrix product.
    /// * `b`: the right operand [MatZnxDft] of the vector matrix product.
    /// * `buf`: scratch space, the size can be obtained with [MatZnxDftOps::vmp_apply_dft_to_dft_tmp_bytes].
    fn vmp_apply_dft_to_dft(
        &self,
        c: &mut VecZnxDft<DataMut, B>,
        a: &VecZnxDft<Data, B>,
        b: &MatZnxDft<Data, B>,
        scratch: &mut ScratchBorr,
    );
}

impl<B: Backend> MatZnxDftAlloc<B> for Module<B> {
    fn bytes_of_mat_znx_dft(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        MatZnxDftAllocOwned::bytes_of(self, rows, cols_in, cols_out, size)
    }

    fn new_mat_znx_dft(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnxDftAllocOwned<B> {
        MatZnxDftAllocOwned::new(self, rows, cols_in, cols_out, size)
    }

    fn new_mat_znx_dft_from_bytes(
        &self,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
        bytes: Vec<u8>,
    ) -> MatZnxDftAllocOwned<B> {
        MatZnxDftAllocOwned::new_from_bytes(self, rows, cols_in, cols_out, size, bytes)
    }
}

impl<B: Backend> MatZnxDftScratch for Module<B> {
    fn vmp_prepare_row_tmp_bytes(&self, cols_out: usize, size: usize) -> usize {
        <Self as VecZnxDftAlloc<_>>::bytes_of_vec_znx_dft(self, cols_out, size)
    }

    fn vmp_extract_row_tmp_bytes(&self, cols_out: usize, size: usize) -> usize {
        <Self as VecZnxDftAlloc<_>>::bytes_of_vec_znx_dft(self, cols_out, size)
            + <Self as VecZnxBigScratch>::vec_znx_big_normalize_tmp_bytes(self)
    }

    fn vmp_apply_dft_tmp_bytes(
        &self,
        c_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        unsafe {
            vmp::vmp_apply_dft_tmp_bytes(
                self.ptr,
                c_size as u64,
                a_size as u64,
                (b_rows * b_cols_in) as u64,
                (b_size * b_cols_out) as u64,
            ) as usize
        }
    }
    fn vmp_apply_dft_to_dft_tmp_bytes(
        &self,
        c_cols: usize,
        c_size: usize,
        a_cols: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        unsafe {
            vmp::vmp_apply_dft_to_dft_tmp_bytes(
                self.ptr,
                (c_size * c_cols) as u64,
                (a_size * a_cols) as u64,
                (b_rows * b_cols_in) as u64,
                (b_size * b_cols_out) as u64,
            ) as usize
        }
    }
}

impl<DataMut, Data> MatZnxDftOps<DataMut, Data, FFT64> for Module<FFT64>
where
    DataMut: AsMut<[u8]> + AsRef<[u8]> + for<'a> From<&'a mut [u8]>,
    Data: AsRef<[u8]>,
{
    fn vmp_prepare_row(
        &self,
        b: &mut MatZnxDft<DataMut, FFT64>,
        b_row: usize,
        b_col_in: usize,
        a: &VecZnx<Data>,
        scratch: &mut ScratchBorr,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(b.n(), self.n());
            assert_eq!(a.n(), self.n());
            assert_eq!(
                a.cols(),
                b.cols_out(),
                "a.cols(): {} != b.cols_out(): {}",
                a.cols(),
                b.cols_out()
            );
            assert!(
                b_row < b.rows(),
                "b_row: {} >= b.rows(): {}",
                b_row,
                b.rows()
            );
            assert!(
                b_col_in < b.cols_in(),
                "b_col_in: {} >= b.cols_in(): {}",
                b_col_in,
                b.cols_in()
            );
            assert_eq!(
                b.size(),
                a.size(),
                "b.size(): {} != a.size(): {}",
                b.size(),
                a.size()
            );
            // assert!(
            //     tmp_bytes.len()
            //         >= <Self as MatZnxDftOps<DataMut, Data, FFT64>>::vmp_prepare_row_tmp_bytes(self, a.cols(), a.size())
            // );
            // assert!(is_aligned(tmp_bytes.as_ptr()))
        }

        let cols_out: usize = a.cols();
        let a_size: usize = a.size();

        // let (tmp_bytes_a_dft, _) = tmp_bytes.split_at_mut(self.bytes_of_vec_znx_dft(cols_out, a_size));
        let (mut a_dft, _) = scratch.tmp_scalar_slice(12);
        DataMut::from(a_dft);
        // let (mut a_dft, _) = scratch.tmp_vec_znx_dft::<DataMut, _>(self, cols_out, a_size);
        (0..cols_out).for_each(|i| self.vec_znx_dft(&mut a_dft, i, &a, i));
        Self::vmp_prepare_row_dft(&self, b, b_row, b_col_in, &a_dft);
    }

    fn vmp_extract_row(
        &self,
        log_base2k: usize,
        b: &mut VecZnx<DataMut>,
        a: &MatZnxDft<Data, FFT64>,
        a_row: usize,
        a_col_in: usize,
        mut scratch: &mut ScratchBorr,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(b.n(), self.n());
            assert_eq!(a.n(), self.n());
            assert_eq!(
                b.cols(),
                a.cols_out(),
                "b.cols(): {} != a.cols_out(): {}",
                b.cols(),
                a.cols_out()
            );
            assert!(
                a_row < a.rows(),
                "a_row: {} >= a.rows(): {}",
                a_row,
                a.rows()
            );
            assert!(
                a_col_in < a.cols_in(),
                "a_col_in: {} >= a.cols_in(): {}",
                a_col_in,
                a.cols_in()
            );
            assert_eq!(
                b.size(),
                a.size(),
                "b.size(): {} != a.size(): {}",
                b.size(),
                a.size()
            );
            // assert!(tmp_bytes.len() >= self.vmp_extract_row_tmp_bytes(a.cols(), a.size()));
            // assert!(is_aligned(tmp_bytes.as_ptr()))
        }

        let cols_out: usize = b.cols();
        let size: usize = b.size();

        // let (bytes_a_dft, tmp_bytes) = tmp_bytes.split_at_mut(self.bytes_of_vec_znx_dft(cols_out, size));
        let (mut b_dft, scratch) = scratch.tmp_vec_znx_dft(self, cols_out, size);
        Self::vmp_extract_row_dft(&self, &mut b_dft, a, a_row, a_col_in);
        let (mut b_big, scratch) = scratch.tmp_vec_znx_big(self, cols_out, size);
        (0..cols_out).for_each(|i| {
            <Self as VecZnxDftOps<DataMut, Data, FFT64>>::vec_znx_idft_tmp_a(self, &mut b_big, i, &mut b_dft, i);
            self.vec_znx_big_normalize(log_base2k, b, i, &b_big, i, scratch);
        });
    }

    fn vmp_prepare_row_dft(&self, b: &mut MatZnxDft<DataMut, FFT64>, b_row: usize, b_col_in: usize, a: &VecZnxDft<Data, FFT64>) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(b.n(), self.n());
            assert_eq!(a.n(), self.n());
            assert_eq!(
                a.cols(),
                b.cols_out(),
                "a.cols(): {} != b.cols_out(): {}",
                a.cols(),
                b.cols_out()
            );
            assert!(
                b_row < b.rows(),
                "b_row: {} >= b.rows(): {}",
                b_row,
                b.rows()
            );
            assert!(
                b_col_in < b.cols_in(),
                "b_col_in: {} >= b.cols_in(): {}",
                b_col_in,
                b.cols_in()
            );
            assert_eq!(
                b.size(),
                a.size(),
                "b.size(): {} != a.size(): {}",
                b.size(),
                a.size()
            );
        }

        unsafe {
            vmp::vmp_prepare_row_dft(
                self.ptr,
                b.as_mut_ptr() as *mut vmp::vmp_pmat_t,
                a.as_ptr() as *const vec_znx_dft_t,
                (b_row * b.cols_in() + b_col_in) as u64,
                (b.rows() * b.cols_in()) as u64,
                (b.size() * b.cols_out()) as u64,
            );
        }
    }

    fn vmp_extract_row_dft(&self, b: &mut VecZnxDft<DataMut, FFT64>, a: &MatZnxDft<Data, FFT64>, a_row: usize, a_col_in: usize) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(b.n(), self.n());
            assert_eq!(a.n(), self.n());
            assert_eq!(
                b.cols(),
                a.cols_out(),
                "b.cols(): {} != a.cols_out(): {}",
                b.cols(),
                a.cols_out()
            );
            assert!(
                a_row < a.rows(),
                "a_row: {} >= a.rows(): {}",
                a_row,
                a.rows()
            );
            assert!(
                a_col_in < a.cols_in(),
                "a_col_in: {} >= a.cols_in(): {}",
                a_col_in,
                a.cols_in()
            );
            assert_eq!(
                b.size(),
                a.size(),
                "b.size(): {} != a.size(): {}",
                b.size(),
                a.size()
            );
        }
        unsafe {
            vmp::vmp_extract_row_dft(
                self.ptr,
                b.as_mut_ptr() as *mut vec_znx_dft_t,
                a.as_ptr() as *const vmp::vmp_pmat_t,
                (a_row * a.cols_in() + a_col_in) as u64,
                (a.rows() * a.cols_in()) as u64,
                (a.size() * a.cols_out()) as u64,
            );
        }
    }

    fn vmp_apply_dft(
        &self,
        c: &mut VecZnxDft<DataMut, FFT64>,
        a: &VecZnx<Data>,
        b: &MatZnxDft<Data, FFT64>,
        mut scratch: &mut ScratchBorr,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(c.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(a.n(), self.n());
            assert_eq!(
                c.cols(),
                b.cols_out(),
                "c.cols(): {} != b.cols_out: {}",
                c.cols(),
                b.cols_out()
            );
            assert_eq!(
                a.cols(),
                b.cols_in(),
                "a.cols(): {} != b.cols_in: {}",
                a.cols(),
                b.cols_in()
            );
            // assert!(
            //     tmp_bytes.len()
            //         >= self.vmp_apply_dft_tmp_bytes(
            //             c.size(),
            //             a.size(),
            //             b.rows(),
            //             b.cols_in(),
            //             b.cols_out(),
            //             b.size()
            //         )
            // );
            // assert_alignement(tmp_bytes.as_ptr());
        }
        let (tmp_bytes, _) = scratch.tmp_scalar_slice(<Self as MatZnxDftScratch>::vmp_apply_dft_tmp_bytes(
            self,
            c.size(),
            a.size(),
            b.rows(),
            b.cols_in(),
            b.cols_out(),
            b.size(),
        ));

        unsafe {
            vmp::vmp_apply_dft(
                self.ptr,
                c.as_mut_ptr() as *mut vec_znx_dft_t,
                (c.size() * c.cols()) as u64,
                a.as_ptr(),
                (a.size() * a.cols()) as u64,
                a.n() as u64,
                b.as_ptr() as *const vmp::vmp_pmat_t,
                (b.rows() * b.cols_in()) as u64,
                (b.size() * b.cols_out()) as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }

    fn vmp_apply_dft_to_dft(
        &self,
        c: &mut VecZnxDft<DataMut, FFT64>,
        a: &VecZnxDft<Data, FFT64>,
        b: &MatZnxDft<Data, FFT64>,
        mut scratch: &mut ScratchBorr,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(c.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(a.n(), self.n());
            assert_eq!(
                c.cols(),
                b.cols_out(),
                "c.cols(): {} != b.cols_out: {}",
                c.cols(),
                b.cols_out()
            );
            assert_eq!(
                a.cols(),
                b.cols_in(),
                "a.cols(): {} != b.cols_in: {}",
                a.cols(),
                b.cols_in()
            );
            // assert!(
            //     tmp_bytes.len()
            //         >= self.vmp_apply_dft_to_dft_tmp_bytes(
            //             c.cols(),
            //             c.size(),
            //             a.cols(),
            //             a.size(),
            //             b.rows(),
            //             b.cols_in(),
            //             b.cols_out(),
            //             b.size()
            //         )
            // );
            // assert_alignement(tmp_bytes.as_ptr());
        }

        let (tmp_bytes, _) = scratch.tmp_scalar_slice(self.vmp_apply_dft_to_dft_tmp_bytes(
            c.cols(),
            c.size(),
            a.cols(),
            a.size(),
            b.rows(),
            b.cols_in(),
            b.cols_out(),
            b.size(),
        ));
        unsafe {
            vmp::vmp_apply_dft_to_dft(
                self.ptr,
                c.as_mut_ptr() as *mut vec_znx_dft_t,
                c.poly_count() as u64,
                a.as_ptr() as *const vec_znx_dft_t,
                a.poly_count() as u64,
                b.as_ptr() as *const vmp::vmp_pmat_t,
                b.rows() as u64,
                (b.size() * b.cols()) as u64,
                tmp_bytes.as_mut_ptr(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ScratchOwned;
    use crate::mat_znx_dft_ops::*;
    use crate::vec_znx_big_ops::*;
    use crate::vec_znx_dft_ops::*;
    use crate::vec_znx_ops::*;
    use crate::{
        FFT64, MatZnxDft, MatZnxDftOps, Module, Sampling, VecZnx, VecZnxBig, VecZnxBigOps, VecZnxDft, VecZnxDftOps, alloc_aligned,
    };
    use sampling::source::Source;

    #[test]
    fn vmp_prepare_row_dft() {
        let module: Module<FFT64> = Module::<FFT64>::new(16);
        let log_base2k: usize = 8;
        let mat_rows: usize = 4;
        let mat_cols_in: usize = 2;
        let mat_cols_out: usize = 2;
        let mat_size: usize = 5;
        let mut a: VecZnx<_> = module.new_vec_znx(mat_cols_out, mat_size);
        let mut b: VecZnx<_> = module.new_vec_znx(mat_cols_out, mat_size);
        let mut a_dft: VecZnxDft<_, FFT64> = module.new_vec_znx_dft(mat_cols_out, mat_size);
        let mut a_big: VecZnxBig<_, FFT64> = module.new_vec_znx_big(mat_cols_out, mat_size);
        let mut b_dft: VecZnxDft<_, FFT64> = module.new_vec_znx_dft(mat_cols_out, mat_size);
        let mut vmpmat_0: MatZnxDft<_, FFT64> = module.new_mat_znx_dft(mat_rows, mat_cols_in, mat_cols_out, mat_size);
        let mut vmpmat_1: MatZnxDft<_, FFT64> = module.new_mat_znx_dft(mat_rows, mat_cols_in, mat_cols_out, mat_size);

        // let mut tmp_bytes: Vec<u8> =
        // alloc_aligned(module.vmp_prepare_row_tmp_bytes(mat_cols_out, mat_size) | module.vec_znx_big_normalize_tmp_bytes());
        let mut scratch = ScratchOwned::new(
            2 * (module.vmp_prepare_row_tmp_bytes(mat_cols_out, mat_size) + module.vec_znx_big_normalize_tmp_bytes()),
        );
        let mut tmp_bytes: Vec<u8> =
            alloc_aligned::<u8>(<Module<FFT64> as VecZnxDftOps<Vec<u8>, Vec<u8>, _>>::vec_znx_idft_tmp_bytes(&module));

        for col_in in 0..mat_cols_in {
            for row_i in 0..mat_rows {
                let mut source: Source = Source::new([0u8; 32]);

                (0..mat_cols_out).for_each(|col_out| {
                    module.fill_uniform(log_base2k, &mut a, col_out, mat_size, &mut source);
                    module.vec_znx_dft(&mut a_dft, col_out, &a, col_out);
                });

                // let g = vmpmat_0.to_mut();

                module.vmp_prepare_row(&mut vmpmat_0.to_mut(), row_i, col_in, &a, scratch.borrow());

                // Checks that prepare(mat_znx_dft, a) = prepare_dft(mat_znx_dft, a_dft)
                module.vmp_prepare_row_dft(&mut vmpmat_1, row_i, col_in, &a_dft);
                assert_eq!(vmpmat_0.raw(), vmpmat_1.raw());

                // Checks that a_dft = extract_dft(prepare(mat_znx_dft, a), b_dft)
                module.vmp_extract_row_dft(&mut b_dft, &vmpmat_0, row_i, col_in);
                assert_eq!(a_dft.raw(), b_dft.raw());

                // Checks that a_big = extract(prepare_dft(mat_znx_dft, a_dft), b_big)
                // module.vmp_extract_row(
                //     log_base2k,
                //     &mut b.to_mut(),
                //     &vmpmat_0.to_ref(),
                //     row_i,
                //     col_in,
                //     scratch.borrow(),
                // );

                (0..mat_cols_out).for_each(|col_out| {
                    module.vec_znx_idft(&mut a_big, col_out, &a_dft, col_out, &mut tmp_bytes);
                    module.vec_znx_big_normalize(
                        log_base2k,
                        &mut a.to_mut(),
                        col_out,
                        &a_big.to_ref(),
                        col_out,
                        scratch.borrow(),
                    );
                });

                assert_eq!(a.raw(), b.raw());
            }
        }

        module.free();
    }
}
