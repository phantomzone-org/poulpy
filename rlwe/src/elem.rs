use base2k::{
    Backend, DataView, DataViewMut, MatZnxDft, MatZnxDftAlloc, MatZnxDftToMut, MatZnxDftToRef, Module, VecZnx, VecZnxAlloc,
    VecZnxDft, VecZnxDftAlloc, VecZnxDftToMut, VecZnxDftToRef, VecZnxToMut, VecZnxToRef, ZnxInfos,
};

pub trait Infos<T>
where
    T: ZnxInfos,
{
    fn inner(&self) -> &T;

    /// Returns the ring degree of the polynomials.
    fn n(&self) -> usize {
        self.inner().n()
    }

    /// Returns the base two logarithm of the ring dimension of the polynomials.
    fn log_n(&self) -> usize {
        self.inner().log_n()
    }

    /// Returns the number of rows.
    fn rows(&self) -> usize {
        self.inner().rows()
    }

    /// Returns the number of polynomials in each row.
    fn cols(&self) -> usize {
        self.inner().cols()
    }

    /// Returns the number of size per polynomial.
    fn size(&self) -> usize {
        let size: usize = self.inner().size();
        debug_assert_eq!(size, derive_size(self.log_base2k(), self.log_q()));
        size
    }

    /// Returns the total number of small polynomials.
    fn poly_count(&self) -> usize {
        self.rows() * self.cols() * self.size()
    }

    /// Returns the base 2 logarithm of the ciphertext base.
    fn log_base2k(&self) -> usize;

    /// Returns the base 2 logarithm of the ciphertext modulus.
    fn log_q(&self) -> usize;
}

pub struct Ciphertext<T> {
    data: T,
    log_base2k: usize,
    log_q: usize,
}

impl<T> Infos<T> for Ciphertext<T>
where
    T: ZnxInfos,
{
    fn inner(&self) -> &T {
        &self.data
    }

    fn log_base2k(&self) -> usize {
        self.log_base2k
    }

    fn log_q(&self) -> usize {
        self.log_q
    }
}

impl<D> DataView for Ciphertext<D> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D> DataViewMut for Ciphertext<D> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

pub struct Plaintext<T> {
    data: T,
    log_base2k: usize,
    log_q: usize,
}

impl<T> Infos<T> for Plaintext<T>
where
    T: ZnxInfos,
{
    fn inner(&self) -> &T {
        &self.data
    }

    fn log_base2k(&self) -> usize {
        self.log_base2k
    }

    fn log_q(&self) -> usize {
        self.log_q
    }
}

impl<T> Plaintext<T> {
    pub fn data(&self) -> &T {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

pub(crate) type CtVecZnx<C> = Ciphertext<VecZnx<C>>;
pub(crate) type CtVecZnxDft<C, B: Backend> = Ciphertext<VecZnxDft<C, B>>;
pub(crate) type CtMatZnxDft<C, B: Backend> = Ciphertext<MatZnxDft<C, B>>;
pub(crate) type PtVecZnx<C> = Plaintext<VecZnx<C>>;
pub(crate) type PtVecZnxDft<C, B: Backend> = Plaintext<VecZnxDft<C, B>>;
pub(crate) type PtMatZnxDft<C, B: Backend> = Plaintext<MatZnxDft<C, B>>;

impl<D> VecZnxToMut for Ciphertext<D>
where
    D: VecZnxToMut,
{
    fn to_mut(&mut self) -> VecZnx<&mut [u8]> {
        self.data_mut().to_mut()
    }
}

impl<D> VecZnxToRef for Ciphertext<D>
where
    D: VecZnxToRef,
{
    fn to_ref(&self) -> VecZnx<&[u8]> {
        self.data().to_ref()
    }
}

impl Ciphertext<VecZnx<Vec<u8>>> {
    pub fn new<B: Backend>(module: &Module<B>, log_base2k: usize, log_q: usize, cols: usize) -> Self {
        Self {
            data: module.new_vec_znx(cols, derive_size(log_base2k, log_q)),
            log_base2k: log_base2k,
            log_q: log_q,
        }
    }
}

impl<D> VecZnxToMut for Plaintext<D>
where
    D: VecZnxToMut,
{
    fn to_mut(&mut self) -> VecZnx<&mut [u8]> {
        self.data_mut().to_mut()
    }
}

impl<D> VecZnxToRef for Plaintext<D>
where
    D: VecZnxToRef,
{
    fn to_ref(&self) -> VecZnx<&[u8]> {
        self.data().to_ref()
    }
}

impl Plaintext<VecZnx<Vec<u8>>> {
    pub fn new<B: Backend>(module: &Module<B>, log_base2k: usize, log_q: usize) -> Self {
        Self {
            data: module.new_vec_znx(1, derive_size(log_base2k, log_q)),
            log_base2k: log_base2k,
            log_q: log_q,
        }
    }
}

impl<D, B: Backend> VecZnxDftToMut<B> for Ciphertext<D>
where
    D: VecZnxDftToMut<B>,
{
    fn to_mut(&mut self) -> VecZnxDft<&mut [u8], B> {
        self.data_mut().to_mut()
    }
}

impl<D, B: Backend> VecZnxDftToRef<B> for Ciphertext<D>
where
    D: VecZnxDftToRef<B>,
{
    fn to_ref(&self) -> VecZnxDft<&[u8], B> {
        self.data().to_ref()
    }
}

impl<B: Backend> Ciphertext<VecZnxDft<Vec<u8>, B>> {
    pub fn new(module: &Module<B>, log_base2k: usize, log_q: usize, cols: usize) -> Self {
        Self {
            data: module.new_vec_znx_dft(cols, derive_size(log_base2k, log_q)),
            log_base2k: log_base2k,
            log_q: log_q,
        }
    }
}

impl<D, B: Backend> MatZnxDftToMut<B> for Ciphertext<D>
where
    D: MatZnxDftToMut<B>,
{
    fn to_mut(&mut self) -> MatZnxDft<&mut [u8], B> {
        self.data_mut().to_mut()
    }
}

impl<D, B: Backend> MatZnxDftToRef<B> for Ciphertext<D>
where
    D: MatZnxDftToRef<B>,
{
    fn to_ref(&self) -> MatZnxDft<&[u8], B> {
        self.data().to_ref()
    }
}

impl<B: Backend> Ciphertext<MatZnxDft<Vec<u8>, B>> {
    pub fn new(module: &Module<B>, log_base2k: usize, rows: usize, cols_in: usize, cols_out: usize, log_q: usize) -> Self {
        Self {
            data: module.new_mat_znx_dft(rows, cols_in, cols_out, derive_size(log_base2k, log_q)),
            log_base2k: log_base2k,
            log_q: log_q,
        }
    }
}

pub(crate) fn derive_size(log_base2k: usize, log_q: usize) -> usize {
    (log_q + log_base2k - 1) / log_base2k
}
