use base2k::{
    Backend, Module, VecZnx, VecZnxAlloc, VecZnxDft, VecZnxDftAlloc, VecZnxDftToMut, VecZnxDftToRef, VecZnxToMut, VecZnxToRef,
    ZnxInfos,
};

pub trait Infos {
    type Inner: ZnxInfos;

    fn inner(&self) -> &Self::Inner;

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
        debug_assert_eq!(size, derive_size(self.log_base2k(), self.log_k()));
        size
    }

    /// Returns the total number of small polynomials.
    fn poly_count(&self) -> usize {
        self.rows() * self.cols() * self.size()
    }

    /// Returns the base 2 logarithm of the ciphertext base.
    fn log_base2k(&self) -> usize;

    /// Returns the bit precision of the ciphertext.
    fn log_k(&self) -> usize;
}

pub struct RLWECt<C> {
    pub data: VecZnx<C>,
    pub log_base2k: usize,
    pub log_k: usize,
}

impl<T> Infos for RLWECt<T> {
    type Inner = VecZnx<T>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn log_base2k(&self) -> usize {
        self.log_base2k
    }

    fn log_k(&self) -> usize {
        self.log_k
    }
}

impl<C> VecZnxToMut for RLWECt<C>
where
    VecZnx<C>: VecZnxToMut,
{
    fn to_mut(&mut self) -> VecZnx<&mut [u8]> {
        self.data.to_mut()
    }
}

impl<C> VecZnxToRef for RLWECt<C>
where
    VecZnx<C>: VecZnxToRef,
{
    fn to_ref(&self) -> VecZnx<&[u8]> {
        self.data.to_ref()
    }
}

pub struct RLWEPt<C> {
    pub data: VecZnx<C>,
    pub log_base2k: usize,
    pub log_k: usize,
}

impl<T> Infos for RLWEPt<T> {
    type Inner = VecZnx<T>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn log_base2k(&self) -> usize {
        self.log_base2k
    }

    fn log_k(&self) -> usize {
        self.log_k
    }
}

impl<C> VecZnxToMut for RLWEPt<C>
where
    VecZnx<C>: VecZnxToMut,
{
    fn to_mut(&mut self) -> VecZnx<&mut [u8]> {
        self.data.to_mut()
    }
}

impl<C> VecZnxToRef for RLWEPt<C>
where
    VecZnx<C>: VecZnxToRef,
{
    fn to_ref(&self) -> VecZnx<&[u8]> {
        self.data.to_ref()
    }
}

impl RLWECt<Vec<u8>> {
    pub fn new<B: Backend>(module: &Module<B>, log_base2k: usize, log_k: usize, cols: usize) -> Self {
        Self {
            data: module.new_vec_znx(cols, derive_size(log_base2k, log_k)),
            log_base2k: log_base2k,
            log_k: log_k,
        }
    }
}

impl RLWEPt<Vec<u8>> {
    pub fn new<B: Backend>(module: &Module<B>, log_base2k: usize, log_k: usize) -> Self {
        Self {
            data: module.new_vec_znx(1, derive_size(log_base2k, log_k)),
            log_base2k: log_base2k,
            log_k: log_k,
        }
    }
}

pub struct RLWECtDft<C, B: Backend> {
    pub data: VecZnxDft<C, B>,
    pub log_base2k: usize,
    pub log_k: usize,
}

impl<B: Backend> RLWECtDft<Vec<u8>, B> {
    pub fn new(module: &Module<B>, log_base2k: usize, log_k: usize) -> Self {
        Self {
            data: module.new_vec_znx_dft(1, derive_size(log_base2k, log_k)),
            log_base2k: log_base2k,
            log_k: log_k,
        }
    }
}

impl<T, B: Backend> Infos for RLWECtDft<T, B> {
    type Inner = VecZnxDft<T, B>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn log_base2k(&self) -> usize {
        self.log_base2k
    }

    fn log_k(&self) -> usize {
        self.log_k
    }
}

impl<C, B: Backend> VecZnxDftToMut<B> for RLWECtDft<C, B>
where
    VecZnxDft<C, B>: VecZnxDftToMut<B>,
{
    fn to_mut(&mut self) -> VecZnxDft<&mut [u8], B> {
        self.data.to_mut()
    }
}

impl<C, B: Backend> VecZnxDftToRef<B> for RLWECtDft<C, B>
where
    VecZnxDft<C, B>: VecZnxDftToRef<B>,
{
    fn to_ref(&self) -> VecZnxDft<&[u8], B> {
        self.data.to_ref()
    }
}

pub(crate) fn derive_size(log_base2k: usize, log_k: usize) -> usize {
    (log_k + log_base2k - 1) / log_base2k
}
