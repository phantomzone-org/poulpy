use backend::{Backend, FFT64, Module, VecZnx, VecZnxAlloc, VecZnxDftOps, VecZnxToMut, VecZnxToRef};

use crate::{FourierGLWECiphertext, GLWEOps, Infos, SetMetaData};

pub struct GLWECiphertext<C> {
    pub data: VecZnx<C>,
    pub basek: usize,
    pub k: usize,
}

impl GLWECiphertext<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank: usize) -> Self {
        Self {
            data: module.new_vec_znx(rank + 1, k.div_ceil(basek)),
            basek,
            k,
        }
    }

    pub fn bytes_of(module: &Module<FFT64>, basek: usize, k: usize, rank: usize) -> usize {
        module.bytes_of_vec_znx(rank + 1, k.div_ceil(basek))
    }
}

impl<T> Infos for GLWECiphertext<T> {
    type Inner = VecZnx<T>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.basek
    }

    fn k(&self) -> usize {
        self.k
    }
}

impl<T> GLWECiphertext<T> {
    pub fn rank(&self) -> usize {
        self.cols() - 1
    }
}

impl<C: AsRef<[u8]>> GLWECiphertext<C> {
    #[allow(dead_code)]
    pub(crate) fn dft<R: AsMut<[u8]> + AsRef<[u8]>>(&self, module: &Module<FFT64>, res: &mut FourierGLWECiphertext<R, FFT64>) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), res.rank());
            assert_eq!(self.basek(), res.basek())
        }

        (0..self.rank() + 1).for_each(|i| {
            module.vec_znx_dft(1, 0, &mut res.data, i, &self.data, i);
        })
    }
}

impl<DataSelf: AsRef<[u8]>> GLWECiphertext<DataSelf> {
    pub fn clone(&self) -> GLWECiphertext<Vec<u8>> {
        GLWECiphertext {
            data: self.data.clone(),
            basek: self.basek(),
            k: self.k(),
        }
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> SetMetaData for GLWECiphertext<DataSelf> {
    fn set_k(&mut self, k: usize) {
        self.k = k
    }

    fn set_basek(&mut self, basek: usize) {
        self.basek = basek
    }
}

pub trait GLWECiphertextToRef: Infos {
    fn to_ref(&self) -> GLWECiphertext<&[u8]>;
}

impl<D: AsRef<[u8]>> GLWECiphertextToRef for GLWECiphertext<D> {
    fn to_ref(&self) -> GLWECiphertext<&[u8]> {
        GLWECiphertext {
            data: self.data.to_ref(),
            basek: self.basek,
            k: self.k,
        }
    }
}

pub trait GLWECiphertextToMut: Infos {
    fn to_mut(&mut self) -> GLWECiphertext<&mut [u8]>;
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> GLWECiphertextToMut for GLWECiphertext<D> {
    fn to_mut(&mut self) -> GLWECiphertext<&mut [u8]> {
        GLWECiphertext {
            data: self.data.to_mut(),
            basek: self.basek,
            k: self.k,
        }
    }
}

impl<D> GLWEOps for GLWECiphertext<D>
where
    D: AsRef<[u8]> + AsMut<[u8]>,
    GLWECiphertext<D>: GLWECiphertextToMut + Infos + SetMetaData,
{
}
