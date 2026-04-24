use std::fmt;

use poulpy_hal::layouts::{
    Backend, Data, DataMut, DataRef, Module, TransferFrom, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef, VecZnxToMut,
    VecZnxToRef, ZnxInfos,
};

use crate::api::ModuleTransfer;
use crate::layouts::{
    Base2K, Degree, GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GLWEToMut, GLWEToRef, LWEInfos, Rank, SetLWEInfos,
    TorusPrecision,
};

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWEPlaintextLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
}

impl LWEInfos for GLWEPlaintextLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        self.n
    }

    fn size(&self) -> usize {
        self.k.as_usize().div_ceil(self.base2k.as_usize())
    }
}

impl GLWEInfos for GLWEPlaintextLayout {
    fn rank(&self) -> Rank {
        Rank(0)
    }
}

pub struct GLWEPlaintext<D: Data> {
    pub data: VecZnx<D>,
    pub base2k: Base2K,
}

pub type GLWEPlaintextBackendRef<'a, BE> = GLWEPlaintext<<BE as Backend>::BufRef<'a>>;
pub type GLWEPlaintextBackendMut<'a, BE> = GLWEPlaintext<<BE as Backend>::BufMut<'a>>;

impl<D: DataMut> SetLWEInfos for GLWEPlaintext<D> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }
}

impl<D: Data> LWEInfos for GLWEPlaintext<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn size(&self) -> usize {
        self.data.size()
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }
}

impl<D: Data> GLWEInfos for GLWEPlaintext<D> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32 - 1)
    }
}

impl<D: DataRef> GLWEPlaintext<D> {
    /// Copies this plaintext's backing bytes into an owned buffer of
    /// backend `To`, routing via host bytes.
    pub fn to_backend<BE, To>(&self, dst: &Module<To>) -> GLWEPlaintext<To::OwnedBuf>
    where
        BE: Backend<OwnedBuf = D>,
        To: Backend,
        To: TransferFrom<BE>,
    {
        dst.upload_glwe_plaintext(self)
    }
}

impl<D: Data> GLWEPlaintext<D> {
    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> GLWEPlaintext<To::OwnedBuf>
    where
        To: Backend<OwnedBuf = D>,
    {
        GLWEPlaintext {
            data: VecZnx::from_data_with_max_size(
                self.data.data,
                self.data.n,
                self.data.cols,
                self.data.size,
                self.data.max_size,
            ),
            base2k: self.base2k,
        }
    }
}

impl<D: DataRef> fmt::Display for GLWEPlaintext<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GLWEPlaintext: base2k={} k={}: {}",
            self.base2k().0,
            self.max_k().0,
            self.data
        )
    }
}

impl GLWEPlaintext<Vec<u8>> {
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.max_k())
    }

    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision) -> Self {
        GLWEPlaintext {
            data: VecZnx::alloc(n.into(), 1, k.0.div_ceil(base2k.0) as usize),
            base2k,
        }
    }
}

impl GLWEPlaintext<Vec<u8>> {
    pub fn alloc_with_meta(n: Degree, base2k: Base2K, k: TorusPrecision) -> Self {
        GLWEPlaintext {
            data: VecZnx::alloc(n.into(), 1, k.0.div_ceil(base2k.0) as usize),
            base2k,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), infos.base2k(), infos.max_k())
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision) -> usize {
        VecZnx::bytes_of(n.into(), 1, k.0.div_ceil(base2k.0) as usize)
    }
}

impl<D: DataRef> GLWEToRef for GLWEPlaintext<D> {
    fn to_ref(&self) -> GLWE<&[u8]> {
        GLWE {
            base2k: self.base2k,
            data: self.data.to_ref(),
        }
    }
}

impl<BE: Backend> GLWEToBackendRef<BE> for GLWEPlaintext<BE::OwnedBuf> {
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>> {
        GLWE {
            base2k: self.base2k,
            data: <VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(&self.data),
        }
    }
}

impl<D: DataMut> GLWEToMut for GLWEPlaintext<D> {
    fn to_mut(&mut self) -> GLWE<&mut [u8]> {
        GLWE {
            base2k: self.base2k,
            data: self.data.to_mut(),
        }
    }
}

impl<BE: Backend> GLWEToBackendMut<BE> for GLWEPlaintext<BE::OwnedBuf> {
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>> {
        GLWE {
            base2k: self.base2k,
            data: <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut self.data),
        }
    }
}

pub trait GLWEPlaintextToRef {
    fn to_ref(&self) -> GLWEPlaintext<&[u8]>;
}

pub trait GLWEPlaintextToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> GLWEPlaintextBackendRef<'_, BE>;
}

impl<BE: Backend> GLWEPlaintextToBackendRef<BE> for GLWEPlaintext<BE::OwnedBuf> {
    fn to_backend_ref(&self) -> GLWEPlaintextBackendRef<'_, BE> {
        GLWEPlaintext {
            data: <VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(&self.data),
            base2k: self.base2k,
        }
    }
}

impl<D: DataRef> GLWEPlaintextToRef for GLWEPlaintext<D> {
    fn to_ref(&self) -> GLWEPlaintext<&[u8]> {
        GLWEPlaintext {
            data: self.data.to_ref(),
            base2k: self.base2k,
        }
    }
}

pub trait GLWEPlaintextToMut {
    fn to_mut(&mut self) -> GLWEPlaintext<&mut [u8]>;
}

pub trait GLWEPlaintextToBackendMut<BE: Backend>: GLWEPlaintextToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> GLWEPlaintextBackendMut<'_, BE>;
}

impl<BE: Backend> GLWEPlaintextToBackendMut<BE> for GLWEPlaintext<BE::OwnedBuf> {
    fn to_backend_mut(&mut self) -> GLWEPlaintextBackendMut<'_, BE> {
        GLWEPlaintext {
            base2k: self.base2k,
            data: <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut self.data),
        }
    }
}

impl<D: DataMut> GLWEPlaintextToMut for GLWEPlaintext<D> {
    fn to_mut(&mut self) -> GLWEPlaintext<&mut [u8]> {
        GLWEPlaintext {
            base2k: self.base2k,
            data: self.data.to_mut(),
        }
    }
}

impl<D: DataMut> GLWEPlaintext<D> {
    pub fn data_mut(&mut self) -> &mut VecZnx<D> {
        &mut self.data
    }
}

impl<D: DataRef> GLWEPlaintext<D> {
    pub fn data(&self) -> &VecZnx<D> {
        &self.data
    }
}
