use backend::hal::{
    api::ZnxInfos,
    layouts::{ReaderFrom, VecZnx, VecZnxToMut, VecZnxToRef, WriterTo},
};

use crate::{Infos, SetMetaData};

pub struct LWECiphertext<D> {
    pub(crate) data: VecZnx<D>,
    pub(crate) k: usize,
    pub(crate) basek: usize,
}

impl LWECiphertext<Vec<u8>> {
    pub fn alloc(n: usize, basek: usize, k: usize) -> Self {
        Self {
            data: VecZnx::new::<i64>(n + 1, 1, k.div_ceil(basek)),
            k: k,
            basek: basek,
        }
    }
}

impl<T> Infos for LWECiphertext<T>
where
    VecZnx<T>: ZnxInfos,
{
    type Inner = VecZnx<T>;

    fn n(&self) -> usize {
        &self.inner().n() - 1
    }

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

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> SetMetaData for LWECiphertext<DataSelf> {
    fn set_k(&mut self, k: usize) {
        self.k = k
    }

    fn set_basek(&mut self, basek: usize) {
        self.basek = basek
    }
}

pub trait LWECiphertextToRef {
    fn to_ref(&self) -> LWECiphertext<&[u8]>;
}

impl<D: AsRef<[u8]>> LWECiphertextToRef for LWECiphertext<D> {
    fn to_ref(&self) -> LWECiphertext<&[u8]> {
        LWECiphertext {
            data: self.data.to_ref(),
            basek: self.basek,
            k: self.k,
        }
    }
}

pub trait LWECiphertextToMut {
    #[allow(dead_code)]
    fn to_mut(&mut self) -> LWECiphertext<&mut [u8]>;
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> LWECiphertextToMut for LWECiphertext<D> {
    fn to_mut(&mut self) -> LWECiphertext<&mut [u8]> {
        LWECiphertext {
            data: self.data.to_mut(),
            basek: self.basek,
            k: self.k,
        }
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: AsRef<[u8]> + AsMut<[u8]>> ReaderFrom for LWECiphertext<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = reader.read_u64::<LittleEndian>()? as usize;
        self.basek = reader.read_u64::<LittleEndian>()? as usize;
        self.data.read_from(reader)
    }
}

impl<D: AsRef<[u8]>> WriterTo for LWECiphertext<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.k as u64)?;
        writer.write_u64::<LittleEndian>(self.basek as u64)?;
        self.data.write_to(writer)
    }
}
