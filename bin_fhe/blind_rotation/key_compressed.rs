use backend::hal::{
    api::{FillUniform, Reset, ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxAddScalarInplace, ZnxView, ZnxViewMut},
    layouts::{Backend, Data, DataMut, DataRef, Module, ReaderFrom, ScalarZnx, ScalarZnxToRef, Scratch, WriterTo},
};
use sampling::source::Source;

use crate::{
    Distribution, Infos,
    layouts::{LWESecret, compressed::GGSWCiphertextCompressed, prepared::GLWESecretExec},
};
use std::fmt;

use crate::trait_families::GGSWEncryptSkFamily;

#[derive(Clone)]
pub struct BlindRotationKeyCGGICompressed<D: Data> {
    pub(crate) keys: Vec<GGSWCiphertextCompressed<D>>,
    pub(crate) dist: Distribution,
}

impl<D: DataRef> fmt::Debug for BlindRotationKeyCGGICompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: Data> PartialEq for BlindRotationKeyCGGICompressed<D> {
    fn eq(&self, other: &Self) -> bool {
        if self.keys.len() != other.keys.len() {
            return false;
        }
        for (a, b) in self.keys.iter().zip(other.keys.iter()) {
            if a != b {
                return false;
            }
        }
        self.dist == other.dist
    }
}

impl<D: Data> Eq for BlindRotationKeyCGGICompressed<D> {}

impl<D: DataRef> fmt::Display for BlindRotationKeyCGGICompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, key) in self.keys.iter().enumerate() {
            write!(f, "key[{}]: {}", i, key)?;
        }
        writeln!(f, "{:?}", self.dist)
    }
}

impl<D: DataMut> Reset for BlindRotationKeyCGGICompressed<D> {
    fn reset(&mut self) {
        self.keys.iter_mut().for_each(|key| key.reset());
        self.dist = Distribution::NONE;
    }
}

impl<D: DataMut> FillUniform for BlindRotationKeyCGGICompressed<D> {
    fn fill_uniform(&mut self, source: &mut sampling::source::Source) {
        self.keys
            .iter_mut()
            .for_each(|key| key.fill_uniform(source));
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: DataMut> ReaderFrom for BlindRotationKeyCGGICompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        match Distribution::read_from(reader) {
            Ok(dist) => self.dist = dist,
            Err(e) => return Err(e),
        }
        let len: usize = reader.read_u64::<LittleEndian>()? as usize;
        if self.keys.len() != len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("self.keys.len()={} != read len={}", self.keys.len(), len),
            ));
        }
        for key in &mut self.keys {
            key.read_from(reader)?;
        }
        Ok(())
    }
}

impl<D: DataRef> WriterTo for BlindRotationKeyCGGICompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        match self.dist.write_to(writer) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        writer.write_u64::<LittleEndian>(self.keys.len() as u64)?;
        for key in &self.keys {
            key.write_to(writer)?;
        }
        Ok(())
    }
}

impl BlindRotationKeyCGGICompressed<Vec<u8>> {
    pub fn alloc(n_gglwe: usize, n_lwe: usize, basek: usize, k: usize, rows: usize, rank: usize) -> Self {
        let mut data: Vec<GGSWCiphertextCompressed<Vec<u8>>> = Vec::with_capacity(n_lwe);
        (0..n_lwe).for_each(|_| {
            data.push(GGSWCiphertextCompressed::alloc(
                n_gglwe, basek, k, rows, 1, rank,
            ))
        });
        Self {
            keys: data,
            dist: Distribution::NONE,
        }
    }

    pub fn generate_from_sk_scratch_space<B: Backend>(module: &Module<B>, n: usize, basek: usize, k: usize, rank: usize) -> usize
    where
        Module<B>: GGSWEncryptSkFamily<B>,
    {
        GGSWCiphertextCompressed::encrypt_sk_scratch_space(module, n, basek, k, rank)
    }
}

impl<D: DataRef> BlindRotationKeyCGGICompressed<D> {
    #[allow(dead_code)]
    pub(crate) fn n(&self) -> usize {
        self.keys[0].n()
    }

    #[allow(dead_code)]
    pub(crate) fn rows(&self) -> usize {
        self.keys[0].rows()
    }

    #[allow(dead_code)]
    pub(crate) fn k(&self) -> usize {
        self.keys[0].k()
    }

    #[allow(dead_code)]
    pub(crate) fn size(&self) -> usize {
        self.keys[0].size()
    }

    #[allow(dead_code)]
    pub(crate) fn rank(&self) -> usize {
        self.keys[0].rank()
    }

    #[allow(dead_code)]
    pub(crate) fn basek(&self) -> usize {
        self.keys[0].basek()
    }

    #[allow(dead_code)]
    pub(crate) fn block_size(&self) -> usize {
        match self.dist {
            Distribution::BinaryBlock(value) => value,
            _ => 1,
        }
    }
}

impl<D: DataMut> BlindRotationKeyCGGICompressed<D> {
    pub fn generate_from_sk<DataSkGLWE, DataSkLWE, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_glwe: &GLWESecretExec<DataSkGLWE, B>,
        sk_lwe: &LWESecret<DataSkLWE>,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        DataSkGLWE: DataRef,
        DataSkLWE: DataRef,
        Module<B>: GGSWEncryptSkFamily<B> + VecZnxAddScalarInplace,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.keys.len(), sk_lwe.n());
            assert!(sk_glwe.n() <= module.n());
            assert_eq!(sk_glwe.rank(), self.keys[0].rank());
            match sk_lwe.dist {
                Distribution::BinaryBlock(_)
                | Distribution::BinaryFixed(_)
                | Distribution::BinaryProb(_)
                | Distribution::ZERO => {}
                _ => panic!(
                    "invalid GLWESecret distribution: must be BinaryBlock, BinaryFixed or BinaryProb (or ZERO for debugging)"
                ),
            }
        }

        self.dist = sk_lwe.dist;

        let mut pt: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(sk_glwe.n(), 1);
        let sk_ref: ScalarZnx<&[u8]> = sk_lwe.data.to_ref();

        let mut source_xa: Source = Source::new(seed_xa);

        self.keys.iter_mut().enumerate().for_each(|(i, ggsw)| {
            pt.at_mut(0, 0)[0] = sk_ref.at(0, 0)[i];
            ggsw.encrypt_sk(
                module,
                &pt,
                sk_glwe,
                source_xa.new_seed(),
                source_xe,
                sigma,
                scratch,
            );
        });
    }
}
