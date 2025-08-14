use backend::hal::{
    api::{
        FillUniform, Reset, ScratchAvailable, SvpPPolAlloc, SvpPrepare, TakeVecZnx, TakeVecZnxDft, VecZnxAddScalarInplace,
        VmpPMatAlloc, VmpPMatPrepare, ZnxInfos, ZnxView, ZnxViewMut,
    },
    layouts::{Backend, Data, DataMut, DataRef, Module, ReaderFrom, ScalarZnx, ScalarZnxToRef, Scratch, SvpPPol, WriterTo},
};
use sampling::source::Source;

use std::fmt;

use core::{
    Distribution,
    layouts::{
        GGSWCiphertext, Infos, LWESecret,
        prepared::{GGSWCiphertextExec, GLWESecretExec},
    },
    trait_families::GGSWEncryptSkFamily,
};

#[derive(Clone)]
pub struct BlindRotationKeyCGGI<D: Data> {
    pub(crate) keys: Vec<GGSWCiphertext<D>>,
    pub(crate) dist: Distribution,
}

impl<D: DataRef> fmt::Debug for BlindRotationKeyCGGI<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: Data> PartialEq for BlindRotationKeyCGGI<D> {
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

impl<D: Data> Eq for BlindRotationKeyCGGI<D> {}

impl<D: DataRef> fmt::Display for BlindRotationKeyCGGI<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, key) in self.keys.iter().enumerate() {
            write!(f, "key[{}]: {}", i, key)?;
        }
        writeln!(f, "{:?}", self.dist)
    }
}

impl<D: DataMut> Reset for BlindRotationKeyCGGI<D> {
    fn reset(&mut self) {
        self.keys.iter_mut().for_each(|key| key.reset());
        self.dist = Distribution::NONE;
    }
}

impl<D: DataMut> FillUniform for BlindRotationKeyCGGI<D> {
    fn fill_uniform(&mut self, source: &mut sampling::source::Source) {
        self.keys
            .iter_mut()
            .for_each(|key| key.fill_uniform(source));
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: DataMut> ReaderFrom for BlindRotationKeyCGGI<D> {
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

impl<D: DataRef> WriterTo for BlindRotationKeyCGGI<D> {
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

impl BlindRotationKeyCGGI<Vec<u8>> {
    pub fn alloc(n_gglwe: usize, n_lwe: usize, basek: usize, k: usize, rows: usize, rank: usize) -> Self {
        let mut data: Vec<GGSWCiphertext<Vec<u8>>> = Vec::with_capacity(n_lwe);
        (0..n_lwe).for_each(|_| data.push(GGSWCiphertext::alloc(n_gglwe, basek, k, rows, 1, rank)));
        Self {
            keys: data,
            dist: Distribution::NONE,
        }
    }

    pub fn generate_from_sk_scratch_space<B: Backend>(module: &Module<B>, n: usize, basek: usize, k: usize, rank: usize) -> usize
    where
        Module<B>: GGSWEncryptSkFamily<B>,
    {
        GGSWCiphertext::encrypt_sk_scratch_space(module, n, basek, k, rank)
    }
}

impl<D: DataRef> BlindRotationKeyCGGI<D> {
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

impl<D: DataMut> BlindRotationKeyCGGI<D> {
    pub fn generate_from_sk<DataSkGLWE, DataSkLWE, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_glwe: &GLWESecretExec<DataSkGLWE, B>,
        sk_lwe: &LWESecret<DataSkLWE>,
        source_xa: &mut Source,
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
            match sk_lwe.dist() {
                Distribution::BinaryBlock(_)
                | Distribution::BinaryFixed(_)
                | Distribution::BinaryProb(_)
                | Distribution::ZERO => {}
                _ => panic!(
                    "invalid GLWESecret distribution: must be BinaryBlock, BinaryFixed or BinaryProb (or ZERO for debugging)"
                ),
            }
        }

        self.dist = sk_lwe.dist();

        let mut pt: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(sk_glwe.n(), 1);
        let sk_ref: ScalarZnx<&[u8]> = sk_lwe.data().to_ref();

        self.keys.iter_mut().enumerate().for_each(|(i, ggsw)| {
            pt.at_mut(0, 0)[0] = sk_ref.at(0, 0)[i];
            ggsw.encrypt_sk(module, &pt, sk_glwe, source_xa, source_xe, sigma, scratch);
        });
    }
}

#[derive(PartialEq, Eq)]
pub struct BlindRotationKeyCGGIExec<D: Data, B: Backend> {
    pub(crate) data: Vec<GGSWCiphertextExec<D, B>>,
    pub(crate) dist: Distribution,
    pub(crate) x_pow_a: Option<Vec<SvpPPol<Vec<u8>, B>>>,
}

impl<D: Data, B: Backend> BlindRotationKeyCGGIExec<D, B> {
    #[allow(dead_code)]
    pub(crate) fn n(&self) -> usize {
        self.data[0].n()
    }

    #[allow(dead_code)]
    pub(crate) fn rows(&self) -> usize {
        self.data[0].rows()
    }

    #[allow(dead_code)]
    pub(crate) fn k(&self) -> usize {
        self.data[0].k()
    }

    #[allow(dead_code)]
    pub(crate) fn size(&self) -> usize {
        self.data[0].size()
    }

    #[allow(dead_code)]
    pub(crate) fn rank(&self) -> usize {
        self.data[0].rank()
    }

    pub(crate) fn basek(&self) -> usize {
        self.data[0].basek()
    }

    pub(crate) fn block_size(&self) -> usize {
        match self.dist {
            Distribution::BinaryBlock(value) => value,
            _ => 1,
        }
    }
}

pub trait BlindRotationKeyCGGIExecLayoutFamily<B: Backend> = SvpPPolAlloc<B> + SvpPrepare<B>;

impl<B: Backend> BlindRotationKeyCGGIExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, n_glwe: usize, n_lwe: usize, basek: usize, k: usize, rows: usize, rank: usize) -> Self
    where
        Module<B>: BlindRotationKeyCGGIExecLayoutFamily<B> + VmpPMatAlloc<B> + VmpPMatPrepare<B>,
    {
        let mut data: Vec<GGSWCiphertextExec<Vec<u8>, B>> = Vec::with_capacity(n_lwe);
        (0..n_lwe).for_each(|_| {
            data.push(GGSWCiphertextExec::alloc(
                module, n_glwe, basek, k, rows, 1, rank,
            ))
        });
        Self {
            data,
            dist: Distribution::NONE,
            x_pow_a: None,
        }
    }

    pub fn from<DataOther>(module: &Module<B>, other: &BlindRotationKeyCGGI<DataOther>, scratch: &mut Scratch<B>) -> Self
    where
        DataOther: DataRef,
        Module<B>: BlindRotationKeyCGGIExecLayoutFamily<B> + VmpPMatAlloc<B> + VmpPMatPrepare<B>,
    {
        let mut brk: BlindRotationKeyCGGIExec<Vec<u8>, B> = Self::alloc(
            module,
            other.n(),
            other.keys.len(),
            other.basek(),
            other.k(),
            other.rows(),
            other.rank(),
        );
        brk.prepare(module, other, scratch);
        brk
    }
}

impl<D: DataMut, B: Backend> BlindRotationKeyCGGIExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &BlindRotationKeyCGGI<DataOther>, scratch: &mut Scratch<B>)
    where
        DataOther: DataRef,
        Module<B>: BlindRotationKeyCGGIExecLayoutFamily<B> + VmpPMatAlloc<B> + VmpPMatPrepare<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.data.len(), other.keys.len());
        }

        let n: usize = other.n();

        self.data
            .iter_mut()
            .zip(other.keys.iter())
            .for_each(|(ggsw_exec, other)| {
                ggsw_exec.prepare(module, other, scratch);
            });

        self.dist = other.dist;

        match other.dist {
            Distribution::BinaryBlock(_) => {
                let mut x_pow_a: Vec<SvpPPol<Vec<u8>, B>> = Vec::with_capacity(n << 1);
                let mut buf: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
                (0..n << 1).for_each(|i| {
                    let mut res: SvpPPol<Vec<u8>, B> = module.svp_ppol_alloc(n, 1);
                    set_xai_plus_y(module, i, 0, &mut res, &mut buf);
                    x_pow_a.push(res);
                });
                self.x_pow_a = Some(x_pow_a);
            }
            _ => {}
        }
    }
}

pub fn set_xai_plus_y<A, C, B: Backend>(module: &Module<B>, ai: usize, y: i64, res: &mut SvpPPol<A, B>, buf: &mut ScalarZnx<C>)
where
    A: DataMut,
    C: DataMut,
    Module<B>: SvpPrepare<B>,
{
    let n: usize = res.n();

    {
        let raw: &mut [i64] = buf.at_mut(0, 0);
        if ai < n {
            raw[ai] = 1;
        } else {
            raw[(ai - n) & (n - 1)] = -1;
        }
        raw[0] += y;
    }

    module.svp_prepare(res, 0, buf, 0);

    {
        let raw: &mut [i64] = buf.at_mut(0, 0);

        if ai < n {
            raw[ai] = 0;
        } else {
            raw[(ai - n) & (n - 1)] = 0;
        }
        raw[0] = 0;
    }
}
