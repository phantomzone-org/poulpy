use backend::hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPMatPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VmpPMat},
};

use crate::{AutomorphismKey, GGLWECiphertext, GGLWEExecLayoutFamily, GLWESwitchingKey, GLWETensorKey, Infos};

#[derive(PartialEq, Eq)]
pub struct GGLWECiphertextExec<D: Data, B: Backend> {
    pub(crate) data: VmpPMat<D, B>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) digits: usize,
}

impl<B: Backend> GGLWECiphertextExec<Vec<u8>, B> {
    pub fn alloc(
        module: &Module<B>,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        let size: usize = k.div_ceil(basek);
        debug_assert!(
            size > digits,
            "invalid gglwe: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid gglwe: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
        );

        Self {
            data: module.vmp_pmat_alloc(n, rows, rank_in, rank_out + 1, size),
            basek: basek,
            k,
            digits,
        }
    }

    pub fn bytes_of(
        module: &Module<B>,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        let size: usize = k.div_ceil(basek);
        debug_assert!(
            size > digits,
            "invalid gglwe: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid gglwe: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
        );

        module.vmp_pmat_alloc_bytes(n, rows, rank_in, rank_out + 1, rows)
    }
}

impl<D: Data, B: Backend> Infos for GGLWECiphertextExec<D, B> {
    type Inner = VmpPMat<D, B>;

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

impl<D: Data, B: Backend> GGLWECiphertextExec<D, B> {
    pub fn rank(&self) -> usize {
        self.data.cols_out() - 1
    }

    pub fn digits(&self) -> usize {
        self.digits
    }

    pub fn rank_in(&self) -> usize {
        self.data.cols_in()
    }

    pub fn rank_out(&self) -> usize {
        self.data.cols_out() - 1
    }
}

impl<D: DataMut, B: Backend> GGLWECiphertextExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &GGLWECiphertext<DataOther>, scratch: &mut Scratch<B>)
    where
        DataOther: DataRef,
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        module.vmp_prepare(&mut self.data, &other.data, scratch);
        self.basek = other.basek;
        self.k = other.k;
        self.digits = other.digits;
    }
}

#[derive(PartialEq, Eq)]
pub struct GLWESwitchingKeyExec<D: Data, B: Backend> {
    pub(crate) key: GGLWECiphertextExec<D, B>,
    pub(crate) sk_in_n: usize,  // Degree of sk_in
    pub(crate) sk_out_n: usize, // Degree of sk_out
}

impl<B: Backend> GLWESwitchingKeyExec<Vec<u8>, B> {
    pub fn alloc(
        module: &Module<B>,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        GLWESwitchingKeyExec::<Vec<u8>, B> {
            key: GGLWECiphertextExec::alloc(module, n, basek, k, rows, digits, rank_in, rank_out),
            sk_in_n: 0,
            sk_out_n: 0,
        }
    }

    pub fn bytes_of(
        module: &Module<B>,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        GGLWECiphertextExec::bytes_of(module, n, basek, k, rows, digits, rank_in, rank_out)
    }

    pub fn from<DataOther: DataRef>(module: &Module<B>, other: &GLWESwitchingKey<DataOther>, scratch: &mut Scratch<B>) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        let mut ksk_exec: GLWESwitchingKeyExec<Vec<u8>, B> = Self::alloc(
            module,
            other.n(),
            other.basek(),
            other.k(),
            other.rows(),
            other.digits(),
            other.rank_in(),
            other.rank_out(),
        );
        ksk_exec.prepare(module, other, scratch);
        ksk_exec
    }
}

impl<D: Data, B: Backend> Infos for GLWESwitchingKeyExec<D, B> {
    type Inner = VmpPMat<D, B>;

    fn inner(&self) -> &Self::Inner {
        self.key.inner()
    }

    fn basek(&self) -> usize {
        self.key.basek()
    }

    fn k(&self) -> usize {
        self.key.k()
    }
}

impl<D: Data, B: Backend> GLWESwitchingKeyExec<D, B> {
    pub fn rank(&self) -> usize {
        self.key.data.cols_out() - 1
    }

    pub fn rank_in(&self) -> usize {
        self.key.data.cols_in()
    }

    pub fn rank_out(&self) -> usize {
        self.key.data.cols_out() - 1
    }

    pub fn digits(&self) -> usize {
        self.key.digits()
    }

    pub fn sk_degree_in(&self) -> usize {
        self.sk_in_n
    }

    pub fn sk_degree_out(&self) -> usize {
        self.sk_out_n
    }
}

impl<D: DataMut, B: Backend> GLWESwitchingKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &GLWESwitchingKey<DataOther>, scratch: &mut Scratch<B>)
    where
        DataOther: DataRef,
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        self.key.prepare(module, &other.key, scratch);
        self.sk_in_n = other.sk_in_n;
        self.sk_out_n = other.sk_out_n;
    }
}

#[derive(PartialEq, Eq)]
pub struct AutomorphismKeyExec<D: Data, B: Backend> {
    pub(crate) key: GLWESwitchingKeyExec<D, B>,
    pub(crate) p: i64,
}

impl<B: Backend> AutomorphismKeyExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        AutomorphismKeyExec::<Vec<u8>, B> {
            key: GLWESwitchingKeyExec::alloc(module, n, basek, k, rows, digits, rank, rank),
            p: 0,
        }
    }

    pub fn bytes_of(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        GLWESwitchingKeyExec::bytes_of(module, n, basek, k, rows, digits, rank, rank)
    }

    pub fn from<DataOther: DataRef>(module: &Module<B>, other: &AutomorphismKey<DataOther>, scratch: &mut Scratch<B>) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        let mut atk_exec: AutomorphismKeyExec<Vec<u8>, B> = Self::alloc(
            module,
            other.n(),
            other.basek(),
            other.k(),
            other.rows(),
            other.digits(),
            other.rank(),
        );
        atk_exec.prepare(module, other, scratch);
        atk_exec
    }
}

impl<D: DataMut, B: Backend> AutomorphismKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &AutomorphismKey<DataOther>, scratch: &mut Scratch<B>)
    where
        DataOther: DataRef,
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        self.key.prepare(module, &other.key, scratch);
        self.p = other.p;
    }
}

impl<D: Data, B: Backend> Infos for AutomorphismKeyExec<D, B> {
    type Inner = VmpPMat<D, B>;

    fn inner(&self) -> &Self::Inner {
        &self.key.inner()
    }

    fn basek(&self) -> usize {
        self.key.basek()
    }

    fn k(&self) -> usize {
        self.key.k()
    }
}

impl<D: Data, B: Backend> AutomorphismKeyExec<D, B> {
    pub fn p(&self) -> i64 {
        self.p
    }

    pub fn digits(&self) -> usize {
        self.key.digits()
    }

    pub fn rank(&self) -> usize {
        self.key.rank()
    }

    pub fn rank_in(&self) -> usize {
        self.key.rank_in()
    }

    pub fn rank_out(&self) -> usize {
        self.key.rank_out()
    }
}

#[derive(PartialEq, Eq)]
pub struct GLWETensorKeyExec<D: Data, B: Backend> {
    pub(crate) keys: Vec<GLWESwitchingKeyExec<D, B>>,
}

impl<B: Backend> GLWETensorKeyExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        let mut keys: Vec<GLWESwitchingKeyExec<Vec<u8>, B>> = Vec::new();
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);
        (0..pairs).for_each(|_| {
            keys.push(GLWESwitchingKeyExec::alloc(
                module, n, basek, k, rows, digits, 1, rank,
            ));
        });
        Self { keys }
    }

    pub fn bytes_of(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);
        pairs * GLWESwitchingKeyExec::bytes_of(module, n, basek, k, rows, digits, 1, rank)
    }
}

impl<D: Data, B: Backend> Infos for GLWETensorKeyExec<D, B> {
    type Inner = VmpPMat<D, B>;

    fn inner(&self) -> &Self::Inner {
        &self.keys[0].inner()
    }

    fn basek(&self) -> usize {
        self.keys[0].basek()
    }

    fn k(&self) -> usize {
        self.keys[0].k()
    }
}

impl<D: Data, B: Backend> GLWETensorKeyExec<D, B> {
    pub fn rank(&self) -> usize {
        self.keys[0].rank()
    }

    pub fn rank_in(&self) -> usize {
        self.keys[0].rank_in()
    }

    pub fn rank_out(&self) -> usize {
        self.keys[0].rank_out()
    }

    pub fn digits(&self) -> usize {
        self.keys[0].digits()
    }
}

impl<D: DataMut, B: Backend> GLWETensorKeyExec<D, B> {
    // Returns a mutable reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at_mut(&mut self, mut i: usize, mut j: usize) -> &mut GLWESwitchingKeyExec<D, B> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank();
        &mut self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<D: DataRef, B: Backend> GLWETensorKeyExec<D, B> {
    // Returns a reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at(&self, mut i: usize, mut j: usize) -> &GLWESwitchingKeyExec<D, B> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank();
        &self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<D: DataMut, B: Backend> GLWETensorKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &GLWETensorKey<DataOther>, scratch: &mut Scratch<B>)
    where
        DataOther: DataRef,
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.keys.len(), other.keys.len());
        }
        self.keys
            .iter_mut()
            .zip(other.keys.iter())
            .for_each(|(a, b)| {
                a.prepare(module, b, scratch);
            });
    }
}
