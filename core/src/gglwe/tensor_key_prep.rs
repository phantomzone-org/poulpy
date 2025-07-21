use backend::{Backend, Module, VmpPMat, VmpPMatAlloc};

use crate::{GLWESwitchingKey, GLWESwitchingKeyPrep, Infos};

pub struct GLWETensorKeyPrep<D, B: Backend> {
    pub(crate) keys: Vec<GLWESwitchingKeyPrep<D, B>>,
}

impl<B: Backend> GLWETensorKeyPrep<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self where Module<B>: VmpPMatAlloc<B>{
        let mut keys: Vec<GLWESwitchingKeyPrep<Vec<u8>, B>> = Vec::new();
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);
        (0..pairs).for_each(|_| {
            keys.push(GLWESwitchingKeyPrep::alloc(
                module, basek, k, rows, digits, 1, rank,
            ));
        });
        Self { keys }
    }

    pub fn bytes_of(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize {
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);
        pairs * GLWESwitchingKey::<Vec<u8>>::bytes_of(module, basek, k, rows, digits, 1, rank)
    }
}

impl<D, B: Backend> Infos for GLWETensorKeyPrep<D, B> {
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

impl<D, B: Backend> GLWETensorKeyPrep<D, B> {
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

impl<D: AsMut<[u8]> + AsRef<[u8]>, B: Backend> GLWETensorKeyPrep<D, B> {
    // Returns a mutable reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at_mut(&mut self, mut i: usize, mut j: usize) -> &mut GLWESwitchingKeyPrep<D, B> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank();
        &mut self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<D: AsRef<[u8]>, B: Backend> GLWETensorKeyPrep<D, B> {
    // Returns a reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at(&self, mut i: usize, mut j: usize) -> &GLWESwitchingKeyPrep<D, B> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank();
        &self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}
