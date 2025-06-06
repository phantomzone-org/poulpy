use backend::{Backend, FFT64, MatZnxDft, Module, ScalarZnxDftOps, Scratch};
use sampling::source::Source;

use crate::{GLWESecret, GLWESwitchingKey, Infos, ScratchCore};

pub struct TensorKey<C, B: Backend> {
    pub(crate) keys: Vec<GLWESwitchingKey<C, B>>,
}

impl TensorKey<Vec<u8>, FFT64> {
    pub fn alloc(module: &Module<FFT64>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self {
        let mut keys: Vec<GLWESwitchingKey<Vec<u8>, FFT64>> = Vec::new();
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);
        (0..pairs).for_each(|_| {
            keys.push(GLWESwitchingKey::alloc(
                module, basek, k, rows, digits, 1, rank,
            ));
        });
        Self { keys: keys }
    }

    pub fn bytes_of(module: &Module<FFT64>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize {
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);
        pairs * GLWESwitchingKey::<Vec<u8>, FFT64>::bytes_of(module, basek, k, rows, digits, 1, rank)
    }
}

impl<T, B: Backend> Infos for TensorKey<T, B> {
    type Inner = MatZnxDft<T, B>;

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

impl<T, B: Backend> TensorKey<T, B> {
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

impl TensorKey<Vec<u8>, FFT64> {
    pub fn generate_from_sk_scratch_space(module: &Module<FFT64>, basek: usize, k: usize, rank: usize) -> usize {
        GLWESecret::bytes_of(module, 1) + GLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k, rank)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> TensorKey<DataSelf, FFT64> {
    pub fn generate_from_sk<DataSk: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        sk: &GLWESecret<DataSk, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), sk.rank());
            assert_eq!(self.n(), module.n());
            assert_eq!(sk.n(), module.n());
        }

        let rank: usize = self.rank();

        let (mut sk_ij, scratch1) = scratch.tmp_sk(module, 1);

        (0..rank).for_each(|i| {
            (i..rank).for_each(|j| {
                module.svp_apply(
                    &mut sk_ij.data_fourier,
                    0,
                    &sk.data_fourier,
                    i,
                    &sk.data_fourier,
                    j,
                );
                module.svp_idft(&mut sk_ij.data, 0, &sk_ij.data_fourier, 0, scratch1);
                self.at_mut(i, j)
                    .generate_from_sk(module, &sk_ij, sk, source_xa, source_xe, sigma, scratch1);
            });
        })
    }

    // Returns a mutable reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at_mut(&mut self, mut i: usize, mut j: usize) -> &mut GLWESwitchingKey<DataSelf, FFT64> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank();
        &mut self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<DataSelf: AsRef<[u8]>> TensorKey<DataSelf, FFT64> {
    // Returns a reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at(&self, mut i: usize, mut j: usize) -> &GLWESwitchingKey<DataSelf, FFT64> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank();
        &self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}
