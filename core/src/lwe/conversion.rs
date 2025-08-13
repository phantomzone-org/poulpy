use backend::hal::{
    api::{ScratchAvailable, TakeVecZnx, TakeVecZnxDft, ZnxView, ZnxViewMut, ZnxZero},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{
    GLWECiphertext, GLWEKeyswitchFamily, GLWEToLWESwitchingKeyExec, Infos, LWECiphertext, LWESwitchingKeyExec,
    LWEToGLWESwitchingKeyExec, TakeGLWECt,
};

impl<DLwe: DataMut> LWECiphertext<DLwe> {
    pub fn sample_extract<DGlwe: DataRef>(&mut self, a: &GLWECiphertext<DGlwe>) {
        #[cfg(debug_assertions)]
        {
            assert!(self.n() <= a.n());
        }

        let min_size: usize = self.size().min(a.size());
        let n: usize = self.n();

        self.data.zero();
        (0..min_size).for_each(|i| {
            let data_lwe: &mut [i64] = self.data.at_mut(0, i);
            data_lwe[0] = a.data.at(0, i)[0];
            data_lwe[1..].copy_from_slice(&a.data.at(1, i)[..n]);
        });
    }

    pub fn from_glwe<DGlwe, DKs, B: Backend>(
        &mut self,
        module: &Module<B>,
        a: &GLWECiphertext<DGlwe>,
        ks: &GLWEToLWESwitchingKeyExec<DKs, B>,
        scratch: &mut Scratch<B>,
    ) where
        DGlwe: DataRef,
        DKs: DataRef,
        Module<B>: GLWEKeyswitchFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.basek(), a.basek());
            assert_eq!(a.n(), ks.n());
        }
        let (mut tmp_glwe, scratch1) = scratch.take_glwe_ct(a.n(), a.basek(), self.k(), 1);
        tmp_glwe.keyswitch(module, a, &ks.0, scratch1);
        self.sample_extract(&tmp_glwe);
    }

    pub fn keyswitch<A, DKs, B: Backend>(
        &mut self,
        module: &Module<B>,
        a: &LWECiphertext<A>,
        ksk: &LWESwitchingKeyExec<DKs, B>,
        scratch: &mut Scratch<B>,
    ) where
        A: DataRef,
        DKs: DataRef,
        Module<B>: GLWEKeyswitchFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert!(self.n() <= module.n());
            assert!(a.n() <= module.n());
            assert_eq!(self.basek(), a.basek());
        }

        let max_k: usize = self.k().max(a.k());
        let basek: usize = self.basek();

        let (mut glwe, scratch1) = scratch.take_glwe_ct(ksk.n(), basek, max_k, 1);
        glwe.data.zero();

        let n_lwe: usize = a.n();

        (0..a.size()).for_each(|i| {
            let data_lwe: &[i64] = a.data.at(0, i);
            glwe.data.at_mut(0, i)[0] = data_lwe[0];
            glwe.data.at_mut(1, i)[..n_lwe].copy_from_slice(&data_lwe[1..]);
        });

        glwe.keyswitch_inplace(module, &ksk.0, scratch1);

        self.sample_extract(&glwe);
    }
}

impl GLWECiphertext<Vec<u8>> {
    pub fn from_lwe_scratch_space<B: Backend>(
        module: &Module<B>,
        n: usize,
        basek: usize,
        k_lwe: usize,
        k_glwe: usize,
        k_ksk: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: GLWEKeyswitchFamily<B>,
    {
        GLWECiphertext::keyswitch_scratch_space(module, n, basek, k_glwe, k_lwe, k_ksk, 1, 1, rank)
            + GLWECiphertext::bytes_of(n, basek, k_lwe, 1)
    }
}

impl<D: DataMut> GLWECiphertext<D> {
    pub fn from_lwe<DLwe, DKsk, B: Backend>(
        &mut self,
        module: &Module<B>,
        lwe: &LWECiphertext<DLwe>,
        ksk: &LWEToGLWESwitchingKeyExec<DKsk, B>,
        scratch: &mut Scratch<B>,
    ) where
        DLwe: DataRef,
        DKsk: DataRef,
        Module<B>: GLWEKeyswitchFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert!(lwe.n() <= self.n());
            assert_eq!(self.basek(), self.basek());
        }

        let (mut glwe, scratch1) = scratch.take_glwe_ct(ksk.n(), lwe.basek(), lwe.k(), 1);
        glwe.data.zero();

        let n_lwe: usize = lwe.n();

        (0..lwe.size()).for_each(|i| {
            let data_lwe: &[i64] = lwe.data.at(0, i);
            glwe.data.at_mut(0, i)[0] = data_lwe[0];
            glwe.data.at_mut(1, i)[..n_lwe].copy_from_slice(&data_lwe[1..]);
        });

        self.keyswitch(module, &glwe, &ksk.0, scratch1);
    }
}
