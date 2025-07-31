use backend::{Backend, MatZnxAlloc, Module, Scratch, VecZnxAutomorphismInplace, ZnxView, ZnxViewMut, ZnxZero};
use sampling::source::Source;

use crate::{
    GGLWEEncryptSkFamily, GGLWEExecLayoutFamily, GLWECiphertext, GLWEKeyswitchFamily, GLWESecret, GLWESecretExec,
    GLWESwitchingKey, GLWESwitchingKeyExec, Infos, LWECiphertext, LWESecret, ScratchCore,
};

/// A special [GLWESwitchingKey] required to for the conversion from [GLWECiphertext] to [LWECiphertext].
pub struct GLWEToLWESwitchingKey<D>(GLWESwitchingKey<D>);
pub struct GLWEToLWESwitchingKeyExec<D, B: Backend>(GLWESwitchingKeyExec<D, B>);

impl<B: Backend> GLWEToLWESwitchingKeyExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rows: usize, rank_in: usize) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        Self(GLWESwitchingKeyExec::alloc(
            module, basek, k, rows, 1, rank_in, 1,
        ))
    }

    pub fn bytes_of(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank_in: usize) -> usize
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        GLWESwitchingKeyExec::<Vec<u8>, B>::bytes_of(module, basek, k, rows, digits, rank_in, 1)
    }

    pub fn from<DataOther: AsRef<[u8]>>(
        module: &Module<B>,
        other: &GLWEToLWESwitchingKey<DataOther>,
        scratch: &mut Scratch,
    ) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        let mut ksk_exec: GLWEToLWESwitchingKeyExec<Vec<u8>, B> = Self::alloc(
            module,
            other.0.basek(),
            other.0.k(),
            other.0.rows(),
            other.0.rank_in(),
        );
        ksk_exec.prepare(module, other, scratch);
        ksk_exec
    }
}

impl<D: AsRef<[u8]> + AsMut<[u8]>, B: Backend> GLWEToLWESwitchingKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &GLWEToLWESwitchingKey<DataOther>, scratch: &mut Scratch)
    where
        DataOther: AsRef<[u8]>,
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        self.0.prepare(module, &other.0, scratch);
    }
}

impl GLWEToLWESwitchingKey<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, rank_in: usize) -> Self
    where
        Module<B>: MatZnxAlloc,
    {
        Self(GLWESwitchingKey::alloc(
            module, basek, k, rows, 1, rank_in, 1,
        ))
    }

    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank_in: usize) -> usize
    where
        Module<B>: GGLWEEncryptSkFamily<B>,
    {
        GLWESecretExec::bytes_of(module, rank_in)
            + (GLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k, rank_in, 1) | GLWESecret::bytes_of(module, rank_in))
    }
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> GLWEToLWESwitchingKey<D> {
    pub fn encrypt_sk<DLwe, DGlwe, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_lwe: &LWESecret<DLwe>,
        sk_glwe: &GLWESecret<DGlwe>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        DLwe: AsRef<[u8]>,
        DGlwe: AsRef<[u8]>,
        Module<B>: GGLWEEncryptSkFamily<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(sk_lwe.n() <= module.n());
        }

        let (mut sk_lwe_as_glwe, scratch1) = scratch.tmp_glwe_secret(module, 1);
        sk_lwe_as_glwe.data.zero();
        sk_lwe_as_glwe.data.at_mut(0, 0)[..sk_lwe.n()].copy_from_slice(sk_lwe.data.at(0, 0));
        module.vec_znx_automorphism_inplace(-1, &mut sk_lwe_as_glwe.data, 0);

        self.0.encrypt_sk(
            module,
            sk_glwe,
            &sk_lwe_as_glwe,
            source_xa,
            source_xe,
            sigma,
            scratch1,
        );
    }
}

/// A special [GLWESwitchingKey] required to for the conversion from [LWECiphertext] to [GLWECiphertext].
pub struct LWEToGLWESwitchingKeyExec<D, B: Backend>(GLWESwitchingKeyExec<D, B>);

impl<B: Backend> LWEToGLWESwitchingKeyExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rows: usize, rank_out: usize) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        Self(GLWESwitchingKeyExec::alloc(
            module, basek, k, rows, 1, 1, rank_out,
        ))
    }

    pub fn bytes_of(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank_out: usize) -> usize
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        GLWESwitchingKeyExec::<Vec<u8>, B>::bytes_of(module, basek, k, rows, digits, 1, rank_out)
    }

    pub fn from<DataOther: AsRef<[u8]>>(
        module: &Module<B>,
        other: &LWEToGLWESwitchingKey<DataOther>,
        scratch: &mut Scratch,
    ) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        let mut ksk_exec: LWEToGLWESwitchingKeyExec<Vec<u8>, B> = Self::alloc(
            module,
            other.0.basek(),
            other.0.k(),
            other.0.rows(),
            other.0.rank(),
        );
        ksk_exec.prepare(module, other, scratch);
        ksk_exec
    }
}

impl<D: AsRef<[u8]> + AsMut<[u8]>, B: Backend> LWEToGLWESwitchingKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &LWEToGLWESwitchingKey<DataOther>, scratch: &mut Scratch)
    where
        DataOther: AsRef<[u8]>,
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        self.0.prepare(module, &other.0, scratch);
    }
}
pub struct LWEToGLWESwitchingKey<D>(GLWESwitchingKey<D>);

impl LWEToGLWESwitchingKey<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, rank_out: usize) -> Self
    where
        Module<B>: MatZnxAlloc,
    {
        Self(GLWESwitchingKey::alloc(
            module, basek, k, rows, 1, 1, rank_out,
        ))
    }

    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank_out: usize) -> usize
    where
        Module<B>: GGLWEEncryptSkFamily<B>,
    {
        GLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k, 1, rank_out) + GLWESecret::bytes_of(module, 1)
    }
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> LWEToGLWESwitchingKey<D> {
    pub fn encrypt_sk<DLwe, DGlwe, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_lwe: &LWESecret<DLwe>,
        sk_glwe: &GLWESecret<DGlwe>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        DLwe: AsRef<[u8]>,
        DGlwe: AsRef<[u8]>,
        Module<B>: GGLWEEncryptSkFamily<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(sk_lwe.n() <= module.n());
        }

        let (mut sk_lwe_as_glwe, scratch1) = scratch.tmp_glwe_secret(module, 1);
        sk_lwe_as_glwe.data.at_mut(0, 0)[..sk_lwe.n()].copy_from_slice(sk_lwe.data.at(0, 0));
        sk_lwe_as_glwe.data.at_mut(0, 0)[sk_lwe.n()..].fill(0);
        module.vec_znx_automorphism_inplace(-1, &mut sk_lwe_as_glwe.data, 0);

        self.0.encrypt_sk(
            module,
            &sk_lwe_as_glwe,
            &sk_glwe,
            source_xa,
            source_xe,
            sigma,
            scratch1,
        );
    }
}

pub struct LWESwitchingKeyExec<D, B: Backend>(GLWESwitchingKeyExec<D, B>);

impl<B: Backend> LWESwitchingKeyExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rows: usize) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        Self(GLWESwitchingKeyExec::alloc(module, basek, k, rows, 1, 1, 1))
    }

    pub fn bytes_of(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize) -> usize
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        GLWESwitchingKeyExec::<Vec<u8>, B>::bytes_of(module, basek, k, rows, digits, 1, 1)
    }

    pub fn from<DataOther: AsRef<[u8]>>(module: &Module<B>, other: &LWESwitchingKey<DataOther>, scratch: &mut Scratch) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        let mut ksk_exec: LWESwitchingKeyExec<Vec<u8>, B> = Self::alloc(module, other.0.basek(), other.0.k(), other.0.rows());
        ksk_exec.prepare(module, other, scratch);
        ksk_exec
    }
}

impl<D: AsRef<[u8]> + AsMut<[u8]>, B: Backend> LWESwitchingKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &LWESwitchingKey<DataOther>, scratch: &mut Scratch)
    where
        DataOther: AsRef<[u8]>,
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        self.0.prepare(module, &other.0, scratch);
    }
}
pub struct LWESwitchingKey<D>(GLWESwitchingKey<D>);

impl LWESwitchingKey<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize) -> Self
    where
        Module<B>: MatZnxAlloc,
    {
        Self(GLWESwitchingKey::alloc(module, basek, k, rows, 1, 1, 1))
    }

    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize) -> usize
    where
        Module<B>: GGLWEEncryptSkFamily<B>,
    {
        GLWESecret::bytes_of(module, 1)
            + GLWESecretExec::bytes_of(module, 1)
            + GLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k, 1, 1)
    }
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> LWESwitchingKey<D> {
    pub fn encrypt_sk<DIn, DOut, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_lwe_in: &LWESecret<DIn>,
        sk_lwe_out: &LWESecret<DOut>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        DIn: AsRef<[u8]>,
        DOut: AsRef<[u8]>,
        Module<B>: GGLWEEncryptSkFamily<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(sk_lwe_in.n() <= module.n());
            assert!(sk_lwe_out.n() <= module.n());
        }

        let (mut sk_in_glwe, scratch1) = scratch.tmp_glwe_secret(module, 1);
        let (mut sk_out_glwe, scratch2) = scratch1.tmp_glwe_secret(module, 1);

        sk_out_glwe.data.at_mut(0, 0)[..sk_lwe_out.n()].copy_from_slice(sk_lwe_out.data.at(0, 0));
        sk_out_glwe.data.at_mut(0, 0)[sk_lwe_out.n()..].fill(0);
        module.vec_znx_automorphism_inplace(-1, &mut sk_out_glwe.data, 0);

        sk_in_glwe.data.at_mut(0, 0)[..sk_lwe_in.n()].copy_from_slice(sk_lwe_in.data.at(0, 0));
        sk_in_glwe.data.at_mut(0, 0)[sk_lwe_in.n()..].fill(0);
        module.vec_znx_automorphism_inplace(-1, &mut sk_in_glwe.data, 0);

        self.0.encrypt_sk(
            module,
            &sk_in_glwe,
            &sk_out_glwe,
            source_xa,
            source_xe,
            sigma,
            scratch2,
        );
    }
}

impl LWECiphertext<Vec<u8>> {
    pub fn from_glwe_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_lwe: usize,
        k_glwe: usize,
        k_ksk: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: GLWEKeyswitchFamily<B>,
    {
        GLWECiphertext::bytes_of(module, basek, k_lwe, 1)
            + GLWECiphertext::keyswitch_scratch_space(module, basek, k_lwe, k_glwe, k_ksk, 1, rank, 1)
    }

    pub fn keyswitch_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_lwe_out: usize,
        k_lwe_in: usize,
        k_ksk: usize,
    ) -> usize
    where
        Module<B>: GLWEKeyswitchFamily<B>,
    {
        GLWECiphertext::bytes_of(module, basek, k_lwe_out.max(k_lwe_in), 1)
            + GLWECiphertext::keyswitch_inplace_scratch_space(module, basek, k_lwe_out, k_ksk, 1, 1)
    }
}

impl<DLwe: AsRef<[u8]> + AsMut<[u8]>> LWECiphertext<DLwe> {
    pub fn sample_extract<DGlwe>(&mut self, a: &GLWECiphertext<DGlwe>)
    where
        DGlwe: AsRef<[u8]>,
    {
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
        scratch: &mut Scratch,
    ) where
        DGlwe: AsRef<[u8]>,
        DKs: AsRef<[u8]>,
        Module<B>: GLWEKeyswitchFamily<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.basek(), a.basek());
        }
        let (mut tmp_glwe, scratch1) = scratch.tmp_glwe_ct(module, a.basek(), self.k(), 1);
        tmp_glwe.keyswitch(module, a, &ks.0, scratch1);
        self.sample_extract(&tmp_glwe);
    }

    pub fn keyswitch<A, DKs, B: Backend>(
        &mut self,
        module: &Module<B>,
        a: &LWECiphertext<A>,
        ksk: &LWESwitchingKeyExec<DKs, B>,
        scratch: &mut Scratch,
    ) where
        A: AsRef<[u8]>,
        DKs: AsRef<[u8]>,
        Module<B>: GLWEKeyswitchFamily<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(self.n() <= module.n());
            assert!(a.n() <= module.n());
            assert_eq!(self.basek(), a.basek());
        }

        let max_k: usize = self.k().max(a.k());
        let basek: usize = self.basek();

        let (mut glwe, scratch1) = scratch.tmp_glwe_ct(&module, basek, max_k, 1);
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
        basek: usize,
        k_lwe: usize,
        k_glwe: usize,
        k_ksk: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: GLWEKeyswitchFamily<B>,
    {
        GLWECiphertext::keyswitch_scratch_space(module, basek, k_glwe, k_lwe, k_ksk, 1, 1, rank)
            + GLWECiphertext::bytes_of(module, basek, k_lwe, 1)
    }
}

impl<D: AsRef<[u8]> + AsMut<[u8]>> GLWECiphertext<D> {
    pub fn from_lwe<DLwe, DKsk, B: Backend>(
        &mut self,
        module: &Module<B>,
        lwe: &LWECiphertext<DLwe>,
        ksk: &LWEToGLWESwitchingKeyExec<DKsk, B>,
        scratch: &mut Scratch,
    ) where
        DLwe: AsRef<[u8]>,
        DKsk: AsRef<[u8]>,
        Module<B>: GLWEKeyswitchFamily<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(lwe.n() <= self.n());
            assert_eq!(self.basek(), self.basek());
        }

        let (mut glwe, scratch1) = scratch.tmp_glwe_ct(module, lwe.basek(), lwe.k(), 1);
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
