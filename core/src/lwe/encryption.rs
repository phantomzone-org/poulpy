use backend::hal::{
    api::{
        ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, TakeScalarZnx, TakeVecZnx, TakeVecZnxDft, VecZnxAddNormal,
        VecZnxAddScalarInplace, VecZnxAutomorphismInplace, VecZnxFillUniform, VecZnxNormalizeInplace, VecZnxSwithcDegree,
        ZnxView, ZnxViewMut, ZnxZero,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ScratchOwned, VecZnx},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl},
};
use sampling::source::Source;

use crate::{
    GGLWEEncryptSkFamily, GLWESecret, GLWEToLWESwitchingKey, Infos, LWECiphertext, LWESecret, LWESwitchingKey,
    LWEToGLWESwitchingKey, SIX_SIGMA, TakeGLWESecret, TakeGLWESecretExec, lwe::LWEPlaintext,
};

impl<DataSelf: DataMut> LWECiphertext<DataSelf> {
    pub fn encrypt_sk<DataPt, DataSk, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: &LWEPlaintext<DataPt>,
        sk: &LWESecret<DataSk>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
    ) where
        DataPt: DataRef,
        DataSk: DataRef,
        Module<B>: VecZnxFillUniform + VecZnxAddNormal + VecZnxNormalizeInplace<B>,
        B: ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), sk.n())
        }

        let basek: usize = self.basek();
        let k: usize = self.k();

        module.vec_znx_fill_uniform(basek, &mut self.data, 0, k, source_xa);

        let mut tmp_znx: VecZnx<Vec<u8>> = VecZnx::alloc(1, 1, self.size());

        let min_size = self.size().min(pt.size());

        (0..min_size).for_each(|i| {
            tmp_znx.at_mut(0, i)[0] = pt.data.at(0, i)[0]
                - self.data.at(0, i)[1..]
                    .iter()
                    .zip(sk.data.at(0, 0))
                    .map(|(x, y)| x * y)
                    .sum::<i64>();
        });

        (min_size..self.size()).for_each(|i| {
            tmp_znx.at_mut(0, i)[0] -= self.data.at(0, i)[1..]
                .iter()
                .zip(sk.data.at(0, 0))
                .map(|(x, y)| x * y)
                .sum::<i64>();
        });

        module.vec_znx_add_normal(
            basek,
            &mut self.data,
            0,
            k,
            source_xe,
            sigma,
            sigma * SIX_SIGMA,
        );

        module.vec_znx_normalize_inplace(
            basek,
            &mut tmp_znx,
            0,
            ScratchOwned::alloc(size_of::<i64>()).borrow(),
        );

        (0..self.size()).for_each(|i| {
            self.data.at_mut(0, i)[0] = tmp_znx.at(0, i)[0];
        });
    }
}

impl<D: DataMut> GLWEToLWESwitchingKey<D> {
    pub fn encrypt_sk<DLwe, DGlwe, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_lwe: &LWESecret<DLwe>,
        sk_glwe: &GLWESecret<DGlwe>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        DLwe: DataRef,
        DGlwe: DataRef,
        Module<B>: GGLWEEncryptSkFamily<B> + VecZnxAutomorphismInplace + VecZnxSwithcDegree + VecZnxAddScalarInplace,
        Scratch<B>: ScratchAvailable + TakeScalarZnx + TakeVecZnxDft<B> + TakeGLWESecretExec<B> + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert!(sk_lwe.n() <= module.n());
        }

        let (mut sk_lwe_as_glwe, scratch1) = scratch.take_glwe_secret(sk_glwe.n(), 1);
        sk_lwe_as_glwe.data.zero();
        sk_lwe_as_glwe.data.at_mut(0, 0)[..sk_lwe.n()].copy_from_slice(sk_lwe.data.at(0, 0));
        module.vec_znx_automorphism_inplace(-1, &mut sk_lwe_as_glwe.data.as_vec_znx_mut(), 0);

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

impl<D: DataMut> LWEToGLWESwitchingKey<D> {
    pub fn encrypt_sk<DLwe, DGlwe, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_lwe: &LWESecret<DLwe>,
        sk_glwe: &GLWESecret<DGlwe>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        DLwe: DataRef,
        DGlwe: DataRef,
        Module<B>: GGLWEEncryptSkFamily<B> + VecZnxAutomorphismInplace + VecZnxSwithcDegree + VecZnxAddScalarInplace,
        Scratch<B>: ScratchAvailable + TakeScalarZnx + TakeVecZnxDft<B> + TakeGLWESecretExec<B> + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert!(sk_lwe.n() <= module.n());
        }

        let (mut sk_lwe_as_glwe, scratch1) = scratch.take_glwe_secret(sk_glwe.n(), 1);
        sk_lwe_as_glwe.data.at_mut(0, 0)[..sk_lwe.n()].copy_from_slice(sk_lwe.data.at(0, 0));
        sk_lwe_as_glwe.data.at_mut(0, 0)[sk_lwe.n()..].fill(0);
        module.vec_znx_automorphism_inplace(-1, &mut sk_lwe_as_glwe.data.as_vec_znx_mut(), 0);

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

impl<D: DataMut> LWESwitchingKey<D> {
    pub fn encrypt_sk<DIn, DOut, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_lwe_in: &LWESecret<DIn>,
        sk_lwe_out: &LWESecret<DOut>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        DIn: DataRef,
        DOut: DataRef,
        Module<B>: GGLWEEncryptSkFamily<B> + VecZnxAutomorphismInplace + VecZnxSwithcDegree + VecZnxAddScalarInplace,
        Scratch<B>: ScratchAvailable + TakeScalarZnx + TakeVecZnxDft<B> + TakeGLWESecretExec<B> + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert!(sk_lwe_in.n() <= self.n());
            assert!(sk_lwe_out.n() <= self.n());
            assert!(self.n() <= module.n());
        }

        let (mut sk_in_glwe, scratch1) = scratch.take_glwe_secret(self.n(), 1);
        let (mut sk_out_glwe, scratch2) = scratch1.take_glwe_secret(self.n(), 1);

        sk_out_glwe.data.at_mut(0, 0)[..sk_lwe_out.n()].copy_from_slice(sk_lwe_out.data.at(0, 0));
        sk_out_glwe.data.at_mut(0, 0)[sk_lwe_out.n()..].fill(0);
        module.vec_znx_automorphism_inplace(-1, &mut sk_out_glwe.data.as_vec_znx_mut(), 0);

        sk_in_glwe.data.at_mut(0, 0)[..sk_lwe_in.n()].copy_from_slice(sk_lwe_in.data.at(0, 0));
        sk_in_glwe.data.at_mut(0, 0)[sk_lwe_in.n()..].fill(0);
        module.vec_znx_automorphism_inplace(-1, &mut sk_in_glwe.data.as_vec_znx_mut(), 0);

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
