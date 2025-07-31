use backend::{
    Backend, Module, ScalarZnx, ScalarZnxAllocBytes, ScalarZnxAutomorphism, Scratch, SvpApply, VecZnxAddScalarInplace,
    VecZnxBigAllocBytes, VecZnxDftToVecZnxBigTmpA, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSwithcDegree, ZnxInfos,
    ZnxZero,
};
use sampling::source::Source;

use crate::{
    AutomorphismKey, GGLWECiphertext, GLWECiphertext, GLWEDecryptFamily, GLWEEncryptSkFamily, GLWEPlaintext, GLWESecret,
    GLWESecretExec, GLWESecretFamily, GLWESwitchingKey, GLWETensorKey, Infos, ScratchCore,
};

pub trait GGLWEEncryptSkFamily<B: Backend> = GLWEEncryptSkFamily<B> + GLWESecretFamily<B>;

impl GGLWECiphertext<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize) -> usize
    where
        Module<B>: GGLWEEncryptSkFamily<B>,
    {
        GLWECiphertext::encrypt_sk_scratch_space(module, basek, k)
            + (GLWEPlaintext::byte_of(module, basek, k) | module.vec_znx_normalize_tmp_bytes())
    }

    pub fn encrypt_pk_scratch_space<B: Backend>(_module: &Module<B>, _basek: usize, _k: usize, _rank: usize) -> usize {
        unimplemented!()
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GGLWECiphertext<DataSelf> {
    pub fn encrypt_sk<DataPt: AsRef<[u8]>, DataSk: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: &ScalarZnx<DataPt>,
        sk: &GLWESecretExec<DataSk, B>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        Module<B>: GGLWEEncryptSkFamily<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.rank_in(),
                pt.cols(),
                "self.rank_in(): {} != pt.cols(): {}",
                self.rank_in(),
                pt.cols()
            );
            assert_eq!(
                self.rank_out(),
                sk.rank(),
                "self.rank_out(): {} != sk.rank(): {}",
                self.rank_out(),
                sk.rank()
            );
            assert_eq!(self.n(), module.n());
            assert_eq!(sk.n(), module.n());
            assert_eq!(pt.n(), module.n());
            assert!(
                scratch.available() >= GGLWECiphertext::encrypt_sk_scratch_space(module, self.basek(), self.k()),
                "scratch.available: {} < GGLWECiphertext::encrypt_sk_scratch_space(module, self.rank()={}, self.size()={}): {}",
                scratch.available(),
                self.rank(),
                self.size(),
                GGLWECiphertext::encrypt_sk_scratch_space(module, self.basek(), self.k())
            );
            assert!(
                self.rows() * self.digits() * self.basek() <= self.k(),
                "self.rows() : {} * self.digits() : {} * self.basek() : {} = {} >= self.k() = {}",
                self.rows(),
                self.digits(),
                self.basek(),
                self.rows() * self.digits() * self.basek(),
                self.k()
            );
        }

        let rows: usize = self.rows();
        let digits: usize = self.digits();
        let basek: usize = self.basek();
        let k: usize = self.k();
        let rank_in: usize = self.rank_in();

        let (mut tmp_pt, scrach_1) = scratch.tmp_glwe_pt(module, basek, k);
        // For each input column (i.e. rank) produces a GGLWE ciphertext of rank_out+1 columns
        //
        // Example for ksk rank 2 to rank 3:
        //
        // (-(a0*s0 + a1*s1 + a2*s2) + s0', a0, a1, a2)
        // (-(b0*s0 + b1*s1 + b2*s2) + s0', b0, b1, b2)
        //
        // Example ksk rank 2 to rank 1
        //
        // (-(a*s) + s0, a)
        // (-(b*s) + s1, b)
        (0..rank_in).for_each(|col_i| {
            (0..rows).for_each(|row_i| {
                // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
                tmp_pt.data.zero(); // zeroes for next iteration
                module.vec_znx_add_scalar_inplace(
                    &mut tmp_pt.data,
                    0,
                    (digits - 1) + row_i * digits,
                    pt,
                    col_i,
                );
                module.vec_znx_normalize_inplace(basek, &mut tmp_pt.data, 0, scrach_1);

                // rlwe encrypt of vec_znx_pt into vec_znx_ct
                self.at_mut(row_i, col_i)
                    .encrypt_sk(module, &tmp_pt, sk, source_xa, source_xe, sigma, scrach_1);
            });
        });
    }
}

pub trait GLWESwitchingKeyEncryptSkFamily<B: Backend> = GGLWEEncryptSkFamily<B>;

impl GLWESwitchingKey<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> usize
    where
        Module<B>: GLWESwitchingKeyEncryptSkFamily<B>,
    {
        (GGLWECiphertext::encrypt_sk_scratch_space(module, basek, k) | module.scalar_znx_alloc_bytes(1))
            + module.scalar_znx_alloc_bytes(rank_in)
            + GLWESecretExec::bytes_of(module, rank_out)
    }

    pub fn encrypt_pk_scratch_space<B: Backend>(
        module: &Module<B>,
        _basek: usize,
        _k: usize,
        _rank_in: usize,
        _rank_out: usize,
    ) -> usize {
        GGLWECiphertext::encrypt_pk_scratch_space(module, _basek, _k, _rank_out)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GLWESwitchingKey<DataSelf> {
    pub fn encrypt_sk<DataSkIn: AsRef<[u8]>, DataSkOut: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_in: &GLWESecret<DataSkIn>,
        sk_out: &GLWESecret<DataSkOut>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        Module<B>: GLWESwitchingKeyEncryptSkFamily<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(sk_in.n() <= module.n());
            assert!(sk_out.n() <= module.n());
            assert!(
                scratch.available()
                    >= GLWESwitchingKey::encrypt_sk_scratch_space(
                        module,
                        self.basek(),
                        self.k(),
                        self.rank_in(),
                        self.rank_out()
                    ),
                "scratch.available()={} < GLWESwitchingKey::encrypt_sk_scratch_space={}",
                scratch.available(),
                GLWESwitchingKey::encrypt_sk_scratch_space(
                    module,
                    self.basek(),
                    self.k(),
                    self.rank_in(),
                    self.rank_out()
                )
            )
        }

        let (mut sk_in_tmp, scratch1) = scratch.tmp_scalar_znx(module, sk_in.rank());
        (0..sk_in.rank()).for_each(|i| {
            module.vec_znx_switch_degree(&mut sk_in_tmp, i, &sk_in.data, i);
        });

        let (mut sk_out_tmp, scratch2) = scratch1.tmp_glwe_secret_exec(module, sk_out.rank());
        {
            let (mut tmp, _) = scratch2.tmp_scalar_znx(module, 1);
            (0..sk_out.rank()).for_each(|i| {
                module.vec_znx_switch_degree(&mut tmp, 0, &sk_out.data, i);
                module.svp_prepare(&mut sk_out_tmp.data, i, &tmp, 0);
            });
        }

        self.key.encrypt_sk(
            module,
            &sk_in_tmp,
            &sk_out_tmp,
            source_xa,
            source_xe,
            sigma,
            scratch2,
        );
        self.sk_in_n = sk_in.n();
        self.sk_out_n = sk_out.n();
    }
}

pub trait AutomorphismKeyEncryptSkFamily<B: Backend> = GGLWEEncryptSkFamily<B>;

impl AutomorphismKey<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank: usize) -> usize
    where
        Module<B>: AutomorphismKeyEncryptSkFamily<B>,
    {
        GLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k, rank, rank) + GLWESecret::bytes_of(module, rank)
    }

    pub fn encrypt_pk_scratch_space<B: Backend>(module: &Module<B>, _basek: usize, _k: usize, _rank: usize) -> usize {
        GLWESwitchingKey::encrypt_pk_scratch_space(module, _basek, _k, _rank, _rank)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> AutomorphismKey<DataSelf> {
    pub fn encrypt_sk<DataSk: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        p: i64,
        sk: &GLWESecret<DataSk>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        Module<B>: AutomorphismKeyEncryptSkFamily<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), module.n());
            assert_eq!(sk.n(), module.n());
            assert_eq!(self.rank_out(), self.rank_in());
            assert_eq!(sk.rank(), self.rank());
            assert!(
                scratch.available() >= AutomorphismKey::encrypt_sk_scratch_space(module, self.basek(), self.k(), self.rank()),
                "scratch.available(): {} < AutomorphismKey::encrypt_sk_scratch_space(module, self.rank()={}, self.size()={}): {}",
                scratch.available(),
                self.rank(),
                self.size(),
                AutomorphismKey::encrypt_sk_scratch_space(module, self.basek(), self.k(), self.rank())
            )
        }

        let (mut sk_out, scratch_1) = scratch.tmp_glwe_secret(module, sk.rank());

        {
            (0..self.rank()).for_each(|i| {
                module.scalar_znx_automorphism(
                    module.galois_element_inv(p),
                    &mut sk_out.data,
                    i,
                    &sk.data,
                    i,
                );
            });
        }

        self.key
            .encrypt_sk(module, &sk, &sk_out, source_xa, source_xe, sigma, scratch_1);

        self.p = p;
    }
}

pub trait GLWETensorKeyEncryptSkFamily<B: Backend> =
    GGLWEEncryptSkFamily<B> + VecZnxBigAllocBytes + VecZnxDftToVecZnxBigTmpA<B> + SvpApply<B>;

impl GLWETensorKey<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank: usize) -> usize
    where
        Module<B>: GLWETensorKeyEncryptSkFamily<B>,
    {
        GLWESecretExec::bytes_of(module, rank)
            + module.vec_znx_dft_alloc_bytes(rank, 1)
            + module.vec_znx_big_alloc_bytes(1, 1)
            + module.vec_znx_dft_alloc_bytes(1, 1)
            + GLWESecret::bytes_of(module, 1)
            + GLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k, rank, rank)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GLWETensorKey<DataSelf> {
    pub fn encrypt_sk<DataSk: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk: &GLWESecret<DataSk>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        Module<B>: GLWETensorKeyEncryptSkFamily<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), sk.rank());
            assert_eq!(self.n(), module.n());
            assert_eq!(sk.n(), module.n());
        }

        let rank: usize = self.rank();

        let (mut sk_dft_prep, scratch1) = scratch.tmp_glwe_secret_exec(module, rank);
        sk_dft_prep.prepare(module, &sk);

        let (mut sk_dft, scratch2) = scratch1.tmp_vec_znx_dft(module, rank, 1);

        (0..rank).for_each(|i| {
            module.vec_znx_dft_from_vec_znx(1, 0, &mut sk_dft, i, &sk.data, i);
        });

        let (mut sk_ij_big, scratch3) = scratch2.tmp_vec_znx_big(module, 1, 1);
        let (mut sk_ij, scratch4) = scratch3.tmp_glwe_secret(module, 1);
        let (mut sk_ij_dft, scratch5) = scratch4.tmp_vec_znx_dft(module, 1, 1);

        (0..rank).for_each(|i| {
            (i..rank).for_each(|j| {
                module.svp_apply(&mut sk_ij_dft, 0, &sk_dft_prep.data, j, &sk_dft, i);

                module.vec_znx_dft_to_vec_znx_big_tmp_a(&mut sk_ij_big, 0, &mut sk_ij_dft, 0);
                module.vec_znx_big_normalize(self.basek(), &mut sk_ij.data, 0, &sk_ij_big, 0, scratch5);

                self.at_mut(i, j)
                    .encrypt_sk(module, &sk_ij, sk, source_xa, source_xe, sigma, scratch5);
            });
        })
    }
}
