use backend::hal::{
    api::{
        ScalarZnxAllocBytes, ScratchAvailable, SvpApply, SvpApplyInplace, SvpPPolAllocBytes, SvpPrepare, TakeScalarZnx,
        TakeSvpPPol, TakeVecZnx, TakeVecZnxDft, VecZnxAddInplace, VecZnxAddNormal, VecZnxAllocBytes, VecZnxBigAddNormal,
        VecZnxBigAddSmallInplace, VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxDftFromVecZnx,
        VecZnxDftToVecZnxBigConsume, VecZnxFillUniform, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes,
        VecZnxSub, VecZnxSubABInplace, ZnxInfos, ZnxZero,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, VecZnx, VecZnxBig},
};
use sampling::source::Source;

use crate::{
    GLWECiphertext, GLWECiphertextCompressed, GLWEPlaintext, GLWEPublicKeyExec, GLWESecretExec, Infos, SIX_SIGMA,
    dist::Distribution,
};

pub trait GLWEEncryptSkFamily<B: Backend> = VecZnxDftAllocBytes
    + VecZnxBigNormalize<B>
    + VecZnxDftFromVecZnx<B>
    + SvpApplyInplace<B>
    + VecZnxDftToVecZnxBigConsume<B>
    + VecZnxNormalizeTmpBytes
    + VecZnxFillUniform
    + VecZnxSubABInplace
    + VecZnxAddInplace
    + VecZnxNormalizeInplace<B>
    + VecZnxAddNormal
    + VecZnxNormalize<B>
    + VecZnxSub
    + VecZnxAllocBytes;

pub trait GLWEEncryptPkFamily<B: Backend> = VecZnxDftAllocBytes
    + VecZnxBigAllocBytes
    + SvpPPolAllocBytes
    + SvpPrepare<B>
    + SvpApply<B>
    + VecZnxDftToVecZnxBigConsume<B>
    + VecZnxBigAddNormal<B>
    + VecZnxBigAddSmallInplace<B>
    + VecZnxBigNormalize<B>
    + ScalarZnxAllocBytes
    + VecZnxNormalizeTmpBytes;

impl GLWECiphertext<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize) -> usize
    where
        Module<B>: GLWEEncryptSkFamily<B>,
    {
        let size: usize = k.div_ceil(basek);
        module.vec_znx_normalize_tmp_bytes(module.n())
            + 2 * module.vec_znx_alloc_bytes(1, size)
            + module.vec_znx_dft_alloc_bytes(1, size)
    }
    pub fn encrypt_pk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize) -> usize
    where
        Module<B>: GLWEEncryptPkFamily<B>,
    {
        let size: usize = k.div_ceil(basek);
        ((module.vec_znx_dft_alloc_bytes(1, size) + module.vec_znx_big_alloc_bytes(1, size)) | module.scalar_znx_alloc_bytes(1))
            + module.svp_ppol_alloc_bytes(1)
            + module.vec_znx_normalize_tmp_bytes(module.n())
    }
}

impl<DataSelf: DataMut> GLWECiphertext<DataSelf> {
    pub fn encrypt_sk<DataPt: DataRef, DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: &GLWEPlaintext<DataPt>,
        sk: &GLWESecretExec<DataSk, B>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEEncryptSkFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), sk.rank());
            assert_eq!(sk.n(), self.n());
            assert_eq!(pt.n(), self.n());
            assert!(
                scratch.available() >= GLWECiphertext::encrypt_sk_scratch_space(module, self.basek(), self.k()),
                "scratch.available(): {} < GLWECiphertext::encrypt_sk_scratch_space: {}",
                scratch.available(),
                GLWECiphertext::encrypt_sk_scratch_space(module, self.basek(), self.k())
            )
        }

        self.encrypt_sk_internal(
            module,
            Some((pt, 0)),
            sk,
            source_xa,
            source_xe,
            sigma,
            scratch,
        );
    }

    pub fn encrypt_zero_sk<DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk: &GLWESecretExec<DataSk, B>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEEncryptSkFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), sk.rank());
            assert_eq!(sk.n(), self.n());
            assert!(
                scratch.available() >= GLWECiphertext::encrypt_sk_scratch_space(module, self.basek(), self.k()),
                "scratch.available(): {} < GLWECiphertext::encrypt_sk_scratch_space: {}",
                scratch.available(),
                GLWECiphertext::encrypt_sk_scratch_space(module, self.basek(), self.k())
            )
        }
        self.encrypt_sk_internal(
            module,
            None::<(&GLWEPlaintext<Vec<u8>>, usize)>,
            sk,
            source_xa,
            source_xe,
            sigma,
            scratch,
        );
    }

    pub(crate) fn encrypt_sk_internal<DataPt: DataRef, DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: Option<(&GLWEPlaintext<DataPt>, usize)>,
        sk: &GLWESecretExec<DataSk, B>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEEncryptSkFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx<B>,
    {
        let cols: usize = self.rank() + 1;
        encrypt_sk_internal(
            module,
            self.basek(),
            self.k(),
            &mut self.data,
            cols,
            false,
            pt,
            sk,
            source_xa,
            source_xe,
            sigma,
            scratch,
        );
    }

    pub fn encrypt_pk<DataPt: DataRef, DataPk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: &GLWEPlaintext<DataPt>,
        pk: &GLWEPublicKeyExec<DataPk, B>,
        source_xu: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEEncryptPkFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + TakeSvpPPol<B> + TakeScalarZnx<B>,
    {
        self.encrypt_pk_internal::<DataPt, DataPk, B>(
            module,
            Some((pt, 0)),
            pk,
            source_xu,
            source_xe,
            sigma,
            scratch,
        );
    }

    pub fn encrypt_zero_pk<DataPk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        pk: &GLWEPublicKeyExec<DataPk, B>,
        source_xu: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEEncryptPkFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + TakeSvpPPol<B> + TakeScalarZnx<B>,
    {
        self.encrypt_pk_internal::<Vec<u8>, DataPk, B>(
            module,
            None::<(&GLWEPlaintext<Vec<u8>>, usize)>,
            pk,
            source_xu,
            source_xe,
            sigma,
            scratch,
        );
    }

    pub(crate) fn encrypt_pk_internal<DataPt: DataRef, DataPk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: Option<(&GLWEPlaintext<DataPt>, usize)>,
        pk: &GLWEPublicKeyExec<DataPk, B>,
        source_xu: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + SvpPPolAllocBytes
            + SvpPrepare<B>
            + SvpApply<B>
            + VecZnxDftToVecZnxBigConsume<B>
            + VecZnxBigAddNormal<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + TakeSvpPPol<B> + TakeScalarZnx<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.basek(), pk.basek());
            assert_eq!(self.n(), module.n());
            assert_eq!(pk.n(), module.n());
            assert_eq!(self.rank(), pk.rank());
            if let Some((pt, _)) = pt {
                assert_eq!(pt.basek(), pk.basek());
                assert_eq!(pt.n(), module.n());
            }
        }

        let basek: usize = pk.basek();
        let size_pk: usize = pk.size();
        let cols: usize = self.rank() + 1;

        // Generates u according to the underlying secret distribution.
        let (mut u_dft, scratch_1) = scratch.take_svp_ppol(module, 1);

        {
            let (mut u, _) = scratch_1.take_scalar_znx(module, 1);
            match pk.dist {
                Distribution::NONE => panic!(
                    "invalid public key: SecretDistribution::NONE, ensure it has been correctly intialized through \
                     Self::generate"
                ),
                Distribution::TernaryFixed(hw) => u.fill_ternary_hw(0, hw, source_xu),
                Distribution::TernaryProb(prob) => u.fill_ternary_prob(0, prob, source_xu),
                Distribution::BinaryFixed(hw) => u.fill_binary_hw(0, hw, source_xu),
                Distribution::BinaryProb(prob) => u.fill_binary_prob(0, prob, source_xu),
                Distribution::BinaryBlock(block_size) => u.fill_binary_block(0, block_size, source_xu),
                Distribution::ZERO => {}
            }

            module.svp_prepare(&mut u_dft, 0, &u, 0);
        }

        // ct[i] = pk[i] * u + ei (+ m if col = i)
        (0..cols).for_each(|i| {
            let (mut ci_dft, scratch_2) = scratch_1.take_vec_znx_dft(module, 1, size_pk);
            // ci_dft = DFT(u) * DFT(pk[i])
            module.svp_apply(&mut ci_dft, 0, &u_dft, 0, &pk.data, i);

            // ci_big = u * p[i]
            let mut ci_big = module.vec_znx_dft_to_vec_znx_big_consume(ci_dft);

            // ci_big = u * pk[i] + e
            module.vec_znx_big_add_normal(
                basek,
                &mut ci_big,
                0,
                pk.k(),
                source_xe,
                sigma,
                sigma * SIX_SIGMA,
            );

            // ci_big = u * pk[i] + e + m (if col = i)
            if let Some((pt, col)) = pt {
                if col == i {
                    module.vec_znx_big_add_small_inplace(&mut ci_big, 0, &pt.data, 0);
                }
            }

            // ct[i] = norm(ci_big)
            module.vec_znx_big_normalize(basek, &mut self.data, i, &ci_big, 0, scratch_2);
        });
    }
}

impl GLWECiphertextCompressed<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize) -> usize
    where
        Module<B>: GLWEEncryptSkFamily<B>,
    {
        GLWECiphertext::encrypt_sk_scratch_space(module, basek, k)
    }
}
impl<D: DataMut> GLWECiphertextCompressed<D> {
    pub fn encrypt_sk<DataPt: DataRef, DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: &GLWEPlaintext<DataPt>,
        sk: &GLWESecretExec<DataSk, B>,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEEncryptSkFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx<B>,
    {
        self.encrypt_sk_internal(
            module,
            Some((pt, 0)),
            sk,
            seed_xa,
            source_xe,
            sigma,
            scratch,
        );
    }

    pub(crate) fn encrypt_sk_internal<DataPt: DataRef, DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: Option<(&GLWEPlaintext<DataPt>, usize)>,
        sk: &GLWESecretExec<DataSk, B>,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEEncryptSkFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx<B>,
    {
        let mut source_xa = Source::new(seed_xa);
        let cols: usize = self.rank() + 1;
        encrypt_sk_internal(
            module,
            self.basek(),
            self.k(),
            &mut self.data,
            cols,
            true,
            pt,
            sk,
            &mut source_xa,
            source_xe,
            sigma,
            scratch,
        );
        self.seed = seed_xa;
    }
}

pub(crate) fn encrypt_sk_internal<DataCt: DataMut, DataPt: DataRef, DataSk: DataRef, B: Backend>(
    module: &Module<B>,
    basek: usize,
    k: usize,
    ct: &mut VecZnx<DataCt>,
    cols: usize,
    compressed: bool,
    pt: Option<(&GLWEPlaintext<DataPt>, usize)>,
    sk: &GLWESecretExec<DataSk, B>,
    source_xa: &mut Source,
    source_xe: &mut Source,
    sigma: f64,
    scratch: &mut Scratch<B>,
) where
    Module<B>: GLWEEncryptSkFamily<B>,
    Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx<B>,
{
    #[cfg(debug_assertions)]
    {
        if compressed {
            assert_eq!(
                ct.cols(),
                1,
                "invalid ciphertext: compressed tag=true but #cols={} != 1",
                ct.cols()
            )
        }
    }

    let size: usize = ct.size();

    let (mut c0, scratch_1) = scratch.take_vec_znx(module, 1, size);
    c0.zero();

    {
        let (mut ci, scratch_2) = scratch_1.take_vec_znx(module, 1, size);

        // ct[i] = uniform
        // ct[0] -= c[i] * s[i],
        (1..cols).for_each(|i| {
            let col_ct: usize;
            if compressed {
                col_ct = 0;
            } else {
                col_ct = i;
            }

            // ct[i] = uniform (+ pt)
            module.vec_znx_fill_uniform(basek, ct, col_ct, k, source_xa);

            let (mut ci_dft, scratch_3) = scratch_2.take_vec_znx_dft(module, 1, size);

            // ci = ct[i] - pt
            // i.e. we act as we sample ct[i] already as uniform + pt
            // and if there is a pt, then we subtract it before applying DFT
            if let Some((pt, col)) = pt {
                if i == col {
                    module.vec_znx_sub(&mut ci, 0, ct, col_ct, &pt.data, 0);
                    module.vec_znx_normalize_inplace(basek, &mut ci, 0, scratch_3);
                    module.vec_znx_dft_from_vec_znx(1, 0, &mut ci_dft, 0, &ci, 0);
                } else {
                    module.vec_znx_dft_from_vec_znx(1, 0, &mut ci_dft, 0, ct, col_ct);
                }
            } else {
                module.vec_znx_dft_from_vec_znx(1, 0, &mut ci_dft, 0, ct, col_ct);
            }

            module.svp_apply_inplace(&mut ci_dft, 0, &sk.data, i - 1);
            let ci_big: VecZnxBig<&mut [u8], B> = module.vec_znx_dft_to_vec_znx_big_consume(ci_dft);

            // use c[0] as buffer, which is overwritten later by the normalization step
            module.vec_znx_big_normalize(basek, &mut ci, 0, &ci_big, 0, scratch_3);

            // c0_tmp = -c[i] * s[i] (use c[0] as buffer)
            module.vec_znx_sub_ab_inplace(&mut c0, 0, &ci, 0);
        });
    }

    // c[0] += e
    module.vec_znx_add_normal(basek, &mut c0, 0, k, source_xe, sigma, sigma * SIX_SIGMA);

    // c[0] += m if col = 0
    if let Some((pt, col)) = pt {
        if col == 0 {
            module.vec_znx_add_inplace(&mut c0, 0, &pt.data, 0);
        }
    }

    // c[0] = norm(c[0])
    module.vec_znx_normalize(basek, ct, 0, &c0, 0, scratch_1);
}
