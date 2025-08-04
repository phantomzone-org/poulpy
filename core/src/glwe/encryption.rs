use backend::hal::{
    api::{
        ScalarZnxAllocBytes, ScratchTakeScalarZnx, ScratchTakeSvpPPol, ScratchTakeVecZnx, ScratchTakeVecZnxDft, SvpApply,
        SvpApplyInplace, SvpPPolAllocBytes, SvpPrepare, VecZnxAddInplace, VecZnxAddNormal, VecZnxBigAddNormal,
        VecZnxBigAddSmallInplace, VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxDftFromVecZnx,
        VecZnxDftToVecZnxBigConsume, VecZnxFillUniform, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes,
        VecZnxSubABInplace, ZnxZero,
    },
    layouts::{Backend, Module, Scratch, VecZnxBig},
};
use sampling::source::Source;

use crate::{GLWECiphertext, GLWEPlaintext, GLWEPublicKey, GLWESecretExec, Infos, SIX_SIGMA, dist::Distribution};

pub trait GLWEEncryptSkFamily<B: Backend> =
    VecZnxDftAllocBytes + VecZnxBigNormalize<B> + VecZnxDftFromVecZnx<B> + SvpApplyInplace<B> + VecZnxDftToVecZnxBigConsume<B>;

pub trait GLWEEncryptPkFamily<B: Backend> = VecZnxDftAllocBytes
    + VecZnxBigAllocBytes
    + SvpPPolAllocBytes
    + SvpPrepare<B>
    + SvpApply<B>
    + VecZnxDftToVecZnxBigConsume<B>
    + VecZnxBigAddNormal<B>
    + VecZnxBigAddSmallInplace<B>
    + VecZnxBigNormalize<B>;

impl GLWECiphertext<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize) -> usize
    where
        Module<B>: GLWEEncryptSkFamily<B>,
    {
        let size: usize = k.div_ceil(basek);
        module.vec_znx_normalize_tmp_bytes(module.n()) + module.vec_znx_dft_alloc_bytes(1, size) + module.vec_znx_dft_alloc_bytes(1, size)
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

impl<DataSelf: AsRef<[u8]> + AsMut<[u8]>> GLWECiphertext<DataSelf> {
    pub fn encrypt_sk<DataPt: AsRef<[u8]>, DataSk: AsRef<[u8]>, B: Backend>(
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
        Scratch<B>: ScratchTakeVecZnxDft<B>,
    {
        self.encrypt_sk_private(
            module,
            Some((pt, 0)),
            sk,
            source_xa,
            source_xe,
            sigma,
            scratch,
        );
    }

    pub fn encrypt_zero_sk<DataSk: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk: &GLWESecretExec<DataSk, B>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEEncryptSkFamily<B>,
        Scratch<B>: ScratchTakeVecZnxDft<B>,
    {
        self.encrypt_sk_private(
            module,
            None::<(&GLWEPlaintext<Vec<u8>>, usize)>,
            sk,
            source_xa,
            source_xe,
            sigma,
            scratch,
        );
    }

    pub fn encrypt_pk<DataPt: AsRef<[u8]>, DataPk: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: &GLWEPlaintext<DataPt>,
        pk: &GLWEPublicKey<DataPk, B>,
        source_xu: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEEncryptPkFamily<B>,
        Scratch<B>: ScratchTakeVecZnxDft<B> + ScratchTakeSvpPPol<B>,
    {
        self.encrypt_pk_private::<DataPt, DataPk, B>(
            module,
            Some((pt, 0)),
            pk,
            source_xu,
            source_xe,
            sigma,
            scratch,
        );
    }

    pub fn encrypt_zero_pk<DataPk: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        pk: &GLWEPublicKey<DataPk, B>,
        source_xu: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEEncryptPkFamily<B>,
        Scratch<B>: ScratchTakeVecZnxDft<B> + ScratchTakeSvpPPol<B>,
    {
        self.encrypt_pk_private::<Vec<u8>, DataPk, B>(
            module,
            None::<(&GLWEPlaintext<Vec<u8>>, usize)>,
            pk,
            source_xu,
            source_xe,
            sigma,
            scratch,
        );
    }

    pub(crate) fn encrypt_sk_private<DataPt: AsRef<[u8]>, DataSk: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: Option<(&GLWEPlaintext<DataPt>, usize)>,
        sk: &GLWESecretExec<DataSk, B>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VecZnxBigNormalize<B>
            + VecZnxDftFromVecZnx<B>
            + SvpApplyInplace<B>
            + VecZnxDftToVecZnxBigConsume<B>,
        Scratch<B>: ScratchTakeVecZnxDft<B>,
    {
        #[cfg(debug_assertions)]
        {
            use backend::hal::api::ScratchAvailable;

            assert_eq!(self.rank(), sk.rank());
            assert_eq!(sk.n(), module.n());
            assert_eq!(self.n(), module.n());
            if let Some((pt, col)) = pt {
                assert_eq!(pt.n(), module.n());
                assert!(col < self.rank() + 1);
            }
            assert!(
                scratch.available() >= GLWECiphertext::encrypt_sk_scratch_space(module, self.basek(), self.k()),
                "scratch.available(): {} < GLWECiphertext::encrypt_sk_scratch_space: {}",
                scratch.available(),
                GLWECiphertext::encrypt_sk_scratch_space(module, self.basek(), self.k())
            )
        }

        let basek: usize = self.basek();
        let k: usize = self.k();
        let size: usize = self.size();
        let cols: usize = self.rank() + 1;

        let (mut c0, scratch_1) = scratch.take_vec_znx(module, 1, size);
        c0.zero();

        {
            // c[i] = uniform
            // c[0] -= c[i] * s[i],
            (1..cols).for_each(|i| {
                let (mut ci_dft, scratch_2) = scratch_1.take_vec_znx_dft(module, 1, size);

                // c[i] = uniform
                module.vec_znx_fill_uniform(basek, &mut self.data, i, k, source_xa);

                // c[i] = norm(IDFT(DFT(c[i]) * DFT(s[i])))
                module.vec_znx_dft_from_vec_znx(1, 0, &mut ci_dft, 0, &self.data, i);
                module.svp_apply_inplace(&mut ci_dft, 0, &sk.data, i - 1);
                let ci_big: VecZnxBig<&mut [u8], B> = module.vec_znx_dft_to_vec_znx_big_consume(ci_dft);

                // use c[0] as buffer, which is overwritten later by the normalization step
                module.vec_znx_big_normalize(basek, &mut self.data, 0, &ci_big, 0, scratch_2);

                // c0_tmp = -c[i] * s[i] (use c[0] as buffer)
                module.vec_znx_sub_ab_inplace(&mut c0, 0, &self.data, 0);

                // c[i] += m if col = i
                if let Some((pt, col)) = pt {
                    if i == col {
                        module.vec_znx_add_inplace(&mut self.data, i, &pt.data, 0);
                        module.vec_znx_normalize_inplace(basek, &mut self.data, i, scratch_2);
                    }
                }
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
        module.vec_znx_normalize(basek, &mut self.data, 0, &c0, 0, scratch_1);
    }

    pub(crate) fn encrypt_pk_private<DataPt: AsRef<[u8]>, DataPk: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: Option<(&GLWEPlaintext<DataPt>, usize)>,
        pk: &GLWEPublicKey<DataPk, B>,
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
        Scratch<B>: ScratchTakeVecZnxDft<B> + ScratchTakeSvpPPol<B>,
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
