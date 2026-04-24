use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, ScratchOwnedAlloc, SvpApplyDftToDft, SvpApplyDftToDftInplace, SvpPPolBytesOf, SvpPrepare,
        VecZnxAddAssign, VecZnxAddNormal, VecZnxBigAddNormal, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxCopy, VecZnxDftApply, VecZnxDftBytesOf, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNormalize,
        VecZnxNormalizeInplaceBackend, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace,
    },
    layouts::{
        Backend, HostDataMut, Module, ScalarZnx, ScratchArena, ScratchOwned, VecZnx, VecZnxBigToBackendRef, VecZnxDft,
        VecZnxDftToBackendMut, VecZnxToBackendMut, VecZnxToBackendRef, VecZnxToMut, ZnxInfos, ZnxZero,
        svp_ppol_backend_ref_from_mut, vec_znx_backend_mut_from_mut, vec_znx_big_backend_ref_from_mut,
    },
    source::Source,
};

use crate::{
    EncryptionInfos, GetDistribution, ScratchArenaTakeCore,
    dist::Distribution,
    layouts::{
        GLWE, GLWEInfos, GLWEPlaintext, GLWEPlaintextToRef, GLWEToMut, LWEInfos,
        prepared::{GLWEPreparedToRef, GLWESecretPreparedToBackendRef},
    },
};

pub(crate) fn normalize_scratch_vec_znx<'a, BE: Backend + 'a>(
    module: &Module<BE>,
    base2k: usize,
    vec: &mut VecZnx<BE::BufMut<'a>>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    Module<BE>: VecZnxNormalizeInplaceBackend<BE>,
{
    scratch.scope(|mut scratch| {
        let mut vec_mut = vec_znx_backend_mut_from_mut::<BE>(vec);
        <Module<BE> as VecZnxNormalizeInplaceBackend<BE>>::vec_znx_normalize_inplace_backend(
            module,
            base2k,
            &mut vec_mut,
            0,
            &mut scratch,
        );
    });
}

#[doc(hidden)]
pub trait GLWEEncryptSkDefault<BE: Backend> {
    fn glwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_encrypt_sk<'s, R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToMut,
        P: GLWEPlaintextToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn glwe_encrypt_zero_sk<'s, R, E, S>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToMut,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;
}

impl<BE: Backend> GLWEEncryptSkDefault<BE> for Module<BE>
where
    Self: Sized + ModuleN + VecZnxNormalizeTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxDftBytesOf + GLWEEncryptSkInternal<BE>,
{
    fn glwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let size: usize = infos.size();
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = VecZnx::bytes_of(self.n(), 1, size).max(self.vec_znx_normalize_tmp_bytes());
        let lvl_1: usize = VecZnx::bytes_of(self.n(), 1, size);
        let lvl_2: usize = self.bytes_of_vec_znx_dft(1, size);
        let lvl_3: usize = self.vec_znx_normalize_tmp_bytes().max(self.vec_znx_big_normalize_tmp_bytes());

        lvl_0 + lvl_1 + lvl_2 + lvl_3
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_encrypt_sk<'s, R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToMut,
        P: GLWEPlaintextToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let pt: &GLWEPlaintext<&[u8]> = &pt.to_ref();
        let sk_ref = sk.to_backend_ref();

        assert_eq!(res.rank(), sk_ref.rank());
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(sk_ref.n(), self.n() as u32);
        assert_eq!(pt.n(), self.n() as u32);
        assert!(
            scratch.available() >= <Module<BE> as GLWEEncryptSkDefault<BE>>::glwe_encrypt_sk_tmp_bytes(self, res),
            "scratch.available(): {} < GLWE::encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            <Module<BE> as GLWEEncryptSkDefault<BE>>::glwe_encrypt_sk_tmp_bytes(self, res)
        );

        let cols: usize = (res.rank() + 1).into();
        self.glwe_encrypt_sk_internal(
            res.base2k().into(),
            res.data_mut(),
            cols,
            false,
            Some((pt, 0)),
            sk,
            enc_infos,
            source_xe,
            source_xa,
            scratch,
        );
    }

    fn glwe_encrypt_zero_sk<'s, R, E, S>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToMut,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let sk_ref = sk.to_backend_ref();

        assert_eq!(res.rank(), sk_ref.rank());
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(sk_ref.n(), self.n() as u32);
        assert!(
            scratch.available() >= <Module<BE> as GLWEEncryptSkDefault<BE>>::glwe_encrypt_sk_tmp_bytes(self, res),
            "scratch.available(): {} < GLWE::encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            <Module<BE> as GLWEEncryptSkDefault<BE>>::glwe_encrypt_sk_tmp_bytes(self, res)
        );

        let cols: usize = (res.rank() + 1).into();
        self.glwe_encrypt_sk_internal(
            res.base2k().into(),
            res.data_mut(),
            cols,
            false,
            None::<(&GLWEPlaintext<Vec<u8>>, usize)>,
            sk,
            enc_infos,
            source_xe,
            source_xa,
            scratch,
        );
    }
}

#[doc(hidden)]
pub trait GLWEEncryptPkDefault<BE: Backend> {
    fn glwe_encrypt_pk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_encrypt_pk<'s, R, P, K, E>(
        &self,
        res: &mut R,
        pt: &P,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        P: GLWEPlaintextToRef + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToRef<BE> + GetDistribution + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut;

    fn glwe_encrypt_zero_pk<'s, R, K, E>(
        &self,
        res: &mut R,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToRef<BE> + GetDistribution + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut;
}

impl<BE: Backend> GLWEEncryptPkDefault<BE> for Module<BE>
where
    Self: GLWEEncryptPkInternal<BE> + VecZnxDftBytesOf + SvpPPolBytesOf + VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes,
{
    fn glwe_encrypt_pk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let size: usize = infos.size();
        let cols: usize = (infos.rank() + 1).into();
        assert_eq!(self.n() as u32, infos.n());
        let lvl_0: usize = self.bytes_of_svp_ppol(1);
        let lvl_1: usize = ScalarZnx::bytes_of(self.n(), 1);
        let lvl_2: usize = cols * (self.bytes_of_vec_znx_dft(1, size) + VecZnx::bytes_of(self.n(), 1, size));
        let lvl_3: usize = self.vec_znx_big_normalize_tmp_bytes();

        lvl_0 + lvl_1 + lvl_2 + lvl_3
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_encrypt_pk<'s, R, P, K, E>(
        &self,
        res: &mut R,
        pt: &P,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        P: GLWEPlaintextToRef + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToRef<BE> + GetDistribution + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        assert!(
            scratch.available() >= <Module<BE> as GLWEEncryptPkDefault<BE>>::glwe_encrypt_pk_tmp_bytes(self, res),
            "scratch.available(): {} < GLWEEncryptPk::glwe_encrypt_pk_tmp_bytes: {}",
            scratch.available(),
            <Module<BE> as GLWEEncryptPkDefault<BE>>::glwe_encrypt_pk_tmp_bytes(self, res)
        );
        self.glwe_encrypt_pk_internal(res, Some((pt, 0)), pk, enc_infos, source_xu, source_xe, scratch)
    }

    fn glwe_encrypt_zero_pk<'s, R, K, E>(
        &self,
        res: &mut R,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToRef<BE> + GetDistribution + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        assert!(
            scratch.available() >= <Module<BE> as GLWEEncryptPkDefault<BE>>::glwe_encrypt_pk_tmp_bytes(self, res),
            "scratch.available(): {} < GLWEEncryptPk::glwe_encrypt_pk_tmp_bytes: {}",
            scratch.available(),
            <Module<BE> as GLWEEncryptPkDefault<BE>>::glwe_encrypt_pk_tmp_bytes(self, res)
        );
        self.glwe_encrypt_pk_internal(
            res,
            None::<(&GLWEPlaintext<Vec<u8>>, usize)>,
            pk,
            enc_infos,
            source_xu,
            source_xe,
            scratch,
        )
    }
}

pub(crate) trait GLWEEncryptPkInternal<BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn glwe_encrypt_pk_internal<'s, R, P, K, E>(
        &self,
        res: &mut R,
        pt: Option<(&P, usize)>,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToMut,
        P: GLWEPlaintextToRef + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToRef<BE> + GetDistribution + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut;
}

impl<BE: Backend> GLWEEncryptPkInternal<BE> for Module<BE>
where
    Self: SvpPrepare<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigAddNormal<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxAddAssign
        + VecZnxCopy
        + SvpPPolBytesOf
        + ModuleN
        + VecZnxDftBytesOf,
{
    #[allow(clippy::too_many_arguments)]
    fn glwe_encrypt_pk_internal<'s, R, P, K, E>(
        &self,
        res: &mut R,
        pt: Option<(&P, usize)>,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToMut,
        E: EncryptionInfos,
        P: GLWEPlaintextToRef + GLWEInfos,
        K: GLWEPreparedToRef<BE> + GetDistribution + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

        assert_eq!(res.base2k(), pk.base2k());
        assert_eq!(res.n(), pk.n());
        assert_eq!(res.rank(), pk.rank());
        if let Some((pt, _)) = pt {
            assert_eq!(pt.base2k(), pk.base2k());
            assert_eq!(pt.n(), pk.n());
        }

        let base2k: usize = pk.base2k().into();
        let size_pk: usize = pk.size();
        let cols: usize = (res.rank() + 1).into();

        // Generates u according to the underlying secret distribution.
        let scratch = scratch.borrow();
        let (mut u_dft, mut scratch_1) = scratch.take_svp_ppol(self, 1);

        {
            let (mut u, scratch_2) = scratch_1.take_scalar_znx(self.n(), 1);
            match pk.dist() {
                Distribution::NONE => panic!(
                    "invalid public key: SecretDistribution::NONE, ensure it has been correctly intialized through \
                     Self::generate"
                ),
                Distribution::TernaryFixed(hw) => u.fill_ternary_hw(0, *hw, source_xu),
                Distribution::TernaryProb(prob) => u.fill_ternary_prob(0, *prob, source_xu),
                Distribution::BinaryFixed(hw) => u.fill_binary_hw(0, *hw, source_xu),
                Distribution::BinaryProb(prob) => u.fill_binary_prob(0, *prob, source_xu),
                Distribution::BinaryBlock(block_size) => u.fill_binary_block(0, *block_size, source_xu),
                Distribution::ZERO => {}
            }

            self.svp_prepare(&mut u_dft, 0, &u, 0);
            scratch_1 = scratch_2;
        }

        {
            let pk = pk.to_ref();

            // ct[i] = pk[i] * u + ei (+ m if col = i)
            for i in 0..cols {
                let (mut ci_dft, scratch_2) = scratch_1.take_vec_znx_dft(self, 1, size_pk);
                // ci_dft = DFT(u) * DFT(pk[i])
                let u_dft_ref = svp_ppol_backend_ref_from_mut::<BE>(&u_dft);
                self.svp_apply_dft_to_dft(&mut ci_dft, 0, &u_dft_ref, 0, &pk.data, i);

                // ci_big = u * p[i]
                let mut ci_big = self.vec_znx_idft_apply_consume(ci_dft);

                // ci_big = u * pk[i] + e
                self.vec_znx_big_add_normal(base2k, &mut ci_big, 0, enc_infos.noise_infos(), source_xe);

                let (mut ci, scratch_3) = scratch_2.take_vec_znx(self.n(), 1, size_pk);
                let scratch_next = {
                    let ci_big_ref = vec_znx_big_backend_ref_from_mut::<BE>(&ci_big);
                    scratch_3
                        .apply_mut(|scratch| self.vec_znx_big_normalize(&mut ci, base2k, 0, 0, &ci_big_ref, base2k, 0, scratch))
                };
                scratch_1 = scratch_next;

                if let Some((pt, col)) = pt
                    && col == i
                {
                    self.vec_znx_add_assign(&mut ci, 0, &pt.to_ref().data, 0);
                }

                self.vec_znx_copy(&mut res.data, i, &ci, 0);
            }
        }
    }
}

pub(crate) trait GLWEEncryptSkInternal<BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn glwe_encrypt_sk_internal<'s, R, P, S, E>(
        &self,
        base2k: usize,
        res: &mut R,
        cols: usize,
        compressed: bool,
        pt: Option<(&P, usize)>,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: VecZnxToMut,
        P: GLWEPlaintextToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;
}

impl<BE: Backend> GLWEEncryptSkInternal<BE> for Module<BE>
where
    Self: ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftAssign<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxFillUniform
        + VecZnxSubAssign
        + VecZnxAddAssign
        + VecZnxNormalizeInplaceBackend<BE>
        + VecZnxAddNormal
        + VecZnxNormalize<BE>
        + VecZnxSub
        + VecZnxCopy
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    BE::OwnedBuf: HostDataMut,
{
    fn glwe_encrypt_sk_internal<'s, R, P, S, E>(
        &self,
        base2k: usize,
        res: &mut R,
        cols: usize,
        compressed: bool,
        pt: Option<(&P, usize)>,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        _scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: VecZnxToMut,
        P: GLWEPlaintextToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let ct: &mut VecZnx<&mut [u8]> = &mut res.to_mut();
        let sk = sk.to_backend_ref();

        if compressed {
            assert_eq!(ct.cols(), 1, "invalid glwe: compressed tag=true but #cols={} != 1", ct.cols())
        }

        assert!(
            sk.dist != Distribution::NONE,
            "glwe secret distribution is NONE (have you prepared the key?)"
        );

        let size: usize = ct.size();

        let c0_data: BE::OwnedBuf = BE::alloc_bytes(VecZnx::<Vec<u8>>::bytes_of(self.n(), 1, size));
        let ci_data: BE::OwnedBuf = BE::alloc_bytes(VecZnx::<Vec<u8>>::bytes_of(self.n(), 1, size));
        let mut c0: VecZnx<BE::OwnedBuf> = VecZnx::from_data_with_max_size(c0_data, self.n(), 1, size, size);
        let mut ci: VecZnx<BE::OwnedBuf> = VecZnx::from_data_with_max_size(ci_data, self.n(), 1, size, size);
        let mut normalize_scratch_owned: ScratchOwned<BE> = ScratchOwned::alloc(self.vec_znx_normalize_tmp_bytes());
        let mut big_normalize_scratch_owned: ScratchOwned<BE> = ScratchOwned::alloc(self.vec_znx_big_normalize_tmp_bytes());
        c0.zero();

        for i in 1..cols {
            let col_ct: usize = if compressed { 0 } else { i };
            self.vec_znx_fill_uniform(base2k, ct, col_ct, source_xa);

            if let Some((pt, col)) = pt
                && i == col
            {
                self.vec_znx_sub(&mut ci, 0, ct, col_ct, &pt.to_ref().data, 0);
                {
                    let mut scratch_norm = normalize_scratch_owned.arena();
                    let mut ci_mut = <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut ci);
                    <Module<BE> as VecZnxNormalizeInplaceBackend<BE>>::vec_znx_normalize_inplace_backend(
                        self,
                        base2k,
                        &mut ci_mut,
                        0,
                        &mut scratch_norm,
                    );
                }
            } else {
                self.vec_znx_copy(&mut ci, 0, ct, col_ct);
            }

            {
                let ci_dft_data: BE::OwnedBuf = BE::alloc_bytes(self.bytes_of_vec_znx_dft(1, size));
                let mut ci_dft: VecZnxDft<BE::OwnedBuf, BE> = VecZnxDft::from_data(ci_dft_data, self.n(), 1, size);
                {
                    let ci_ref = <VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(&ci);
                    let mut ci_dft_mut = ci_dft.to_backend_mut();
                    <Module<BE> as VecZnxDftApply<BE>>::vec_znx_dft_apply(self, 1, 0, &mut ci_dft_mut, 0, &ci_ref, 0);
                }

                self.svp_apply_dft_to_dft_inplace(&mut ci_dft, 0, &sk.data, i - 1);
                let ci_big = self.vec_znx_idft_apply_consume(ci_dft);
                {
                    let ci_big_ref = ci_big.to_backend_ref();
                    let mut ci_mut = <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut ci);
                    let mut scratch_norm = big_normalize_scratch_owned.arena();
                    <Module<BE> as VecZnxBigNormalize<BE>>::vec_znx_big_normalize(
                        self,
                        &mut ci_mut,
                        base2k,
                        0,
                        0,
                        &ci_big_ref,
                        base2k,
                        0,
                        &mut scratch_norm,
                    );
                }
            }

            self.vec_znx_sub_inplace(&mut c0, 0, &ci, 0);
        }

        // c[0] += e
        self.vec_znx_add_normal(base2k, &mut c0, 0, enc_infos.noise_infos(), source_xe);

        // c[0] += m if col = 0
        if let Some((pt, col)) = pt
            && col == 0
        {
            self.vec_znx_add_assign(&mut c0, 0, &pt.to_ref().data, 0);
        }

        {
            let mut scratch_norm = normalize_scratch_owned.arena();
            let mut c0_mut = <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut c0);
            <Module<BE> as VecZnxNormalizeInplaceBackend<BE>>::vec_znx_normalize_inplace_backend(
                self,
                base2k,
                &mut c0_mut,
                0,
                &mut scratch_norm,
            );
        }
        self.vec_znx_copy(ct, 0, &c0, 0);
    }
}
