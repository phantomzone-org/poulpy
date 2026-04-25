use poulpy_hal::{
    api::{ModuleN, VecZnxAddScalarAssign, VecZnxDftBytesOf, VecZnxNormalizeInplaceBackend, VecZnxNormalizeTmpBytes},
    layouts::{
        Backend, HostDataMut, Module, ScalarZnx, ScalarZnxToRef, ScratchArena, VecZnxReborrowBackendRef, ZnxInfos, ZnxZero,
    },
    source::Source,
};

use crate::{
    EncryptionInfos, GGSWNoise, GLWEEncryptSk, GLWEEncryptSkInternal, ScratchArenaTakeCore,
    encryption::glwe::normalize_scratch_vec_znx,
    layouts::{
        GGSW, GGSWInfos, GGSWToMut, GLWEInfos, GLWEPlaintext, GLWEPlaintextToRef, LWEInfos,
        prepared::GLWESecretPreparedToBackendRef,
    },
};

#[doc(hidden)]
pub trait GGSWEncryptSkDefault<BE: Backend> {
    fn ggsw_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_encrypt_sk<'s, R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGSWToMut,
        P: ScalarZnxToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut;
}

impl<BE: Backend> GGSWEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN
        + GLWEEncryptSkInternal<BE>
        + GLWEEncryptSk<BE>
        + GGSWNoise<BE>
        + VecZnxDftBytesOf
        + VecZnxNormalizeInplaceBackend<BE>
        + VecZnxAddScalarAssign
        + VecZnxNormalizeTmpBytes,
{
    fn ggsw_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(infos);
        let lvl_encrypt: usize = lvl_0 + self.glwe_encrypt_sk_tmp_bytes(infos).max(self.vec_znx_normalize_tmp_bytes());
        lvl_encrypt.max(self.ggsw_noise_tmp_bytes(infos))
    }

    #[allow(clippy::too_many_arguments)]
    fn ggsw_encrypt_sk<'s, R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGSWToMut,
        P: ScalarZnxToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let pt: &ScalarZnx<&[u8]> = &pt.to_ref();
        let sk_ref = sk.to_backend_ref();

        assert_eq!(res.rank(), sk_ref.rank());
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(pt.n(), self.n());
        assert_eq!(sk_ref.n(), self.n() as u32);
        assert!(
            scratch.available() >= <Module<BE> as GGSWEncryptSkDefault<BE>>::ggsw_encrypt_sk_tmp_bytes(self, res),
            "scratch.available(): {} < GGSWEncryptSk::ggsw_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            <Module<BE> as GGSWEncryptSkDefault<BE>>::ggsw_encrypt_sk_tmp_bytes(self, res)
        );

        let base2k: usize = res.base2k().into();
        let rank: usize = res.rank().into();
        let dsize: usize = res.dsize().into();
        let cols: usize = rank + 1;
        let scratch = scratch.borrow();
        let (mut tmp_pt, mut scratch_1) = scratch.take_glwe_plaintext(res);

        for row_i in 0..res.dnum().into() {
            tmp_pt.data.zero();
            // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
            self.vec_znx_add_scalar_assign(&mut tmp_pt.data, 0, (dsize - 1) + row_i * dsize, pt, 0);
            scratch_1.scope(|mut scratch| {
                normalize_scratch_vec_znx(self, base2k, &mut tmp_pt.data, &mut scratch);
            });
            for col_j in 0..rank + 1 {
                {
                    let mut scratch = scratch_1.borrow();
                    let tmp_pt_ref = tmp_pt.to_ref();
                    let tmp_pt_backend = GLWEPlaintext {
                        data: <poulpy_hal::layouts::VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(
                            &tmp_pt.data,
                        ),
                        base2k: tmp_pt.base2k,
                    };
                    <Module<BE> as GLWEEncryptSkInternal<BE>>::glwe_encrypt_sk_internal(
                        self,
                        base2k,
                        res.at_mut(row_i, col_j).data_mut(),
                        cols,
                        false,
                        Some((tmp_pt_ref, tmp_pt_backend, col_j)),
                        sk,
                        enc_infos,
                        source_xe,
                        source_xa,
                        &mut scratch,
                    );
                }
            }
        }
    }
}
