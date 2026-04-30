use poulpy_hal::{
    api::{
        ModuleN, VecZnxAddScalarAssignBackend, VecZnxDftBytesOf, VecZnxNormalizeAssignBackend, VecZnxNormalizeTmpBytes,
        VecZnxZeroBackend,
    },
    layouts::{Backend, Module, ScalarZnxToBackendRef, ScratchArena, VecZnxReborrowBackendMut, VecZnxReborrowBackendRef},
    source::Source,
};

use crate::{
    EncryptionInfos, GLWEEncryptSk, GLWEEncryptSkInternal, ScratchArenaTakeCore,
    encryption::glwe::normalize_scratch_vec_znx,
    layouts::{
        GGSWInfos, GGSWToBackendMut, GLWEInfos, GLWEPlaintext, LWEInfos, ggsw_at_backend_mut_from_mut,
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
        R: GGSWToBackendMut<BE>,
        P: ScalarZnxToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;
}

impl<BE: Backend> GGSWEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN
        + GLWEEncryptSkInternal<BE>
        + GLWEEncryptSk<BE>
        + VecZnxDftBytesOf
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxAddScalarAssignBackend<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxZeroBackend<BE>,
{
    fn ggsw_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(infos);
        lvl_0 + self.glwe_encrypt_sk_tmp_bytes(infos).max(self.vec_znx_normalize_tmp_bytes())
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
        R: GGSWToBackendMut<BE>,
        P: ScalarZnxToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let res = &mut res.to_backend_mut();
        let pt_backend = pt.to_backend_ref();
        let sk_ref = sk.to_backend_ref();

        assert_eq!(res.rank(), sk_ref.rank());
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(pt_backend.n(), self.n());
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
            self.vec_znx_zero_backend(&mut tmp_pt.data, 0);
            // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
            {
                let mut tmp_pt_data =
                    <poulpy_hal::layouts::VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendMut<BE>>::reborrow_backend_mut(
                        &mut tmp_pt.data,
                    );
                self.vec_znx_add_scalar_assign_backend(&mut tmp_pt_data, 0, (dsize - 1) + row_i * dsize, &pt_backend, 0);
            }
            scratch_1.scope(|mut scratch| {
                normalize_scratch_vec_znx(self, base2k, &mut tmp_pt.data, &mut scratch);
            });
            for col_j in 0..rank + 1 {
                {
                    let mut scratch = scratch_1.borrow();
                    let tmp_pt_backend = GLWEPlaintext {
                        data: <poulpy_hal::layouts::VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(
                            &tmp_pt.data,
                        ),
                        base2k: tmp_pt.base2k,
                    };
                    let mut ct = ggsw_at_backend_mut_from_mut::<BE>(res, row_i, col_j);
                    <Module<BE> as GLWEEncryptSkInternal<BE>>::glwe_encrypt_sk_internal(
                        self,
                        base2k,
                        &mut ct.data,
                        cols,
                        false,
                        Some((tmp_pt_backend, col_j)),
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
