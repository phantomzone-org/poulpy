use poulpy_hal::{
    api::{SvpPPolAlloc, SvpPrepare},
    layouts::{Backend, DataMut, DataRef, Module, ScalarZnx, Scratch, SvpPPol},
};

use std::marker::PhantomData;

use poulpy_core::{
    Distribution,
    layouts::{GGSWPreparedFactory, LWEInfos, prepared::GGSWPrepared},
};

use crate::tfhe::blind_rotation::{
    BlindRotationKey, BlindRotationKeyInfos, BlindRotationKeyPrepared, BlindRotationKeyPreparedFactory, CGGI,
    utils::set_xai_plus_y,
};

impl<BE: Backend> BlindRotationKeyPreparedFactory<CGGI, BE> for Module<BE>
where
    Self: GGSWPreparedFactory<BE> + SvpPPolAlloc<BE> + SvpPrepare<BE>,
{
    fn blind_rotation_key_prepared_alloc<A>(&self, infos: &A) -> BlindRotationKeyPrepared<Vec<u8>, CGGI, BE>
    where
        A: BlindRotationKeyInfos,
    {
        BlindRotationKeyPrepared {
            data: (0..infos.n_lwe().as_usize())
                .map(|_| GGSWPrepared::alloc_from_infos(self, infos))
                .collect(),
            dist: Distribution::NONE,
            x_pow_a: None,
            _phantom: PhantomData,
        }
    }

    fn blind_rotation_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: BlindRotationKeyInfos,
    {
        self.ggsw_prepare_tmp_bytes(infos)
    }

    fn prepare_blind_rotation_key<DM, DR>(
        &self,
        res: &mut BlindRotationKeyPrepared<DM, CGGI, BE>,
        other: &BlindRotationKey<DR, CGGI>,
        scratch: &mut Scratch<BE>,
    ) where
        DM: DataMut,
        DR: DataRef,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(res.data.len(), other.keys.len());
        }

        let n: usize = other.n().as_usize();

        for (a, b) in res.data.iter_mut().zip(other.keys.iter()) {
            a.prepare(self, b, scratch);
        }

        res.dist = other.dist;

        if let Distribution::BinaryBlock(_) = other.dist {
            let mut x_pow_a: Vec<SvpPPol<Vec<u8>, BE>> = Vec::with_capacity(n << 1);
            let mut buf: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
            (0..n << 1).for_each(|i| {
                let mut res: SvpPPol<Vec<u8>, BE> = self.svp_ppol_alloc(1);
                set_xai_plus_y(self, i, 0, &mut res, &mut buf);
                x_pow_a.push(res);
            });
            res.x_pow_a = Some(x_pow_a);
        }
    }
}
