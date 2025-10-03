use poulpy_hal::layouts::{Backend, Scratch};

use crate::layouts::prepared::GGSWCiphertextPreparedToRef;

mod gglwe_atk;
mod gglwe_ksk;
mod ggsw_ct;
mod glwe_ct;

pub trait ExternalProduct<R, A, BE: Backend> {
    fn external_product<D>(&self, res: &mut R, a: &A, ggsw: &D, scratch: &mut Scratch<BE>)
    where
        D: GGSWCiphertextPreparedToRef<BE>;
}
