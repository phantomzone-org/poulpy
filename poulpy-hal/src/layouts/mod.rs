mod convolution;
mod encoding;
mod mat_znx;
mod module;
mod scalar_znx;
mod scratch;
mod serialization;
mod stats;
mod svp_ppol;
mod vec_znx;
mod vec_znx_big;
mod vec_znx_dft;
mod vmp_pmat;
mod znx_base;

pub use convolution::*;
pub use mat_znx::*;
pub use module::*;
pub use scalar_znx::*;
pub use scratch::*;
pub use serialization::*;
pub use stats::*;
pub use svp_ppol::*;
pub use vec_znx::*;
pub use vec_znx_big::*;
pub use vec_znx_dft::*;
pub use vmp_pmat::*;
pub use znx_base::*;

pub trait Data = PartialEq + Eq + Sized + Default;
pub trait DataRef = Data + AsRef<[u8]> + Sync;
pub trait DataMut = DataRef + AsMut<[u8]> + Send;

pub trait ToOwnedDeep {
    type Owned;
    fn to_owned_deep(&self) -> Self::Owned;
}

pub trait DigestU64 {
    fn digest_u64(&self) -> u64;
}
