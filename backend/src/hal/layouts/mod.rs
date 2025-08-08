mod mat_znx;
mod module;
mod scalar_znx;
mod scratch;
mod serialization;
mod svp_ppol;
mod vec_znx;
mod vec_znx_big;
mod vec_znx_dft;
mod vmp_pmat;

pub use mat_znx::*;
pub use module::*;
use sampling::source::Source;
pub use scalar_znx::*;
pub use scratch::*;
pub use serialization::*;
pub use svp_ppol::*;
pub use vec_znx::*;
pub use vec_znx_big::*;
pub use vec_znx_dft::*;
pub use vmp_pmat::*;

pub trait Data = PartialEq + Eq + Sized;
pub trait DataRef = Data + AsRef<[u8]>;
pub trait DataMut = DataRef + AsMut<[u8]>;

pub trait FillUniform {
    fn fill_uniform(&mut self, source: &mut Source);
}
