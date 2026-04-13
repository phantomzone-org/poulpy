use std::marker::PhantomData;

use crate::layouts::{Backend, DeviceBuf};

/// Owned scratch buffer for temporary workspace during polynomial operations.
///
/// Operations such as normalization, DFT, and vector-matrix products require
/// temporary scratch memory. `ScratchOwned` holds a backend-owned buffer that
/// can be borrowed as a [`Scratch`] reference.
///
/// The required size for each operation is obtained via the corresponding
/// `*_tmp_bytes` method on the API trait (e.g.
/// [`VecZnxNormalizeTmpBytes`](crate::api::VecZnxNormalizeTmpBytes)).
#[repr(C)]
pub struct ScratchOwned<B: Backend> {
    pub data: DeviceBuf<B>,
    pub _phantom: PhantomData<B>,
}

/// Borrowed scratch buffer (unsized).
///
/// `Scratch` is a dynamically sized type (DST) wrapping `[u8]`. It is
/// always used behind a mutable reference (`&mut Scratch<B>`) and
/// supports arena-style sub-allocation via [`split_at_mut`](Scratch::split_at_mut)
/// and the [`ScratchTakeBasic`](crate::api::ScratchTakeBasic) methods.
#[repr(C)]
pub struct Scratch<B: Backend> {
    pub _phantom: PhantomData<B>,
    pub data: [u8],
}
