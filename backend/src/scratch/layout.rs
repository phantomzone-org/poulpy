use std::marker::PhantomData;

use crate::Backend;

pub struct ScratchOwned<B: Backend> {
    pub(crate) data: Vec<u8>,
    pub(crate) _phantom: PhantomData<B>,
}

pub struct Scratch<B: Backend> {
    pub(crate) _phantom: PhantomData<B>,
    pub(crate) data: [u8],
}
