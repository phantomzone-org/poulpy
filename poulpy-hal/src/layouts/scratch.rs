use std::marker::PhantomData;

use crate::layouts::Backend;

pub struct ScratchOwned<B: Backend> {
    pub data: Vec<u8>,
    pub _phantom: PhantomData<B>,
}

pub struct Scratch<B: Backend> {
    pub _phantom: PhantomData<B>,
    pub data: [u8],
}
