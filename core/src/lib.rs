use std::{collections::HashMap, sync::RwLock};

use once_cell::sync::OnceCell;

use backend::{Backend, Module, ModuleInfos, ModuleKey, BACKEND};

pub mod automorphism;
pub mod elem;
pub mod gglwe_ciphertext;
pub mod ggsw_ciphertext;
pub mod glwe_ciphertext;
pub mod glwe_ciphertext_fourier;
pub mod glwe_ops;
pub mod glwe_plaintext;
pub mod keys;
pub mod keyswitch_key;
pub mod tensor_key;
#[cfg(test)]
mod test_fft64;
pub mod trace;
mod utils;

pub(crate) const SIX_SIGMA: f64 = 6.0;

static MODULE_MAP: OnceCell<RwLock<HashMap<ModuleKey, ModuleInfos>>> = OnceCell::new();

pub fn insert_module<B: Backend>(n: usize){
    let lock: &RwLock<HashMap<(BACKEND, u64), ModuleInfos>> = MODULE_MAP.get_or_init(|| RwLock::new(HashMap::new()));
    let mut map: std::sync::RwLockWriteGuard<'_, HashMap<(BACKEND, u64), ModuleInfos>> = lock.write().expect("Failed to acquire write lock");
    match B::KIND {
        BACKEND::FFT64 => {map.insert((B::KIND, n as u64), ModuleInfos::fft64(n))},
        BACKEND::NTT120 => {map.insert((B::KIND, n as u64), ModuleInfos::ntt120(n))},
    };
}

pub(crate) fn get_module<B: Backend>(n: u64) -> &'static Module<B> {
    let map: &RwLock<HashMap<(BACKEND, u64), ModuleInfos>> = MODULE_MAP.get().expect("MODULE_MAP not initialized");
    let guard: std::sync::RwLockReadGuard<'_, HashMap<(BACKEND, u64), ModuleInfos>> = map.read().expect("RwLock poisoned");
    let info: &ModuleInfos = guard.get(&(B::KIND, n)).expect("Module not found");

    // SAFETY: We assume that MODULE_MAP is never mutated again,
    // and the reference inside has static lifetime
    B::extract(info).map(|r| unsafe { &*(r as *const Module<B>) }).expect("Type mismatch in ModuleInfos")
}