#[cfg(test)]
pub mod fft64_ref;

#[cfg(test)]
pub mod test_suite;

#[cfg(test)]
pub mod ntt120_ref;

#[cfg(test)]
pub mod ntt_ifma_ref;

#[cfg(test)]
#[cfg(all(
    feature = "enable-ifma",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512ifma",
    target_feature = "avx512vl"
))]
pub mod ntt_ifma;
