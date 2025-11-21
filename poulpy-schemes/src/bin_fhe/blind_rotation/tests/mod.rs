#[cfg(test)]
#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
mod fft64_ref;

#[cfg(test)]
#[cfg(all(feature = "enable-avx", target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
mod fft64_avx;

#[cfg(test)]
mod test_suite;
