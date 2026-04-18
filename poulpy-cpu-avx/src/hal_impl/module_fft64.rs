macro_rules! hal_impl_module_fft64 {
    () => {
        fn new(n: u64) -> Module<Self> {
            <Self as FFT64ModuleDefaults<Self>>::module_new_default(n)
        }
    };
}
