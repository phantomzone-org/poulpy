macro_rules! hal_impl_module_ntt120_avx512 {
    () => {
        fn new(n: u64) -> Module<Self> {
            <Self as NTT120ModuleDefaults<Self>>::module_new_default(n)
        }
    };
}
