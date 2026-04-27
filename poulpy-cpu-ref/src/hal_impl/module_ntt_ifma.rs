macro_rules! hal_impl_module_ntt_ifma {
    () => {
        fn new(n: u64) -> Module<Self> {
            <Self as NTT120IfmaModuleDefaults<Self>>::module_new_default(n)
        }
    };
}
