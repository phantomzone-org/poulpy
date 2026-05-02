macro_rules! hal_impl_module_ntt126_ifma {
    () => {
        fn new(n: u64) -> Module<Self> {
            <Self as NTT126IfmaModuleDefaults<Self>>::module_new_default(n)
        }
    };
}
