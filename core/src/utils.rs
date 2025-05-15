pub(crate) fn derive_size(basek: usize, k: usize) -> usize {
    (k + basek - 1) / basek
}
