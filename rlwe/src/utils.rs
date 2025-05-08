pub(crate) fn derive_size(log_base2k: usize, log_k: usize) -> usize {
    (log_k + log_base2k - 1) / log_base2k
}
