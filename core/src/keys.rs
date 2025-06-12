#[derive(Clone, Copy, Debug)]
pub(crate) enum SecretDistribution {
    TernaryFixed(usize), // Ternary with fixed Hamming weight
    TernaryProb(f64),    // Ternary with probabilistic Hamming weight
    BinaryFixed(usize),  // Binary with fixed Hamming weight
    BinaryProb(f64),     // Binary with probabilistic Hamming weight
    BinaryBlock(usize),  // Binary split in block of size 2^k
    ZERO,                // Debug mod
    NONE,                // Unitialized
}
