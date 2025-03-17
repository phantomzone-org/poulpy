pub trait Infos {
    /// Returns the ring degree of the receiver.
    fn n(&self) -> usize;

    /// Returns the base two logarithm of the ring dimension of the receiver.
    fn log_n(&self) -> usize;

    /// Returns the number of columns of the receiver.
    /// This method is equivalent to [Infos::cols].
    fn cols(&self) -> usize;

    /// Returns the number of rows of the receiver.
    fn rows(&self) -> usize;
}
