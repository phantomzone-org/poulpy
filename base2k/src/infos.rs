use crate::{VecZnx, VecZnxBorrow, VmpPMat};

pub trait Infos {
    /// Returns the ring degree of the receiver.
    fn n(&self) -> usize;

    /// Returns the base two logarithm of the ring dimension of the receiver.
    fn log_n(&self) -> usize;

    /// Returns the number of limbs of the receiver.
    /// This method is equivalent to [Infos::cols].
    fn limbs(&self) -> usize;

    /// Returns the number of columns of the receiver.
    /// This method is equivalent to [Infos::limbs].
    fn cols(&self) -> usize;

    /// Returns the number of rows of the receiver.
    fn rows(&self) -> usize;
}

impl Infos for VecZnx {
    /// Returns the base 2 logarithm of the [VecZnx] degree.
    fn log_n(&self) -> usize {
        (usize::BITS - (self.n - 1).leading_zeros()) as _
    }

    /// Returns the [VecZnx] degree.
    fn n(&self) -> usize {
        self.n
    }

    /// Returns the number of limbs of the [VecZnx].
    fn limbs(&self) -> usize {
        self.data.len() / self.n
    }

    /// Returns the number of limbs of the [VecZnx].
    fn cols(&self) -> usize {
        self.data.len() / self.n
    }

    /// Returns the number of limbs of the [VecZnx].
    fn rows(&self) -> usize {
        1
    }
}

impl Infos for VecZnxBorrow {
    /// Returns the base 2 logarithm of the [VecZnx] degree.
    fn log_n(&self) -> usize {
        (usize::BITS - (self.n - 1).leading_zeros()) as _
    }

    /// Returns the [VecZnx] degree.
    fn n(&self) -> usize {
        self.n
    }

    /// Returns the number of limbs of the [VecZnx].
    fn limbs(&self) -> usize {
        self.limbs
    }

    /// Returns the number of limbs of the [VecZnx].
    fn cols(&self) -> usize {
        self.limbs
    }

    /// Returns the number of limbs of the [VecZnx].
    fn rows(&self) -> usize {
        1
    }
}

impl Infos for VmpPMat {
    /// Returns the ring dimension of the [VmpPMat].
    fn n(&self) -> usize {
        self.n
    }

    fn log_n(&self) -> usize {
        (usize::BITS - (self.n() - 1).leading_zeros()) as _
    }

    /// Returns the number of limbs of each [VecZnxDft].
    /// This method is equivalent to [Self::cols].
    fn limbs(&self) -> usize {
        self.cols
    }

    /// Returns the number of rows (i.e. of [VecZnxDft]) of the [VmpPMat]
    fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of cols of the [VmpPMat].
    /// The number of cols refers to the number of limbs  
    /// of each [VecZnxDft].
    /// This method is equivalent to [Self::limbs].
    fn cols(&self) -> usize {
        self.cols
    }
}
