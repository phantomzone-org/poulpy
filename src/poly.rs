pub mod poly;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly<O>(pub Vec<O>);

impl<O> Poly<O>where
    O: Default + Clone,
    {
    pub fn new(n: usize) -> Self{
        Self(vec![O::default();n])
    }

    pub fn new_montgomery(n: usize) -> Self{
        Self(vec![O::default();n])
    }

    pub fn buffer_size(&self) -> usize{
        return self.0.len()
    }

    pub fn from_buffer(&mut self, n: usize, buf: &mut [O]){
        assert!(buf.len() >= n, "invalid buffer: buf.len()={} < n={}", buf.len(), n);
        self.0 = Vec::from(&buf[..n]);
    }

    pub fn n(&self) -> usize{
        return self.0.len()
    }
}