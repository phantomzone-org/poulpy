pub mod poly;

pub struct Poly<O>(pub Vec<O>);

impl Poly<u64>{
    pub fn new(n: usize) -> Self{
        Self(vec![0u64;n])
    }

    pub fn buffer_size(&self) -> usize{
        return self.0.len()
    }

    pub fn from_buffer(&mut self, n: usize, buf: &mut [u64]){
        assert!(buf.len() >= n, "invalid buffer: buf.len()={} < n={}", buf.len(), n);
        self.0 = Vec::from(&buf[..n]);
    }

    pub fn n(&self) -> usize{
        return self.0.len()
    }
}