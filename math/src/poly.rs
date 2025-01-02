pub mod poly;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly<O>(pub Vec<O>);

impl<O> Poly<O>where
    O: Default + Clone,
    {
    pub fn new(n: usize) -> Self{
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolyRNS<O>(pub Vec<Poly<O>>);

impl<O> PolyRNS<O>where
    O: Default + Clone,
    {
    pub fn new(n: usize) -> Self{
        Self(vec![Poly::<O>::new(n);n])
    }

    pub fn n(&self) -> usize{
        self.0[0].n()
    }

    pub fn level(&self) -> usize{
        self.0.len()-1
    }

    pub fn buffer_size(&self) -> usize{
        self.n() * (self.level()+1)
    }

    pub fn from_buffer(&mut self, n: usize, level: usize, buf: &mut [O]){
        assert!(buf.len() >= n * (level+1), "invalid buffer: buf.len()={} < n * (level+1)={}", buf.len(), level+1);
        self.0.clear();
        for chunk in buf.chunks_mut(n).take(level+1) {
            let mut poly: Poly<O> = Poly(Vec::new());
            poly.from_buffer(n, chunk);
            self.0.push(poly);
        }
    }
}