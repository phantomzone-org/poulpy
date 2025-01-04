pub mod poly;
use std::cmp::PartialEq;

#[derive(Clone, Debug, Eq)]
pub struct Poly<O>(pub Vec<O>);

impl<O> Poly<O>where
    O: Default + Clone + Copy,
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
        (usize::BITS - self.0.len().leading_zeros()) as usize
    }

    pub fn log_n(&self) -> usize{
        self.0.len()-1
    }

    pub fn resize(&mut self, n:usize){
        self.0.resize(n, O::default());
    }

    pub fn set_all(&mut self, v: &O){
        self.0.fill(*v)
    }

    pub fn zero(&mut self){
        self.set_all(&O::default())
    }

    pub fn copy_from(&mut self, other: &Poly<O>){
        if std::ptr::eq(self, other){
            return
        }
        self.resize(other.n());
        self.0.copy_from_slice(&other.0)
    }
}

impl<O: PartialEq> PartialEq for Poly<O> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other) || (self.0 == other.0)
    }
}

#[derive(Clone, Debug, Eq)]
pub struct PolyRNS<O>(pub Vec<Poly<O>>);

impl<O> PolyRNS<O>where
    O: Default + Clone + Copy,
    {

    pub fn new(n: usize, level: usize) -> Self{
        let mut polyrns: PolyRNS<O> = PolyRNS::<O>::default();
        let mut buf: Vec<O> = vec![O::default();polyrns.buffer_size(n, level)];
        polyrns.from_buffer(n, level, &mut buf[..]);
        polyrns
    }

    pub fn n(&self) -> usize{
        self.0[0].n()
    }

    pub fn log_n(&self) -> usize{
        self.0[0].log_n()
    }

    pub fn level(&self) -> usize{
        self.0.len()-1
    }

    pub fn buffer_size(&self, n: usize, level:usize) -> usize{
        n * (level+1)
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

    pub fn resize(&mut self, level:usize){
        self.0.resize(level+1, Poly::<O>::new(self.n()));
    }

    pub fn split_at_mut(&mut self, level:usize) -> (&mut [Poly<O>], &mut [Poly<O>]){
        self.0.split_at_mut(level)
    }

    pub fn at(&self, level:usize) -> &Poly<O>{
        assert!(level <= self.level(), "invalid argument level: level={} > self.level()={}", level, self.level());
        &self.0[level]
    }

    pub fn at_mut(&mut self, level:usize) -> &mut Poly<O>{
        &mut self.0[level]
    }

    pub fn set_all(&mut self, v: &O){
        (0..self.level()+1).for_each(|i| self.at_mut(i).set_all(v))
    }

    pub fn zero(&mut self){
        self.set_all(&O::default())
    }

    pub fn copy(&mut self, other: &PolyRNS<O>){
        if std::ptr::eq(self, other){
            return
        }
        self.resize(other.level());
        self.copy_level(other.level(), other);
    }

    pub fn copy_level(&mut self, level:usize, other: &PolyRNS<O>){
        assert!(self.level() <= level, "invalid argument level: level={} > self.level()={}", level, self.level());
        assert!(other.level() <= level, "invalid argument level: level={} > other.level()={}", level, other.level());
        (0..level+1).for_each(|i| self.at_mut(i).copy_from(other.at(i)))
    }
}

impl<O: PartialEq> PartialEq for PolyRNS<O> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other) && (self.0 == other.0)
    }
}

impl<O> Default for PolyRNS<O>{
    fn default() -> Self{
        let polys:Vec<Poly<O>> = Vec::new();
        Self{0:polys}
    }
}