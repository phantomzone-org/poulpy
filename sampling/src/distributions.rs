use rand_distr::{Distribution, Normal, Binomial};

pub enum Distributions{
    Binonial(Binomial),
    Normal(Normal<f64>),
    Ternary()
}