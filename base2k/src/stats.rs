use crate::{Encoding, Infos, VecZnx};
use rug::Float;
use rug::float::Round;
use rug::ops::{AddAssignRound, DivAssignRound, SubAssignRound};

impl VecZnx {
    pub fn std(&self, poly_idx: usize, log_base2k: usize) -> f64 {
        let prec: u32 = (self.cols() * log_base2k) as u32;
        let mut data: Vec<Float> = (0..self.n()).map(|_| Float::with_val(prec, 0)).collect();
        self.decode_vec_float(poly_idx, log_base2k, &mut data);
        // std = sqrt(sum((xi - avg)^2) / n)
        let mut avg: Float = Float::with_val(prec, 0);
        data.iter().for_each(|x| {
            avg.add_assign_round(x, Round::Nearest);
        });
        avg.div_assign_round(Float::with_val(prec, data.len()), Round::Nearest);
        data.iter_mut().for_each(|x| {
            x.sub_assign_round(&avg, Round::Nearest);
        });
        let mut std: Float = Float::with_val(prec, 0);
        data.iter().for_each(|x| std += x * x);
        std.div_assign_round(Float::with_val(prec, data.len()), Round::Nearest);
        std = std.sqrt();
        std.to_f64()
    }
}
