use rug::{
    Float,
    float::Round,
    ops::{AddAssignRound, DivAssignRound, SubAssignRound},
};

use crate::layouts::{DataRef, VecZnx, ZnxInfos};

impl<D: DataRef> VecZnx<D> {
    pub fn std(&self, basek: usize, col: usize) -> f64 {
        let prec: u32 = (self.size() * basek) as u32;
        let mut data: Vec<Float> = (0..self.n()).map(|_| Float::with_val(prec, 0)).collect();
        self.decode_vec_float(basek, col, &mut data);
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
