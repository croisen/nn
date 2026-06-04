use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

#[derive(Default, Debug, Clone, Copy)]
pub struct HEUniform {
    f_in: usize,
    std: StandardUniform,
}

impl HEUniform {
    pub fn new(f_in: usize, _f_out: usize) -> Self {
        Self {
            f_in,
            std: StandardUniform::default(),
        }
    }
}

impl Distribution<f64> for HEUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let r: f64 = self.std.sample(rng);
        let r = r - 0.5;
        let m = (6.0 / self.f_in as f64).sqrt();
        let r = r % m;
        r
    }
}
