use rand::distr::Distribution;

#[derive(Default, Debug, Clone, Copy)]
pub struct HEUniform {
    f_in: usize,
    f_out: usize,
}

impl HEUniform {
    pub fn new(f_in: usize, f_out: usize) -> Self {
        Self { f_in, f_out }
    }
}

impl Distribution<f64> for HEUniform {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let r = rng.next_u32() as f64 / u32::MAX as f64;
        let min = -(6.0 / self.f_in as f64).sqrt();
        let max = -(6.0 / self.f_out as f64).sqrt();
        let r = r % (max + 1.0 - min) + min;
        r
    }
}
