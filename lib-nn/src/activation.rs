use std::f64::consts::E;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use lib_matrix::Matrix;

#[derive(Default, Debug, Clone, Deserialize, Serialize)]
pub enum Activation {
    /// Linear Unit Function = x
    LU,
    /// Rectified Linear Unit = max(0, x)
    #[default]
    RELU,
    /// Leaky Rectified Linear Unit = if x < 0 {x * 0.01} else { x }
    LRELU,
    /// Sigmoid = 1 / (1 + e^(-x))
    SIGMOID,
    /// TANH = 2 / (1 + e^(-2 * x))
    TANH,
    /// Softplus = log(1 + e^x)
    SOFTPLUS,
}

impl Activation {
    pub fn activate(&self, m: impl AsRef<Matrix>) -> Matrix {
        let col = m.as_ref().cols();
        let row = m.as_ref().rows();
        let data = m.as_ref().data();
        let d = match self {
            Activation::LU => data.to_owned(),
            Activation::RELU => data.par_iter().map(|d| d.max(0.0)).collect(),
            Activation::LRELU => data
                .par_iter()
                .map(|d| if *d < 0.0 { *d * 0.01 } else { *d })
                .collect(),
            Activation::SIGMOID => data.par_iter().map(|d| 1.0 / (1.0 + E.powf(-d))).collect(),
            Activation::TANH => data
                .par_iter()
                .map(|d| 2.0 / (1.0 + E.powf(-2.0 * d)))
                .collect(),
            Activation::SOFTPLUS => data.par_iter().map(|d| f64::ln(1.0 + E.powf(*d))).collect(),
        };

        Matrix::with_vec(col, row, d)
    }

    pub fn derivative(&self, m: impl AsRef<Matrix>) -> Matrix {
        let col = m.as_ref().cols();
        let row = m.as_ref().rows();
        let data = m.as_ref().data();
        let d = match self {
            Activation::LU => vec![1.0; row * col],
            Activation::RELU => data
                .par_iter()
                .map(|d| if *d <= 0.0 { 0.0 } else { 1.0 })
                .collect(),
            Activation::LRELU => data
                .par_iter()
                .map(|d| if *d < 0.0 { 0.01 } else { 1.0 })
                .collect(),
            Activation::SIGMOID => data
                .par_iter()
                .map(|d| (1.0 / (1.0 + E.powf(-d))) * (E.powf(-d) / (1.0 + E.powf(-d))))
                .collect(),
            Activation::TANH => data
                .par_iter()
                .map(|d| 1.0 - (2.0 / (1.0 + E.powf(-2.0 * d)).powf(2.0)))
                .collect(),
            Activation::SOFTPLUS => data.par_iter().map(|d| 1.0 / (1.0 + E.powf(-d))).collect(),
        };

        Matrix::with_vec(col, row, d)
    }
}
