use std::f64::consts::E;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use lib_matrix::Matrix;

#[derive(Default, Debug, Clone, Copy, Deserialize, Serialize)]
pub enum Activation {
    /// Linear Unit Function = x
    Lu,
    /// Rectified Linear Unit = max(0, x)
    #[default]
    ReLU,
    /// Leaky Rectified Linear Unit = if x < 0 {x * 0.01} else { x }
    LReLU,
    /// Sigmoid = 1 / (1 + e^(-x))
    Sigmoid,
    /// TANH = 2 / (1 + e^(-2 * x))
    Tanh,
    /// Softplus = log(1 + e^x)
    Softplus,
}

impl Activation {
    pub fn activate(&self, m: impl AsRef<Matrix>) -> Matrix {
        let col = m.as_ref().cols;
        let row = m.as_ref().rows;
        let data = &m.as_ref().data;
        let d = match self {
            Activation::Lu => data.clone(),
            Activation::ReLU => data
                .par_iter()
                .map(|d| d.par_iter().map(|d| d.max(0.0)).collect())
                .collect(),
            Activation::LReLU => data
                .par_iter()
                .map(|d| {
                    d.par_iter()
                        .map(|d| if *d < 0.0 { *d * 0.01 } else { *d })
                        .collect()
                })
                .collect(),
            Activation::Sigmoid => data
                .par_iter()
                .map(|d| d.par_iter().map(|d| 1.0 / (1.0 + E.powf(-d))).collect())
                .collect(),
            Activation::Tanh => data
                .par_iter()
                .map(|d| {
                    d.par_iter()
                        .map(|d| 2.0 / (1.0 + E.powf(-2.0 * d)))
                        .collect()
                })
                .collect(),
            Activation::Softplus => data
                .par_iter()
                .map(|d| d.par_iter().map(|d| f64::ln(1.0 + E.powf(*d))).collect())
                .collect(),
        };

        Matrix::with_vec(row, col, d)
    }

    pub fn derivative(&self, m: impl AsRef<Matrix>) -> Matrix {
        let col = m.as_ref().cols;
        let row = m.as_ref().rows;
        let data = &m.as_ref().data;
        let d = match self {
            Activation::Lu => vec![vec![1.0; col]; row],
            Activation::ReLU => data
                .par_iter()
                .map(|d| {
                    d.par_iter()
                        .map(|d| if *d <= 0.0 { 0f64 } else { 1f64 })
                        .collect::<Vec<f64>>()
                })
                .collect(),
            Activation::LReLU => data
                .par_iter()
                .map(|d| {
                    d.par_iter()
                        .map(|d| if *d < 0.0 { 0.01 } else { 1.0 })
                        .collect()
                })
                .collect(),
            Activation::Sigmoid => data
                .par_iter()
                .map(|d| {
                    d.par_iter()
                        .map(|d| (1.0 / (1.0 + E.powf(-d))) * (E.powf(-d) / (1.0 + E.powf(-d))))
                        .collect()
                })
                .collect(),
            Activation::Tanh => data
                .par_iter()
                .map(|d| {
                    d.par_iter()
                        .map(|d| 1.0 - (2.0 / (1.0 + E.powf(-2.0 * d)).powf(2.0)))
                        .collect()
                })
                .collect(),
            Activation::Softplus => data
                .par_iter()
                .map(|d| d.par_iter().map(|d| 1.0 / (1.0 + E.powf(-d))).collect())
                .collect(),
        };

        Matrix::with_vec(row, col, d)
    }
}
