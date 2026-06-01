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
    /// Softmax = e^x / Σ e^x
    Softmax,
}

impl Activation {
    pub fn activate(&self, m: impl AsRef<Matrix>) -> Matrix {
        let m = m.as_ref();
        match self {
            Activation::Lu => m.map(Self::lu_activate),
            Activation::ReLU => m.map(Self::relu_activate),
            Activation::LReLU => m.map(Self::lrelu_activate),
            Activation::Sigmoid => m.map(Self::sigmoid_activate),
            Activation::Tanh => m.map(Self::tanh_activate),
            Activation::Softplus => m.map(Self::softplus_activate),
            Activation::Softmax => {
                let sum = Self::softmax_sum(&m);
                m.map(|d| Self::softmax_activate(d, sum))
            }
        }
    }

    pub fn derivative(&self, m: impl AsRef<Matrix>) -> Matrix {
        let m = m.as_ref();
        match self {
            Activation::Lu => m.map(Self::lu_derivative),
            Activation::ReLU => m.map(Self::relu_derivative),
            Activation::LReLU => m.map(Self::lrelu_derivative),
            Activation::Sigmoid => m.map(Self::sigmoid_derivative),
            Activation::Tanh => m.map(Self::tanh_derivative),
            Activation::Softplus => m.map(Self::softplus_derivative),
            Activation::Softmax => {
                let sum = Self::softmax_sum(&m);
                m.as_ref()
                    .map(|d| Self::softmax_activate(d, sum) * (sum - d))
            }
        }
    }

    fn lu_activate(v: &f64) -> f64 {
        *v
    }

    fn lu_derivative(_: &f64) -> f64 {
        1.0
    }

    fn relu_activate(v: &f64) -> f64 {
        v.max(0.0)
    }

    fn relu_derivative(v: &f64) -> f64 {
        if *v < 0.0 { 0.0 } else { 1.0 }
    }

    fn lrelu_activate(v: &f64) -> f64 {
        if *v < 0.0 { v * 0.001 } else { *v }
    }

    fn lrelu_derivative(v: &f64) -> f64 {
        if *v < 0.0 { 0.001 } else { 1.0 }
    }

    fn sigmoid_activate(v: &f64) -> f64 {
        1.0 / (1.0 + E.powf(-v))
    }

    fn sigmoid_derivative(v: &f64) -> f64 {
        Self::sigmoid_activate(&v) * (1.0 - Self::sigmoid_activate(&v))
    }

    fn tanh_activate(v: &f64) -> f64 {
        2.0 / (1.0 + E.powf(-2.0 * v))
    }

    fn tanh_derivative(v: &f64) -> f64 {
        1.0 - Self::tanh_activate(v).powi(2)
    }

    fn softplus_activate(v: &f64) -> f64 {
        E.powf(*v).ln_1p()
    }

    fn softplus_derivative(v: &f64) -> f64 {
        let e = E.powf(*v);
        e / (1.0 + e)
    }

    fn softmax_sum(m: impl AsRef<Matrix>) -> f64 {
        m.as_ref().iter().map(|d| E.powf(*d)).sum()
    }

    fn softmax_activate(v: &f64, sum: f64) -> f64 {
        E.powf(*v) / sum
    }
}
