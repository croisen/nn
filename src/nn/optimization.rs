use serde::{Deserialize, Serialize};

use crate::matrix::Matrix;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum Optimization {
    NONE,
    SGD {
        lr: f64,
    },
    // Adam,
    // AdaGrad,
    /// learning rate (0.01), rho (0.9), epsilon (1e-8), leave wc and bc with empty vecs
    RmsProp {
        lr: f64,
        rho: f64,
        eps: f64,
        #[serde(skip)]
        wc: Vec<Matrix>,
        #[serde(skip)]
        bc: Vec<Matrix>,
    },
}

impl Optimization {
    pub fn init_cache(&mut self, w: &Vec<Matrix>, b: &Vec<Matrix>) {
        match self {
            Self::NONE => {}
            Self::SGD { lr: _ } => {}
            Self::RmsProp {
                lr: _,
                rho: _,
                eps: _,
                wc,
                bc,
            } => {
                wc.clear();
                bc.clear();
                w.iter().for_each(|w| wc.push(w.copy_zeroed()));
                b.iter().for_each(|b| bc.push(b.copy_zeroed()));
            }
        }
    }

    /// returns optimized gradients
    pub fn optimize(&mut self, idx: usize, wg: &Matrix, bg: &Matrix) -> (Matrix, Matrix) {
        match self {
            Self::NONE => (wg.to_owned(), bg.to_owned()),
            Self::SGD { lr } => (wg * *lr, bg * *lr),
            Self::RmsProp {
                lr,
                rho,
                eps,
                wc,
                bc,
            } => {
                wc[idx] = &wc[idx] * *rho + wg.hadamard_mul(wg) * (1.0 - *rho);
                bc[idx] = &bc[idx] * *rho + bg.hadamard_mul(bg) * (1.0 - *rho);
                let w = wg.hadamard_div(&wc[idx].map(|d| (d + *eps).sqrt())) * *lr;
                let b = bg.hadamard_div(&bc[idx].map(|d| (d + *eps).sqrt())) * *lr;
                (w, b)
            }
        }
    }

    pub fn none() -> Self {
        Self::NONE
    }

    pub fn sgd(learning_rate: f64) -> Self {
        Self::SGD { lr: learning_rate }
    }

    pub fn rmsprop(learning_rate: f64, rho: f64, epsilon: f64) -> Self {
        Self::RmsProp {
            lr: learning_rate,
            rho,
            eps: epsilon,
            wc: vec![],
            bc: vec![],
        }
    }
}

impl Default for Optimization {
    fn default() -> Self {
        Self::RmsProp {
            lr: 0.01,
            rho: 0.9,
            eps: 1.0e-8,
            wc: vec![],
            bc: vec![],
        }
    }
}
