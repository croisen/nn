use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::matrix::Matrix;

#[derive(Default, Debug, Clone, Deserialize, Serialize)]
pub enum LossFunction {
    #[default]
    MeanSquaredError,
    MeanAbsoluteError,
}

impl LossFunction {
    pub fn loss(&self, result: impl AsRef<Matrix>, correct: impl AsRef<Matrix>) -> f64 {
        let n = result.as_ref().len() as f64;
        match self {
            LossFunction::MeanSquaredError => {
                result
                    .as_ref()
                    .iter()
                    .zip(correct.as_ref().iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>()
                    / n
            }
            LossFunction::MeanAbsoluteError => {
                result
                    .as_ref()
                    .iter()
                    .zip(correct.as_ref().iter())
                    .map(|(p, t)| (p - t).abs())
                    .sum::<f64>()
                    / n
            }
        }
    }

    pub fn derivative(&self, result: impl AsRef<Matrix>, correct: impl AsRef<Matrix>) -> Matrix {
        let n = result.as_ref().len() as f64;
        match self {
            LossFunction::MeanSquaredError => result
                .as_ref()
                .zip(correct.as_ref(), |p, t| 2.0 * (p - t) / n),
            LossFunction::MeanAbsoluteError => result.as_ref().zip(correct.as_ref(), |p, t| {
                if p > t {
                    1.0 / n
                } else if p < t {
                    -1.0 / n
                } else {
                    0.0
                }
            }),
        }
    }
}
