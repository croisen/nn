use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use lib_matrix::Matrix;

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
                    .par_iter()
                    .flatten()
                    .collect::<Vec<&f64>>()
                    .par_iter()
                    .zip(correct.as_ref().par_iter().flatten().collect::<Vec<&f64>>())
                    .map(|(p, t)| (**p - *t).powi(2))
                    .sum::<f64>()
                    / n
            }
            LossFunction::MeanAbsoluteError => {
                result
                    .as_ref()
                    .par_iter()
                    .flatten()
                    .collect::<Vec<&f64>>()
                    .par_iter()
                    .zip(correct.as_ref().par_iter().flatten().collect::<Vec<&f64>>())
                    .map(|(p, t)| (**p - *t).abs())
                    .sum::<f64>()
                    / n
            }
        }
    }

    pub fn derivative(&self, result: impl AsRef<Matrix>, correct: impl AsRef<Matrix>) -> Matrix {
        let n = result.as_ref().len() as f64;
        let data: Vec<f64> = match self {
            LossFunction::MeanSquaredError => result
                .as_ref()
                .par_iter()
                .flatten()
                .collect::<Vec<&f64>>()
                .par_iter()
                .zip(correct.as_ref().par_iter().flatten().collect::<Vec<&f64>>())
                .map(|(p, t)| 2.0 * (**p - *t) / n)
                .collect(),
            LossFunction::MeanAbsoluteError => result
                .as_ref()
                .par_iter()
                .flatten()
                .collect::<Vec<&f64>>()
                .par_iter()
                .zip(correct.as_ref().par_iter().flatten().collect::<Vec<&f64>>())
                .map(|(p, t)| {
                    if **p > *t {
                        1.0 / n
                    } else if **p < *t {
                        -1.0 / n
                    } else {
                        0.0
                    }
                })
                .collect(),
        };

        Matrix::with_vec(
            result.as_ref().rows,
            result.as_ref().cols,
            data.par_chunks(result.as_ref().cols)
                .map(|v| v.to_vec())
                .collect(),
        )
    }
}
