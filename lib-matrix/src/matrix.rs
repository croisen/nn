use rand::RngExt;
use rand::distr::StandardUniform;
use rand::rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone, Deserialize, Serialize)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self::with_vec(rows, cols, vec![vec![0.0; cols]; rows])
    }

    pub fn rand(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: (0..rows)
                .into_iter()
                .map(|_| {
                    (0..cols)
                        .into_iter()
                        .map(|_| rng().sample(StandardUniform))
                        .collect()
                })
                .collect(),
        }
    }

    pub fn with_vec(rows: usize, cols: usize, data: Vec<Vec<f64>>) -> Self {
        Self { rows, cols, data }
    }

    pub fn transpose(&self) -> Matrix {
        Self::with_vec(
            self.cols,
            self.rows,
            (0..self.cols)
                .into_par_iter()
                .map(|i| self.par_iter().map(|a| a[i]).collect())
                .collect(),
        )
    }

    pub fn scalar_mul(&self, k: f64) -> Self {
        Self::with_vec(
            self.rows,
            self.cols,
            self.data
                .par_iter()
                .map(|d| d.par_iter().map(|d| d * k).collect())
                .collect(),
        )
    }

    pub fn hadamard_mul(&self, rhs: &Self) -> Self {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (hadamard mul)\n{self}{rhs}"
        );

        Self::with_vec(
            self.rows,
            self.cols,
            self.data
                .par_iter()
                .zip(rhs.data.par_iter())
                .map(|(l, r)| l.par_iter().zip(r.par_iter()).map(|(l, r)| l * r).collect())
                .collect(),
        )
    }

    pub fn sum_over_rows(&self) -> Self {
        let data: Vec<f64> = (0..self.cols)
            .into_iter()
            .map(|i| self.iter().map(|a| a[i]).sum())
            .collect();

        Self::with_vec(1, data.len(), vec![data])
    }
}

#[macro_export]
macro_rules! matrix {
    ($($($x: expr),+);* $(;)?) => {{
        use rayon::prelude::*;
        let data = [$([$($x as f64),*]),*];
        lib_matrix::Matrix::with_vec(data.len(), data[0].len(), data.iter().map(|v| v.to_vec()).collect())
    }}
}
