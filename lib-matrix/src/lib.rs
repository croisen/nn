use std::fmt::{Display, Formatter, Result};
use std::ops::{Add, Mul, Sub};

use rand::distr::StandardUniform;
use rand::{RngExt, rng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone, Deserialize, Serialize)]
pub struct Matrix {
    cols: usize,
    rows: usize,
    data: Vec<f64>,
}

impl Matrix {
    pub fn new(cols: usize, rows: usize) -> Self {
        Self {
            cols,
            rows,
            data: vec![0.0; cols * rows],
        }
    }

    pub fn rand(cols: usize, rows: usize) -> Self {
        let mut s = Self::new(cols, rows);
        let mut g = rng();
        for i in 0..s.rows * s.cols {
            s.data[i] = g.sample(StandardUniform);
        }

        s
    }

    pub fn with_vec(cols: usize, rows: usize, data: Vec<f64>) -> Matrix {
        Self { cols, rows, data }
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn data(&self) -> &Vec<f64> {
        &self.data
    }

    pub fn transpose(&self) -> Self {
        let v = self.data.chunks(self.cols).collect::<Vec<&[f64]>>();
        let d = (0..self.cols)
            .into_par_iter()
            .map(|i| v.iter().map(|r| r[i]).collect::<Vec<f64>>())
            .flatten()
            .collect::<Vec<f64>>();

        Self {
            cols: self.rows,
            rows: self.cols,
            data: d,
        }
    }

    pub fn scalar_mul(&self, k: f64) -> Self {
        Self {
            cols: self.rows,
            rows: self.cols,
            data: self.data.par_iter().map(|d| d * k).collect::<Vec<f64>>(),
        }
    }

    pub fn dot_mul(&self, rhs: &Self) -> Self {
        assert!(self.cols == rhs.rows);
        let c = rhs.cols;
        let r = rhs.transpose();
        let d = self
            .data
            .par_chunks(self.cols)
            .map(|l| {
                r.data
                    .par_chunks(r.cols)
                    .map(|r| l.par_iter().zip(r).map(|(l, r)| l * r).sum())
                    .collect::<Vec<f64>>()
            })
            .flatten()
            .collect::<Vec<f64>>();

        Self::with_vec(c, self.rows, d)
    }
}

#[macro_export]
macro_rules! matrix {
    ($($($x: expr),+);* $(;)?) => {{
        use rayon::prelude::*;
        use lib_matrix::Matrix;

        let v = [$([$($x as f64),+]),*];
        let rows = v.len();
        let cols = v[0].len();
        let data = v.into_par_iter().flatten().collect::<Vec<f64>>();
        Matrix::with_vec(cols, rows, data)
    }};

    ($(;)*) => {{
        use crate::matrix::Matrix;
        Matrix::new(0, 0)
    }}
}

impl Add for Matrix {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        assert!(self.rows == rhs.rows && self.cols == rhs.cols);
        let l = self.data.par_iter();
        let r = rhs.data.par_iter();
        let d = l.zip(r).map(|(a, b)| a + b).collect::<Vec<f64>>();

        Self {
            rows: self.rows,
            cols: self.cols,
            data: d,
        }
    }
}

impl Sub for Matrix {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        assert!(self.rows == rhs.rows && self.cols == rhs.cols);
        let l = self.data.par_iter();
        let r = rhs.data.par_iter();
        let d = l.zip(r).map(|(a, b)| a - b).collect::<Vec<f64>>();

        Self {
            rows: self.rows,
            cols: self.cols,
            data: d,
        }
    }
}

impl Mul for Matrix {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        assert!(self.rows == rhs.rows && self.cols == rhs.cols);
        let l = self.data.par_iter();
        let r = rhs.data.par_iter();
        let d = l.zip(r).map(|(a, b)| a * b).collect::<Vec<f64>>();

        Self {
            rows: self.rows,
            cols: self.cols,
            data: d,
        }
    }
}

impl AsRef<Matrix> for Matrix {
    fn as_ref(&self) -> &Matrix {
        self
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let cols = self.data.chunks(self.cols);
        writeln!(f)?;
        for col in cols {
            let d = col
                .iter()
                .map(|s| format!("{s:.2}"))
                .collect::<Vec<String>>()
                .join(", ");

            writeln!(f, "{d}")?;
        }

        Ok(())
    }
}
