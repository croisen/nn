use std::fmt::{Display, Result};
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use rand::distr::OpenClosed01;
use rand::{RngExt, rng};
use rayon::prelude::*;
use rayon::slice::Iter;
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone, Deserialize, Serialize)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Box<[f64]>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self::with_slice(rows, cols, vec![0.0; rows * cols])
    }

    pub fn rand(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: (0..rows * cols)
                .into_iter()
                .map(|_| rng().sample(OpenClosed01))
                .collect(),
        }
    }

    pub fn transpose(&self) -> Matrix {
        Self::with_slice(
            self.cols,
            self.rows,
            (0..self.cols)
                .into_par_iter()
                .map(|i| {
                    self.data
                        .par_chunks(self.cols)
                        .map(|a| a[i])
                        .collect::<Box<[f64]>>()
                })
                .flatten()
                .collect::<Box<[f64]>>(),
        )
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn iter(&self) -> Iter<'_, f64> {
        self.data.par_iter()
    }

    pub fn map<T: Fn(&f64) -> f64 + Sync + Send>(&self, f: T) -> Self {
        Self::with_slice(
            self.rows,
            self.cols,
            self.data.par_iter().map(f).collect::<Box<[f64]>>(),
        )
    }

    pub fn zip<T: Fn(&f64, &f64) -> f64 + Sync + Send>(&self, other: &Self, f: T) -> Self {
        assert!(
            self.rows == other.rows && self.cols == other.cols,
            "Column and row count not equal (zip)\n{self}{other}"
        );
        Self::with_slice(
            self.rows,
            self.cols,
            self.data
                .par_iter()
                .zip(other.data.par_iter())
                .map(|(a, b)| f(a, b))
                .collect::<Box<[f64]>>(),
        )
    }

    pub fn scalar_mul(&self, k: f64) -> Self {
        self.map(|d| d * k)
    }

    pub fn hadamard_mul(&self, rhs: &Self) -> Self {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (hadamard mul)\n{self}{rhs}"
        );

        self.zip(rhs, |l, r| l * r)
    }

    pub fn hadamard_div(&self, rhs: &Self) -> Self {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (hadamard mul)\n{self}{rhs}"
        );

        self.zip(rhs, |l, r| l / r)
    }

    pub fn sum_over_rows(&self) -> Self {
        let data: Vec<f64> = (0..self.cols)
            .into_iter()
            .map(|i| self.data.chunks(self.cols).map(|a| a[i]).sum())
            .collect();

        Self::with_slice(1, data.len(), data)
    }

    pub fn with_slice(rows: usize, cols: usize, data: impl Into<Box<[f64]>>) -> Self {
        Self {
            rows,
            cols,
            data: data.into(),
        }
    }

    pub const fn with_static_slice(rows: usize, cols: usize, data: &'static [f64]) -> Self {
        Self {
            rows,
            cols,
            data: unsafe { std::mem::transmute(data) },
        }
    }

    pub fn copy_zeroed(&self) -> Self {
        Self::new(self.rows, self.cols)
    }

    pub fn copy_random(&self) -> Self {
        Self::rand(self.rows, self.cols)
    }
}

#[macro_export]
macro_rules! matrix {
    ($($($x: expr),+);* $(;)?) => {{
        let data = [$([$($x as f64),*]),*];
        let d = [$($($x as f64),*),*];
        lib_matrix::Matrix::with_slice(data.len(), data[0].len(), d)
    }};

    ($idx: expr, $(@)+ $len: expr) => {{
        let mut d = [0.0; $len];
        d[$idx] = 1 as f64;

        lib_matrix::Matrix::with_slice(1, $len, d)
    }};
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result {
        writeln!(f, "Matrix ({} * {}):", self.rows, self.cols)?;
        let len = self.data.len();
        for r in 0..self.rows {
            write!(f, "\t[ ")?;
            for c in 0..self.cols {
                let idx = r * self.cols + c;
                if idx < len {
                    write!(f, "{:4.2}, ", self[idx])?;
                } else {
                    write!(f, "{:4.2}", self[idx])?;
                }
            }

            writeln!(f, " ]")?;
        }

        Ok(())
    }
}

impl AsRef<Matrix> for Matrix {
    fn as_ref(&self) -> &Matrix {
        self
    }
}

impl Index<usize> for Matrix {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl Add<f64> for Matrix {
    type Output = Matrix;
    fn add(self, rhs: f64) -> Self::Output {
        self.map(|d| d * rhs)
    }
}

impl Add<Matrix> for Matrix {
    type Output = Matrix;
    fn add(self, rhs: Matrix) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (addition)\n{self}{rhs}"
        );

        self.zip(&rhs, |l, r| l + r)
    }
}

impl<'b> Add<&'b Matrix> for Matrix {
    type Output = Matrix;
    fn add(self, rhs: &'b Matrix) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (addition)\n{self}{rhs}"
        );

        self.zip(rhs, |l, r| l + r)
    }
}

impl<'a> Add<Matrix> for &'a Matrix {
    type Output = Matrix;
    fn add(self, rhs: Matrix) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (addition)\n{self}{rhs}"
        );

        self.zip(&rhs, |l, r| l + r)
    }
}

impl<'a, 'b> Add<&'b Matrix> for &'a Matrix {
    type Output = Matrix;
    fn add(self, rhs: &'b Matrix) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (addition)\n{self}{rhs}"
        );

        self.zip(rhs, |l, r| l + r)
    }
}

impl AddAssign<f64> for Matrix {
    fn add_assign(&mut self, rhs: f64) {
        self.data = self.data.par_iter().map(|d| d + rhs).collect();
    }
}

impl<'a> AddAssign<&'a Matrix> for Matrix {
    fn add_assign(&mut self, rhs: &'a Matrix) {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (addition)\n{self}{rhs}"
        );

        self.data = self
            .data
            .par_iter()
            .zip(rhs.data.par_iter())
            .map(|(l, r)| l + r)
            .collect();
    }
}

impl AddAssign<Matrix> for Matrix {
    fn add_assign(&mut self, rhs: Matrix) {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (addition)\n{self}{rhs}"
        );

        self.data = self
            .data
            .par_iter()
            .zip(rhs.data.par_iter())
            .map(|(l, r)| l + r)
            .collect();
    }
}

impl Sub<f64> for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: f64) -> Self::Output {
        self.map(|d| d - rhs)
    }
}

impl Sub<Matrix> for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Matrix) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (subtraction)\n{self}{rhs}"
        );

        self.zip(&rhs, |l, r| l - r)
    }
}

impl<'b> Sub<&'b Matrix> for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &'b Matrix) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (subtraction)\n{self}{rhs}"
        );

        self.zip(rhs, |l, r| l - r)
    }
}

impl<'a> Sub<Matrix> for &'a Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Matrix) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (subtraction)\n{self}{rhs}"
        );

        self.zip(&rhs, |l, r| l - r)
    }
}

impl<'a, 'b> Sub<&'b Matrix> for &'a Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &'b Matrix) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (subtraction)\n{self}{rhs}"
        );

        self.zip(rhs, |l, r| l - r)
    }
}

impl SubAssign<f64> for Matrix {
    fn sub_assign(&mut self, rhs: f64) {
        self.data = self.data.par_iter().map(|d| d - rhs).collect();
    }
}

impl<'a> SubAssign<&'a Matrix> for Matrix {
    fn sub_assign(&mut self, rhs: &'a Matrix) {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (subtraction)\n{self}{rhs}"
        );

        self.data = self
            .data
            .par_iter()
            .zip(rhs.data.par_iter())
            .map(|(l, r)| l - r)
            .collect();
    }
}

impl SubAssign<Matrix> for Matrix {
    fn sub_assign(&mut self, rhs: Matrix) {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (subtraction)\n{self}{rhs}"
        );

        self.data = self
            .data
            .par_iter()
            .zip(rhs.data.par_iter())
            .map(|(l, r)| l - r)
            .collect();
    }
}

impl Mul<f64> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: f64) -> Self::Output {
        self.map(|d| d * rhs)
    }
}

impl<'a> Mul<f64> for &'a Matrix {
    type Output = Matrix;
    fn mul(self, rhs: f64) -> Self::Output {
        self.map(|d| d * rhs)
    }
}

impl<'a> Mul<&'a f64> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &'a f64) -> Self::Output {
        self.map(|d| d * rhs)
    }
}

impl<'a> Mul<&'a mut f64> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &'a mut f64) -> Self::Output {
        self.map(|d| d * *rhs)
    }
}

impl<'a, 'b> Mul<&'b f64> for &'a Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &'b f64) -> Self::Output {
        self.map(|d| d * rhs)
    }
}

impl<'a, 'b> Mul<&'b mut f64> for &'a Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &'b mut f64) -> Self::Output {
        self.map(|d| d * *rhs)
    }
}

impl Mul<Matrix> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Matrix) -> Self::Output {
        assert!(
            self.cols == rhs.rows,
            "Left col != right row count (dot mul)\n{self}{rhs}"
        );

        let r = rhs.transpose();
        Self::with_slice(
            self.rows,
            rhs.cols,
            self.data
                .par_chunks(self.cols)
                .map(|l| {
                    r.data
                        .par_chunks(r.cols)
                        .map(|r| r.par_iter().zip(l).map(|(r, l)| r * l).sum())
                        .collect::<Box<[f64]>>()
                })
                .flatten()
                .collect::<Box<[f64]>>(),
        )
    }
}

impl<'b> Mul<&'b Matrix> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &'b Matrix) -> Self::Output {
        assert!(
            self.cols == rhs.rows,
            "Left col != right row count (dot mul)\n{self}{rhs}"
        );

        let r = rhs.transpose();
        Self::with_slice(
            self.rows,
            rhs.cols,
            self.data
                .par_chunks(self.cols)
                .map(|l| {
                    r.data
                        .par_chunks(r.cols)
                        .map(|r| r.par_iter().zip(l).map(|(r, l)| r * l).sum())
                        .collect::<Box<[f64]>>()
                })
                .flatten()
                .collect::<Box<[f64]>>(),
        )
    }
}

impl<'a> Mul<Matrix> for &'a Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Matrix) -> Self::Output {
        assert!(
            self.cols == rhs.rows,
            "Left col != right row count (dot mul)\n{self}{rhs}"
        );

        let r = rhs.transpose();
        Self::Output::with_slice(
            self.rows,
            rhs.cols,
            self.data
                .par_chunks(self.cols)
                .map(|l| {
                    r.data
                        .par_chunks(r.cols)
                        .map(|r| r.par_iter().zip(l).map(|(r, l)| r * l).sum())
                        .collect::<Box<[f64]>>()
                })
                .flatten()
                .collect::<Box<[f64]>>(),
        )
    }
}

impl<'a, 'b> Mul<&'b Matrix> for &'a Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &'b Matrix) -> Self::Output {
        assert!(
            self.cols == rhs.rows,
            "Left col != right row count (dot mul)\n{self}{rhs}"
        );

        let r = rhs.transpose();
        Self::Output::with_slice(
            self.rows,
            rhs.cols,
            self.data
                .par_chunks(self.cols)
                .map(|l| {
                    r.data
                        .par_chunks(r.cols)
                        .map(|r| r.par_iter().zip(l).map(|(r, l)| r * l).sum())
                        .collect::<Box<[f64]>>()
                })
                .flatten()
                .collect::<Box<[f64]>>(),
        )
    }
}

impl<'a> MulAssign<&'a Matrix> for Matrix {
    fn mul_assign(&mut self, rhs: &'a Matrix) {
        assert!(
            self.cols == rhs.rows,
            "Left col != right row count (dot mul)\n{self}{rhs}"
        );

        let r = rhs.transpose();
        let data = self
            .data
            .par_chunks(self.cols)
            .map(|l| {
                r.data
                    .par_chunks(r.cols)
                    .map(|r| r.par_iter().zip(l).map(|(r, l)| r * l).sum())
                    .collect::<Box<[f64]>>()
            })
            .flatten()
            .collect::<Box<[f64]>>();

        self.cols = rhs.cols;
        self.data = data;
    }
}

impl MulAssign<Matrix> for Matrix {
    fn mul_assign(&mut self, rhs: Matrix) {
        assert!(
            self.cols == rhs.rows,
            "Left col != right row count (dot mul)\n{self}{rhs}"
        );

        let r = rhs.transpose();
        let data = self
            .data
            .par_chunks(self.cols)
            .map(|l| {
                r.data
                    .par_chunks(r.cols)
                    .map(|r| r.par_iter().zip(l).map(|(r, l)| r * l).sum())
                    .collect::<Box<[f64]>>()
            })
            .flatten()
            .collect::<Box<[f64]>>();

        self.cols = rhs.cols;
        self.data = data;
    }
}
