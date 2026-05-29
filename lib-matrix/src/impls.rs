use std::fmt::{Display, Result};
use std::ops::{Add, AddAssign, Deref, DerefMut, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use rayon::prelude::*;

use crate::matrix::Matrix;

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result {
        writeln!(f, "Matrix ({} * {}):", self.rows, self.cols)?;
        for d in &self.data {
            writeln!(f, "{}\t{d:?}", d.len())?;
        }

        Ok(())
    }
}

impl AsRef<Matrix> for Matrix {
    fn as_ref(&self) -> &Matrix {
        self
    }
}

impl Deref for Matrix {
    type Target = Vec<Vec<f64>>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for Matrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl Index<usize> for Matrix {
    type Output = Vec<f64>;
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
        Self::Output::with_vec(
            self.rows,
            self.cols,
            self.par_iter()
                .map(|d| d.par_iter().map(|d| d + rhs).collect())
                .collect(),
        )
    }
}

impl Add<Matrix> for Matrix {
    type Output = Matrix;
    fn add(self, rhs: Matrix) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (addition)\n{self}{rhs}"
        );

        Self::Output::with_vec(
            self.rows,
            self.cols,
            self.par_iter()
                .zip(rhs.par_iter())
                .map(|(a, b)| a.par_iter().zip(b.par_iter()).map(|(a, b)| a + b).collect())
                .collect(),
        )
    }
}

impl<'b> Add<&'b Matrix> for Matrix {
    type Output = Matrix;
    fn add(self, rhs: &'b Matrix) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (addition)\n{self}{rhs}"
        );

        Self::Output::with_vec(
            self.rows,
            self.cols,
            self.par_iter()
                .zip(rhs.par_iter())
                .map(|(a, b)| a.par_iter().zip(b.par_iter()).map(|(a, b)| a + b).collect())
                .collect(),
        )
    }
}

impl<'a> Add<Matrix> for &'a Matrix {
    type Output = Matrix;
    fn add(self, rhs: Matrix) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (addition)\n{self}{rhs}"
        );

        Self::Output::with_vec(
            self.rows,
            self.cols,
            self.par_iter()
                .zip(rhs.par_iter())
                .map(|(a, b)| a.par_iter().zip(b.par_iter()).map(|(a, b)| a + b).collect())
                .collect(),
        )
    }
}

impl<'a, 'b> Add<&'b Matrix> for &'a Matrix {
    type Output = Matrix;
    fn add(self, rhs: &'b Matrix) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (addition)\n{self}{rhs}"
        );

        Self::Output::with_vec(
            self.rows,
            self.cols,
            self.par_iter()
                .zip(rhs.par_iter())
                .map(|(a, b)| a.par_iter().zip(b.par_iter()).map(|(a, b)| a + b).collect())
                .collect(),
        )
    }
}

impl AddAssign<f64> for Matrix {
    fn add_assign(&mut self, rhs: f64) {
        self.data = self
            .par_iter()
            .map(|d| d.par_iter().map(|d| d + rhs).collect())
            .collect();
    }
}

impl<'a> AddAssign<&'a Matrix> for Matrix {
    fn add_assign(&mut self, rhs: &'a Matrix) {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (addition)\n{self}{rhs}"
        );

        self.data = self
            .par_iter()
            .zip(rhs.par_iter())
            .map(|(l, r)| l.par_iter().zip(r).map(|(l, r)| l + r).collect())
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
            .par_iter()
            .zip(rhs.par_iter())
            .map(|(l, r)| l.par_iter().zip(r).map(|(l, r)| l + r).collect())
            .collect();
    }
}

impl Sub<f64> for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: f64) -> Self::Output {
        Self::Output::with_vec(
            self.rows,
            self.cols,
            self.par_iter()
                .map(|d| d.par_iter().map(|d| d - rhs).collect())
                .collect(),
        )
    }
}

impl Sub<Matrix> for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Matrix) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (subtraction)\n{self}{rhs}"
        );

        Self::Output::with_vec(
            self.rows,
            self.cols,
            self.par_iter()
                .zip(rhs.par_iter())
                .map(|(a, b)| a.par_iter().zip(b.par_iter()).map(|(a, b)| a - b).collect())
                .collect(),
        )
    }
}

impl<'b> Sub<&'b Matrix> for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &'b Matrix) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (subtraction)\n{self}{rhs}"
        );

        Self::Output::with_vec(
            self.rows,
            self.cols,
            self.par_iter()
                .zip(rhs.par_iter())
                .map(|(a, b)| a.par_iter().zip(b.par_iter()).map(|(a, b)| a - b).collect())
                .collect(),
        )
    }
}

impl<'a> Sub<Matrix> for &'a Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Matrix) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (subtraction)\n{self}{rhs}"
        );

        Self::Output::with_vec(
            self.rows,
            self.cols,
            self.par_iter()
                .zip(rhs.par_iter())
                .map(|(a, b)| a.par_iter().zip(b.par_iter()).map(|(a, b)| a - b).collect())
                .collect(),
        )
    }
}

impl<'a, 'b> Sub<&'b Matrix> for &'a Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &'b Matrix) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (subtraction)\n{self}{rhs}"
        );

        Self::Output::with_vec(
            self.rows,
            self.cols,
            self.par_iter()
                .zip(rhs.par_iter())
                .map(|(a, b)| a.par_iter().zip(b.par_iter()).map(|(a, b)| a - b).collect())
                .collect(),
        )
    }
}

impl SubAssign<f64> for Matrix {
    fn sub_assign(&mut self, rhs: f64) {
        self.data = self
            .par_iter()
            .map(|d| d.par_iter().map(|d| d - rhs).collect())
            .collect();
    }
}

impl<'a> SubAssign<&'a Matrix> for Matrix {
    fn sub_assign(&mut self, rhs: &'a Matrix) {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Column and row count not equal (subtraction)\n{self}{rhs}"
        );

        self.data = self
            .par_iter()
            .zip(rhs.par_iter())
            .map(|(l, r)| l.par_iter().zip(r).map(|(l, r)| l - r).collect())
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
            .par_iter()
            .zip(rhs.par_iter())
            .map(|(l, r)| l.par_iter().zip(r).map(|(l, r)| l - r).collect())
            .collect();
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
        Self::Output::with_vec(
            self.rows,
            rhs.cols,
            self.par_iter()
                .map(|l| {
                    r.par_iter()
                        .map(|r| l.par_iter().zip(r.par_iter()).map(|(l, r)| l * r).sum())
                        .collect()
                })
                .collect(),
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
        let data: Vec<Vec<f64>> = self
            .par_iter()
            .map(|l| {
                r.par_iter()
                    .map(|r| l.par_iter().zip(r.par_iter()).map(|(l, r)| l * r).sum())
                    .collect()
            })
            .collect();

        let rows = data.len();
        let cols = data[0].len();
        Self::Output::with_vec(rows, cols, data)
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
        let data: Vec<Vec<f64>> = self
            .par_iter()
            .map(|l| {
                r.par_iter()
                    .map(|r| l.par_iter().zip(r.par_iter()).map(|(l, r)| l * r).sum())
                    .collect()
            })
            .collect();

        let rows = data.len();
        let cols = data[0].len();
        Self::Output::with_vec(rows, cols, data)
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
        let data: Vec<Vec<f64>> = self
            .par_iter()
            .map(|l| {
                r.par_iter()
                    .map(|r| l.par_iter().zip(r.par_iter()).map(|(l, r)| l * r).sum())
                    .collect()
            })
            .collect();

        let rows = data.len();
        let cols = data[0].len();
        Self::Output::with_vec(rows, cols, data)
    }
}

impl<'a> MulAssign<&'a Matrix> for Matrix {
    fn mul_assign(&mut self, rhs: &'a Matrix) {
        assert!(
            self.cols == rhs.rows,
            "Left col != right row count (dot mul)\n{self}{rhs}"
        );

        let r = rhs.transpose();
        let data: Vec<Vec<f64>> = self
            .par_iter()
            .map(|l| {
                r.par_iter()
                    .map(|r| l.par_iter().zip(r.par_iter()).map(|(l, r)| l * r).sum())
                    .collect()
            })
            .collect();

        let rows = data.len();
        let cols = data[0].len();
        self.rows = rows;
        self.cols = cols;
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
        let data: Vec<Vec<f64>> = self
            .par_iter()
            .map(|l| {
                r.par_iter()
                    .map(|r| l.par_iter().zip(r.par_iter()).map(|(l, r)| l * r).sum())
                    .collect()
            })
            .collect();

        let rows = data.len();
        let cols = data[0].len();
        self.rows = rows;
        self.cols = cols;
        self.data = data;
    }
}
