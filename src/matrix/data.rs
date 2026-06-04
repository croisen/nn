use std::ops::Index;
use std::slice::{Chunks, Iter};

use rayon::prelude::*;
use rayon::slice::{Chunks as PChunks, Iter as PIter};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum Data {
    Box(Box<[f64]>),
    #[serde(skip)]
    Static(&'static [f64]),
}

impl Data {
    pub fn data(&self) -> &[f64] {
        match self {
            Self::Box(b) => b.as_ref(),
            Self::Static(s) => s,
        }
    }

    pub fn iter(&self) -> Iter<'_, f64> {
        self.data().iter()
    }

    pub fn par_iter(&self) -> PIter<'_, f64> {
        self.data().into_par_iter()
    }

    pub fn chunks(&self, size: usize) -> Chunks<'_, f64> {
        self.data().chunks(size)
    }

    pub fn par_chunks(&self, size: usize) -> PChunks<'_, f64> {
        self.data().par_chunks(size)
    }

    pub fn len(&self) -> usize {
        self.data().len()
    }
}

impl Default for Data {
    fn default() -> Self {
        Self::Box(Box::new([]))
    }
}

impl Index<usize> for Data {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data()[index]
    }
}

impl FromIterator<f64> for Data {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        Self::Box(iter.into_iter().collect())
    }
}

impl FromParallelIterator<f64> for Data {
    fn from_par_iter<I: IntoParallelIterator<Item = f64>>(par_iter: I) -> Self {
        Self::Box(par_iter.into_par_iter().collect())
    }
}
