#![allow(dead_code)]

mod matrix;
mod nn;

pub use matrix::Matrix;
pub use nn::{Activation, LossFunction, NeuralNetwork, Optimization};
