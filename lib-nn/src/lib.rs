use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use lib_matrix::Matrix;

mod activation;
pub use activation::Activation;

#[derive(Default, Debug, Clone, Deserialize, Serialize)]
pub struct NeuralNetwork {
    c_layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    activation: Activation,
}

impl NeuralNetwork {
    /// layers = [ (size of input), any size..., (size of output) ]
    /// i.e. an image of 768 pixels and 10 outputs layers = [768, 16, 42, 18, 10]
    pub fn new(layers: impl AsRef<[usize]>, activation: Activation) -> Self {
        let len = layers.as_ref().len();
        let mut weights = Vec::with_capacity(len);
        let mut biases = Vec::with_capacity(len);

        // [ input * hl[1] ]
        for i in 0..len - 1 {
            weights.push(Matrix::rand(layers.as_ref()[i + 1], layers.as_ref()[i]));
        }

        for i in 1..len {
            biases.push(Matrix::rand(1, layers.as_ref()[i]));
        }

        Self {
            activation,
            c_layers: Vec::from(layers.as_ref()),
            weights,
            biases,
        }
    }

    pub fn print_weights(&self) {
        println!("-------------[Weights]-----------------");
        for w in &self.weights {
            println!("{w}");
        }
        println!("---------------------------------------");
    }

    pub fn print_biases(&self) {
        println!("-------------[Biases]-----------------");
        for b in &self.biases {
            println!("{b}");
        }
        println!("---------------------------------------");
    }

    pub fn forward_propagation(&self, input: impl AsRef<Matrix>) -> Matrix {
        let mut m = input.as_ref().to_owned();
        for i in 0..self.weights.len() {
            m = m.dot_mul(&self.weights[i]).transpose() + self.biases[i].clone();
            m = m.transpose();
        }

        self.activation.activate(&m)
    }

    pub fn backward_propagation(
        &mut self,
        activated: impl AsRef<Matrix>,
        correct: impl AsRef<Matrix>,
    ) {
        let loss = correct.as_ref().to_owned() - activated.as_ref().to_owned();
        // https://en.wikipedia.org/wiki/Backpropagation#Example_loss_function
        let d = loss.data().par_iter().map(|d| (*d).abs().powf(2.0) / 2.0);
        let loss = Matrix::with_vec(loss.cols(), loss.rows(), d.collect());
        let dev = self.activation.derivative(loss);
        for i in (0..self.weights.len() - 1).rev() {}
        todo!("Tomorrow is another day")
    }
}
