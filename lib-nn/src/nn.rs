use serde::{Deserialize, Serialize};

use crate::activation::Activation;
use crate::loss::LossFunction;
use lib_matrix::Matrix;

#[derive(Default, Debug, Clone, Deserialize, Serialize)]
pub struct NeuralNetwork {
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    activation: Vec<Activation>,
    loss_function: LossFunction,
    learning_rate: f64,
}

impl NeuralNetwork {
    pub fn new(
        layers: impl AsRef<[(usize, Activation)]>,
        loss_function: LossFunction,
        learning_rate: f64,
    ) -> Self {
        let (layers, activation): (Vec<usize>, Vec<Activation>) =
            layers.as_ref().into_iter().map(|(a, b)| (*a, *b)).unzip();

        let weights: Vec<Matrix> = layers
            .windows(2)
            .map(|a| Matrix::rand(a[0], a[1])) // i rows, x cols
            .collect();

        let biases: Vec<Matrix> = layers
            .windows(2)
            .map(|a| Matrix::rand(1, a[1])) // 1 row, x cols
            .collect();

        for w in &weights {
            println!("Weights: {w}");
        }

        println!("Initialized neural network with layers: {layers:?}");
        println!("\t{} {}", weights.len(), biases.len());
        Self {
            activation,
            loss_function,
            weights,
            biases,
            learning_rate,
        }
    }

    pub fn train(
        &mut self,
        epochs: usize,
        inputs: impl AsRef<Vec<Matrix>>,
        correct: impl AsRef<Vec<Matrix>>,
    ) {
        for _ in 0..epochs {
            for (i, input) in inputs.as_ref().iter().enumerate() {
                let (pre, post) = self.forward_propagation(input);
                self.backward_propagation(pre, post, &correct.as_ref()[i]);
            }
        }
    }

    pub fn guess(&mut self, input: impl AsRef<Matrix>) -> Matrix {
        let mut m = input.as_ref().to_owned();
        for (i, (w, b)) in self.weights.iter().zip(&self.biases).enumerate() {
            m = self.activation[i].activate(m * w + b);
        }

        m
    }

    fn forward_propagation(&self, input: impl AsRef<Matrix>) -> (Vec<Matrix>, Vec<Matrix>) {
        let mut m = input.as_ref().to_owned();
        let mut pre = Vec::with_capacity(self.weights.len());
        let mut post = Vec::with_capacity(self.weights.len() + 1);
        post.push(m.clone());
        for (i, (w, b)) in self.weights.iter().zip(&self.biases).enumerate() {
            let tmp = m * w + b;
            pre.push(tmp.clone());
            m = self.activation[i].activate(tmp);
            post.push(m.clone())
        }

        (pre, post)
    }

    fn backward_propagation(
        &mut self,
        pre: Vec<Matrix>,
        post: Vec<Matrix>,
        correct: impl AsRef<Matrix>,
    ) {
        let output = post.last().unwrap();
        let loss = self.loss_function.loss(output, &correct);
        println!("Loss: {loss}");
        let loss = loss * 1000.0;
        let mut error = self.loss_function.derivative(output, correct);
        for i in (0..post.len() - 1).rev() {
            let activated = error.hadamard_mul(&self.activation[i].derivative(&pre[i]));
            let weight_grad = post[i].transpose() * &activated;
            let bias_grad = activated.sum_over_rows();
            error = activated * self.weights[i].transpose();
            self.weights[i] -= weight_grad.scalar_mul(self.learning_rate * loss);
            self.biases[i] -= bias_grad.scalar_mul(self.learning_rate * loss);
        }
    }
}
