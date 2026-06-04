# Neural Networks

A college project that uses calculus as its backend

Neural networks use backpropagation

Backpropagation is a calculus application

## TODOs

- [ ] Matrices
  - [ ] Static Init
    - [x] matrix!\[0, 1, ...; n, n1, ...\]
    - [ ] matrix!\[idx, @ len\]
  - [ ] Random Init
    - [x] HE Uniform (Idk if I implemented this right)
    - [ ] Glorot Uniform
- [x] Matrix Operations
  - [x] Addition
  - [x] Subtraction
  - [x] Dot Product
  - [x] Hadamard Product
  - [x] Hadamard Quotient
  - [x] Column sum
- [x] Neural Network
  - [x] Forward Propagation
  - [x] Backward Propagation
  - [x] Save (Should this be compressed or be kept as json?)
  - [x] Load
- [ ] Activation Functions (Haven't seen the ones unchecked work)
  - [ ] LU
  - [ ] ReLU
  - [ ] Leaky ReLU
  - [x] Sigmoid
  - [ ] Tanh
  - [x] Softplus
  - [x] Softmax (It goes beyond 1???)
- [ ] Learning Optimizers
  - [x] None
  - [x] SGD (Stochastic Gradient Descent)
  - [ ] Adam (Adaptive Moment Estimation) - The most used
  - [ ] AdaGrad
  - [x] RMSProp (Root Mean Square Propagation)

## References

- [Dot Product Verified With This](https://www.symbolab.com/solver/vector-dot-product-calculator)
- [Matrix Macro](https://users.rust-lang.org/t/solved-optional-trailing-macro-delimiter/27657)
- [NNL](https://github.com/hotplugindev/NNL)
