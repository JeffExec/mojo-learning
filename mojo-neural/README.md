# Mojo Neural Network

A simple neural network implementation demonstrating SIMD-accelerated tensor operations and backpropagation in Mojo.

## Features

- **Tensor Operations**: Add, subtract, multiply, matmul, transpose
- **Activation Functions**: Sigmoid, ReLU with derivatives
- **Loss Functions**: Mean Squared Error (MSE)
- **Dense Layers**: Fully connected layers with forward/backward pass
- **Training**: Gradient descent optimization
- **Test Case**: XOR problem (classic neural network benchmark)

## Usage

```bash
cd mojo-neural
pixi run mojo neural.mojo
```

## Output

```
Training Data (XOR):
  Input -> Output
  [0, 0] -> 0
  [0, 1] -> 1
  [1, 0] -> 1
  [1, 1] -> 0

Training for 10000 epochs...

Epoch 0 - Loss: 0.25...
Epoch 1000 - Loss: 0.12...
...
Epoch 9000 - Loss: 0.001...

Testing:
Input       Target  Prediction  Rounded
[0, 0]      0       0.02        0
[0, 1]      1       0.97        1
[1, 0]      1       0.98        1
[1, 1]      0       0.03        0

Accuracy: 100.0 %
```

## Architecture

```
Input (2) → Dense (8, ReLU) → Dense (1, Sigmoid) → Output
```

## Key Components

### Tensor Struct

```mojo
struct Tensor[dtype: DType]:
    var data: UnsafePointer[Scalar[dtype]]
    var rows: Int
    var cols: Int
    
    fn matmul(self, other: Self) -> Self: ...
    fn add(self, other: Self) -> Self: ...
    fn transpose(self) -> Self: ...
```

### Dense Layer

```mojo
struct DenseLayer[dtype: DType]:
    var weights: Tensor[dtype]
    var biases: Tensor[dtype]
    
    fn forward(mut self, x: Tensor) -> Tensor: ...
    fn backward(mut self, grad: Tensor, lr: Scalar) -> Tensor: ...
```

### Activation Functions

```mojo
fn sigmoid[dtype](x: Scalar[dtype]) -> Scalar[dtype]:
    return 1.0 / (1.0 + exp(-x))

fn relu[dtype](x: Scalar[dtype]) -> Scalar[dtype]:
    return max(0, x)
```

## Learning Points

1. **Generic Structs**: `Tensor[dtype: DType]` for different precisions
2. **SIMD Operations**: Vectorized element-wise operations
3. **Manual Memory**: `UnsafePointer` for tensor data
4. **Backpropagation**: Chain rule through layers
5. **Caching**: Store activations for backward pass
6. **Xavier Initialization**: Proper weight initialization

## Why XOR?

XOR is a classic test for neural networks because:
- Cannot be solved by a single perceptron (not linearly separable)
- Requires at least one hidden layer
- Small enough to train quickly
- Easy to verify correctness

## Limitations

This is an educational implementation. For production use, consider:
- Batch processing
- More efficient matmul (tiled/blocked)
- GPU acceleration
- Proper random initialization
- Additional optimizers (Adam, SGD with momentum)

## Next Steps

- [ ] Add batch normalization
- [ ] Implement dropout
- [ ] Add convolutional layers
- [ ] GPU support via kernel functions
- [ ] MNIST digit classification
