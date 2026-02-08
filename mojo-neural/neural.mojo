"""
Simple Neural Network in Mojo

Demonstrates:
- SIMD-accelerated tensor operations
- Forward and backward propagation
- Gradient descent optimization
- XOR problem as a test case

This is an educational implementation showing how to build
neural network primitives in Mojo with high performance.
"""

from algorithm import vectorize, parallelize
from math import exp, sqrt, log
from memory import UnsafePointer
from random import rand, randn
from sys.info import simdwidthof
from time import perf_counter_ns

alias DTYPE = DType.float32
alias SIMD_WIDTH = simdwidthof[DTYPE]()


# =============================================================================
# Tensor Operations
# =============================================================================

struct Tensor[dtype: DType = DType.float32](Copyable, Movable):
    """A simple 2D tensor with SIMD operations."""
    var data: UnsafePointer[Scalar[dtype]]
    var rows: Int
    var cols: Int
    
    fn __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    
    fn __init__(out self, rows: Int, cols: Int, fill_value: Scalar[dtype]):
        self.__init__(rows, cols)
        self.fill(fill_value)
    
    fn __copyinit__(out self, existing: Self):
        self.rows = existing.rows
        self.cols = existing.cols
        self.data = UnsafePointer[Scalar[dtype]].alloc(self.rows * self.cols)
        for i in range(self.rows * self.cols):
            self.data[i] = existing.data[i]
    
    fn __moveinit__(out self, deinit existing: Self):
        self.rows = existing.rows
        self.cols = existing.cols
        self.data = existing.data
    
    fn __del__(deinit self):
        self.data.free()
    
    fn __getitem__(self, row: Int, col: Int) -> Scalar[dtype]:
        return self.data[row * self.cols + col]
    
    fn __setitem__(mut self, row: Int, col: Int, value: Scalar[dtype]):
        self.data[row * self.cols + col] = value
    
    fn size(self) -> Int:
        return self.rows * self.cols
    
    fn fill(mut self, value: Scalar[dtype]):
        for i in range(self.size()):
            self.data[i] = value
    
    fn fill_random(mut self, scale: Scalar[dtype] = 1.0):
        """Fill with random values scaled by Xavier initialization."""
        rand(self.data, self.size())
        var factor = scale * sqrt(Scalar[dtype](2.0) / Scalar[dtype](self.cols))
        for i in range(self.size()):
            self.data[i] = (self.data[i] * 2.0 - 1.0) * factor
    
    fn add(self, other: Self) -> Self:
        """Element-wise addition."""
        var result = Self(self.rows, self.cols)
        alias width = simdwidthof[dtype]()
        
        @parameter
        fn add_simd[w: Int](i: Int):
            var a = self.data.load[width=w](i)
            var b = other.data.load[width=w](i)
            result.data.store(i, a + b)
        
        vectorize[add_simd, width](self.size())
        return result
    
    fn subtract(self, other: Self) -> Self:
        """Element-wise subtraction."""
        var result = Self(self.rows, self.cols)
        alias width = simdwidthof[dtype]()
        
        @parameter
        fn sub_simd[w: Int](i: Int):
            var a = self.data.load[width=w](i)
            var b = other.data.load[width=w](i)
            result.data.store(i, a - b)
        
        vectorize[sub_simd, width](self.size())
        return result
    
    fn multiply(self, other: Self) -> Self:
        """Element-wise multiplication."""
        var result = Self(self.rows, self.cols)
        alias width = simdwidthof[dtype]()
        
        @parameter
        fn mul_simd[w: Int](i: Int):
            var a = self.data.load[width=w](i)
            var b = other.data.load[width=w](i)
            result.data.store(i, a * b)
        
        vectorize[mul_simd, width](self.size())
        return result
    
    fn scale(self, scalar: Scalar[dtype]) -> Self:
        """Multiply all elements by a scalar."""
        var result = Self(self.rows, self.cols)
        for i in range(self.size()):
            result.data[i] = self.data[i] * scalar
        return result
    
    fn matmul(self, other: Self) -> Self:
        """Matrix multiplication: (m, n) @ (n, p) -> (m, p)"""
        var result = Self(self.rows, other.cols, Scalar[dtype](0))
        
        for i in range(self.rows):
            for j in range(other.cols):
                var sum: Scalar[dtype] = 0
                for k in range(self.cols):
                    sum += self[i, k] * other[k, j]
                result[i, j] = sum
        
        return result
    
    fn transpose(self) -> Self:
        """Return the transpose."""
        var result = Self(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result[j, i] = self[i, j]
        return result
    
    fn sum(self) -> Scalar[dtype]:
        """Sum all elements."""
        var total: Scalar[dtype] = 0
        for i in range(self.size()):
            total += self.data[i]
        return total
    
    fn mean(self) -> Scalar[dtype]:
        """Mean of all elements."""
        return self.sum() / Scalar[dtype](self.size())
    
    fn print_tensor(self, name: String = "Tensor"):
        """Pretty print."""
        print(name, "(", self.rows, "x", self.cols, "):")
        for i in range(min(self.rows, 5)):
            var row = String("  [")
            for j in range(min(self.cols, 5)):
                row += String(self[i, j])
                if j < min(self.cols, 5) - 1:
                    row += ", "
            if self.cols > 5:
                row += ", ..."
            row += "]"
            print(row)
        if self.rows > 5:
            print("  ...")


# =============================================================================
# Activation Functions
# =============================================================================

fn sigmoid[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Sigmoid activation: 1 / (1 + exp(-x))"""
    return Scalar[dtype](1.0) / (Scalar[dtype](1.0) + exp(-x))

fn sigmoid_derivative[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))"""
    var s = sigmoid[dtype](x)
    return s * (Scalar[dtype](1.0) - s)

fn relu[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """ReLU activation: max(0, x)"""
    return max(Scalar[dtype](0), x)

fn relu_derivative[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Derivative of ReLU: 1 if x > 0 else 0"""
    if x > Scalar[dtype](0):
        return Scalar[dtype](1.0)
    return Scalar[dtype](0.0)

fn apply_activation[
    dtype: DType,
    func: fn[DType](Scalar[DType]) -> Scalar[DType]
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """Apply activation function element-wise."""
    var result = Tensor[dtype](tensor.rows, tensor.cols)
    for i in range(tensor.size()):
        result.data[i] = func[dtype](tensor.data[i])
    return result


# =============================================================================
# Loss Functions
# =============================================================================

fn mse_loss[dtype: DType](predictions: Tensor[dtype], targets: Tensor[dtype]) -> Scalar[dtype]:
    """Mean Squared Error loss."""
    var diff = predictions.subtract(targets)
    var sq = diff.multiply(diff)
    return sq.mean()

fn mse_loss_gradient[dtype: DType](predictions: Tensor[dtype], targets: Tensor[dtype]) -> Tensor[dtype]:
    """Gradient of MSE loss with respect to predictions."""
    var diff = predictions.subtract(targets)
    return diff.scale(Scalar[dtype](2.0) / Scalar[dtype](predictions.size()))


# =============================================================================
# Dense Layer
# =============================================================================

struct DenseLayer[dtype: DType = DType.float32]:
    """A fully connected neural network layer."""
    var weights: Tensor[dtype]
    var biases: Tensor[dtype]
    var input_cache: Tensor[dtype]      # Cache for backprop
    var preactivation_cache: Tensor[dtype]  # Cache for backprop
    var use_relu: Bool
    
    fn __init__(out self, input_size: Int, output_size: Int, use_relu: Bool = True):
        self.weights = Tensor[dtype](input_size, output_size)
        self.biases = Tensor[dtype](1, output_size, Scalar[dtype](0))
        self.input_cache = Tensor[dtype](1, input_size)
        self.preactivation_cache = Tensor[dtype](1, output_size)
        self.use_relu = use_relu
        
        # Xavier initialization
        self.weights.fill_random(Scalar[dtype](1.0))
    
    fn forward(mut self, x: Tensor[dtype]) -> Tensor[dtype]:
        """Forward pass: output = activation(x @ W + b)"""
        # Cache input for backprop
        self.input_cache = x.copy()
        
        # Linear transformation
        var z = x.matmul(self.weights)
        
        # Add bias (broadcast across batch)
        for i in range(z.rows):
            for j in range(z.cols):
                z[i, j] = z[i, j] + self.biases[0, j]
        
        # Cache pre-activation for backprop
        self.preactivation_cache = z.copy()
        
        # Activation
        if self.use_relu:
            return apply_activation[dtype, relu](z)
        else:
            return apply_activation[dtype, sigmoid](z)
    
    fn backward(mut self, grad_output: Tensor[dtype], learning_rate: Scalar[dtype]) -> Tensor[dtype]:
        """Backward pass with gradient descent update."""
        # Apply activation derivative
        var grad_preact = Tensor[dtype](grad_output.rows, grad_output.cols)
        for i in range(grad_output.size()):
            var act_deriv: Scalar[dtype]
            if self.use_relu:
                act_deriv = relu_derivative[dtype](self.preactivation_cache.data[i])
            else:
                act_deriv = sigmoid_derivative[dtype](self.preactivation_cache.data[i])
            grad_preact.data[i] = grad_output.data[i] * act_deriv
        
        # Gradient w.r.t. weights: input^T @ grad_preact
        var grad_weights = self.input_cache.transpose().matmul(grad_preact)
        
        # Gradient w.r.t. biases: sum over batch
        var grad_biases = Tensor[dtype](1, self.biases.cols, Scalar[dtype](0))
        for i in range(grad_preact.rows):
            for j in range(grad_preact.cols):
                grad_biases[0, j] = grad_biases[0, j] + grad_preact[i, j]
        
        # Gradient w.r.t. input: grad_preact @ weights^T
        var grad_input = grad_preact.matmul(self.weights.transpose())
        
        # Update parameters
        for i in range(self.weights.size()):
            self.weights.data[i] -= learning_rate * grad_weights.data[i]
        
        for i in range(self.biases.size()):
            self.biases.data[i] -= learning_rate * grad_biases.data[i]
        
        return grad_input


# =============================================================================
# Simple Neural Network
# =============================================================================

struct NeuralNetwork[dtype: DType = DType.float32]:
    """A simple 2-layer neural network for demonstration."""
    var layer1: DenseLayer[dtype]
    var layer2: DenseLayer[dtype]
    
    fn __init__(out self, input_size: Int, hidden_size: Int, output_size: Int):
        self.layer1 = DenseLayer[dtype](input_size, hidden_size, use_relu=True)
        self.layer2 = DenseLayer[dtype](hidden_size, output_size, use_relu=False)
    
    fn forward(mut self, x: Tensor[dtype]) -> Tensor[dtype]:
        """Forward pass through network."""
        var h = self.layer1.forward(x)
        return self.layer2.forward(h)
    
    fn train_step(
        mut self, 
        x: Tensor[dtype], 
        y: Tensor[dtype], 
        learning_rate: Scalar[dtype]
    ) -> Scalar[dtype]:
        """Single training step. Returns loss."""
        # Forward pass
        var predictions = self.forward(x)
        
        # Compute loss
        var loss = mse_loss[dtype](predictions, y)
        
        # Backward pass
        var grad = mse_loss_gradient[dtype](predictions, y)
        grad = self.layer2.backward(grad, learning_rate)
        _ = self.layer1.backward(grad, learning_rate)
        
        return loss


# =============================================================================
# Main: Train on XOR problem
# =============================================================================

fn main() raises:
    print("=" * 60)
    print("Mojo Neural Network - XOR Problem")
    print("=" * 60)
    print()
    
    # XOR dataset
    var X = Tensor[DTYPE](4, 2)
    X[0, 0] = 0; X[0, 1] = 0
    X[1, 0] = 0; X[1, 1] = 1
    X[2, 0] = 1; X[2, 1] = 0
    X[3, 0] = 1; X[3, 1] = 1
    
    var Y = Tensor[DTYPE](4, 1)
    Y[0, 0] = 0  # 0 XOR 0 = 0
    Y[1, 0] = 1  # 0 XOR 1 = 1
    Y[2, 0] = 1  # 1 XOR 0 = 1
    Y[3, 0] = 0  # 1 XOR 1 = 0
    
    print("Training Data (XOR):")
    print("  Input -> Output")
    print("  [0, 0] -> 0")
    print("  [0, 1] -> 1")
    print("  [1, 0] -> 1")
    print("  [1, 1] -> 0")
    print()
    
    # Create network
    var net = NeuralNetwork[DTYPE](2, 8, 1)
    
    # Training parameters
    alias EPOCHS = 10000
    alias LEARNING_RATE: Float32 = 0.5
    alias PRINT_EVERY = 1000
    
    print("Training for", EPOCHS, "epochs...")
    print()
    
    var start = perf_counter_ns()
    
    for epoch in range(EPOCHS):
        var loss = net.train_step(X, Y, LEARNING_RATE)
        
        if epoch % PRINT_EVERY == 0:
            print("Epoch", epoch, "- Loss:", loss)
    
    var elapsed = Float64(perf_counter_ns() - start) / 1e6
    
    print()
    print("Training complete in", elapsed, "ms")
    print()
    
    # Test the network
    print("=" * 60)
    print("Testing:")
    print("=" * 60)
    
    var predictions = net.forward(X)
    
    print()
    print("Input\t\tTarget\tPrediction\tRounded")
    print("-" * 50)
    
    for i in range(4):
        var pred = predictions[i, 0]
        var rounded = Int(pred + 0.5)  # Round to 0 or 1
        print(
            "[", X[i, 0], ",", X[i, 1], "]",
            "\t", Y[i, 0],
            "\t", pred,
            "\t", rounded
        )
    
    print()
    
    # Check accuracy
    var correct = 0
    for i in range(4):
        var pred_rounded = Int(predictions[i, 0] + 0.5)
        var target = Int(Y[i, 0])
        if pred_rounded == target:
            correct += 1
    
    var accuracy = Float64(correct) / 4.0 * 100.0
    print("Accuracy:", accuracy, "%")
    print()
    print("=" * 60)
