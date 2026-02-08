# Mojo Learning Projects

A collection of practical Mojo projects demonstrating high-performance programming techniques.

## 🚀 Quick Start

```bash
# Install pixi (package manager)
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"

# Clone and run
git clone https://github.com/JeffExec/mojo-learning.git
cd mojo-learning
pixi shell

# Run the Mandelbrot demo
mojo mojo-mandelbrot/mandelbrot.mojo
```

## 📁 Projects

### [mojo-mandelbrot](mojo-mandelbrot/) - Fractal Renderer
SIMD-accelerated Mandelbrot set visualization
- Processes 4 pixels simultaneously with SIMD
- ASCII art output
- Performance comparison (scalar vs SIMD)

### [mojo-matrix](mojo-matrix/) - Matrix Library
High-performance matrix operations
- SIMD element-wise operations
- Multiple matmul implementations
- Statistics (sum, mean, norm)

### [mojo-neural](mojo-neural/) - Neural Network
Simple feedforward neural network
- Tensor operations
- Backpropagation
- XOR learning demo

## 🔑 Key Concepts Demonstrated

- **SIMD Vectorization**: Process multiple data points per instruction
- **Compile-time Metaprogramming**: `comptime` and `@parameter`
- **Generic Structs**: Type-parameterized data structures
- **Memory Management**: `UnsafePointer` for manual control
- **Performance Benchmarking**: Comparing implementations

## 📚 Learning Resources

Comprehensive documentation in the companion learning directory:
- Language Fundamentals
- Structs and Traits
- Ownership and Memory
- SIMD and Performance
- Metaprogramming
- GPU Programming

## ⚠️ Note on API Changes

Mojo is a rapidly evolving language. The projects use the nightly build (0.26+) and the API may change. If you encounter errors:
1. Check `mojo --version`
2. Consult the [official docs](https://docs.modular.com/mojo/manual/)
3. Open an issue in this repository

## 📖 Requirements

- **Mojo**: 0.26+ (installed via pixi)
- **OS**: Linux, macOS, or WSL2
- **Optional**: NVIDIA/AMD GPU for GPU examples
