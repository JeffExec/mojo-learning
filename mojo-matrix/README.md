# Mojo Matrix Library

A SIMD-accelerated matrix operations library demonstrating high-performance numerical computing in Mojo.

## Features

- **Matrix Creation**: Zeros, identity, random fill
- **Element-wise Operations**: Add, subtract, multiply, scale
- **Matrix Multiplication**: Three implementations for comparison
  - Naive O(n³) 
  - SIMD vectorized + parallelized
  - Cache-friendly tiled
- **Transpose**: Parallel row-based transpose
- **Statistics**: Sum, mean, min, max, Frobenius norm
- **Benchmarking**: Built-in performance comparison

## Usage

```bash
cd mojo-matrix
pixi run mojo matrix.mojo
```

## API Examples

```mojo
# Create matrices
var A = Matrix[DType.float64](3, 3, 1.0)  # 3x3 filled with 1.0
var B = Matrix[DType.float64](3, 3)
B.fill_random()

# Identity matrix
var I = Matrix[DType.float64].identity(3)

# Operations
var C = A.add(B)
var D = A.subtract(B)
var E = A.multiply_elementwise(B)  # Hadamard product
var F = A.scale(2.5)
var G = A.transpose()

# Matrix multiplication (choose your algorithm)
var H = A.matmul_naive(B)       # Simple, slow
var I = A.matmul_vectorized(B)  # SIMD + parallel
var J = A.matmul_tiled(B)       # Cache-friendly

# Statistics
print(A.sum())
print(A.mean())
print(A.max())
print(A.frobenius_norm())
```

## Performance

Matrix multiplication (256×256) on typical hardware:

| Method | Time | Speedup |
|--------|------|---------|
| Naive | ~800ms | 1x |
| SIMD + Parallel | ~50ms | 16x |
| Tiled | ~100ms | 8x |

Best speedup varies with matrix size and cache hierarchy.

## Implementation Details

### SIMD Element-wise Operations

```mojo
fn add(self, other: Self) -> Self:
    alias width = simdwidthof[dtype]()
    
    @parameter
    fn add_simd[w: Int](i: Int):
        var a = self.data.load[width=w](i)
        var b = other.data.load[width=w](i)
        result.data.store(i, a + b)
    
    vectorize[add_simd, width](self.size())
```

### Tiled Matrix Multiplication

```mojo
alias TILE_SIZE = 32  # Fits in L1 cache

for i0 in range(0, rows, TILE_SIZE):
    for j0 in range(0, cols, TILE_SIZE):
        for k0 in range(0, inner, TILE_SIZE):
            # Compute TILE_SIZE × TILE_SIZE block
            # Maximizes cache reuse
```

### Memory Layout

- Row-major storage (C-style)
- Contiguous memory for SIMD efficiency
- Manual allocation with `UnsafePointer`

## Future Improvements

- [ ] GPU acceleration via kernel functions
- [ ] Strassen's algorithm for large matrices
- [ ] BLAS-level optimization
- [ ] Complex number support
- [ ] Sparse matrix support

## Learning Points

1. **Generic Structs**: `Matrix[dtype: DType]` allows different precisions
2. **SIMD Load/Store**: Batch memory operations
3. **vectorize**: Automatic SIMD loop transformation
4. **parallelize**: Multi-core distribution
5. **Tiling**: Cache-aware algorithm design
6. **Traits**: `Copyable`, `Movable` for value semantics
