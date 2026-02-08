"""
SIMD-Accelerated Matrix Operations Library for Mojo

This library provides high-performance matrix operations using:
- SIMD vectorization for element-wise operations
- Parallelization for large matrices
- Tiled matrix multiplication for cache efficiency

Features:
- Matrix creation and initialization
- Element-wise operations (add, subtract, multiply)
- Matrix multiplication (naive, vectorized, tiled)
- Transpose
- Performance benchmarks against naive implementations
"""

from algorithm import vectorize, parallelize
from math import sqrt, iota
from memory import UnsafePointer, memset_zero
from sys.info import simdwidthof, num_physical_cores
from time import perf_counter_ns
from random import rand

alias DTYPE = DType.float64
alias SIMD_WIDTH = simdwidthof[DTYPE]()


struct Matrix[dtype: DType = DType.float64](Copyable, Movable):
    """A 2D matrix with SIMD-accelerated operations.
    
    Parameters:
        dtype: The data type of matrix elements (default: float64)
    
    Example:
        var m = Matrix[DType.float32](3, 3)
        m.fill(1.0)
        print(m)
    """
    var data: UnsafePointer[Scalar[dtype]]
    var rows: Int
    var cols: Int
    
    fn __init__(out self, rows: Int, cols: Int):
        """Create an uninitialized matrix."""
        self.rows = rows
        self.cols = cols
        self.data = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    
    fn __init__(out self, rows: Int, cols: Int, fill_value: Scalar[dtype]):
        """Create a matrix filled with a specific value."""
        self.__init__(rows, cols)
        self.fill(fill_value)
    
    fn __copyinit__(out self, existing: Self):
        """Deep copy the matrix."""
        self.rows = existing.rows
        self.cols = existing.cols
        self.data = UnsafePointer[Scalar[dtype]].alloc(self.rows * self.cols)
        for i in range(self.rows * self.cols):
            self.data[i] = existing.data[i]
    
    fn __moveinit__(out self, deinit existing: Self):
        """Move the matrix."""
        self.rows = existing.rows
        self.cols = existing.cols
        self.data = existing.data
    
    fn __del__(deinit self):
        """Free the matrix memory."""
        self.data.free()
    
    fn __getitem__(self, row: Int, col: Int) -> Scalar[dtype]:
        """Get element at (row, col)."""
        return self.data[row * self.cols + col]
    
    fn __setitem__(mut self, row: Int, col: Int, value: Scalar[dtype]):
        """Set element at (row, col)."""
        self.data[row * self.cols + col] = value
    
    fn fill(mut self, value: Scalar[dtype]):
        """Fill the matrix with a value."""
        alias width = simdwidthof[dtype]()
        
        @parameter
        fn fill_simd[w: Int](i: Int):
            self.data.store(i, SIMD[dtype, w](value))
        
        vectorize[fill_simd, width](self.rows * self.cols)
    
    fn fill_random(mut self):
        """Fill the matrix with random values in [0, 1)."""
        rand(self.data, self.rows * self.cols)
    
    fn fill_identity(mut self):
        """Fill as identity matrix (must be square)."""
        self.fill(Scalar[dtype](0))
        var min_dim = min(self.rows, self.cols)
        for i in range(min_dim):
            self[i, i] = Scalar[dtype](1)
    
    fn zeros() -> Self:
        """Create a matrix of zeros."""
        return Self(0, 0, Scalar[dtype](0))
    
    @staticmethod
    fn identity(size: Int) -> Self:
        """Create an identity matrix."""
        var m = Self(size, size, Scalar[dtype](0))
        for i in range(size):
            m[i, i] = Scalar[dtype](1)
        return m
    
    fn shape(self) -> Tuple[Int, Int]:
        """Return matrix dimensions as (rows, cols)."""
        return (self.rows, self.cols)
    
    fn size(self) -> Int:
        """Return total number of elements."""
        return self.rows * self.cols
    
    # Element-wise operations with SIMD
    fn add(self, other: Self) -> Self:
        """Element-wise addition."""
        var result = Self(self.rows, self.cols)
        alias width = simdwidthof[dtype]()
        
        @parameter
        fn add_simd[w: Int](i: Int):
            var a = self.data.load[width=w](i)
            var b = other.data.load[width=w](i)
            result.data.store(i, a + b)
        
        vectorize[add_simd, width](self.rows * self.cols)
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
        
        vectorize[sub_simd, width](self.rows * self.cols)
        return result
    
    fn multiply_elementwise(self, other: Self) -> Self:
        """Element-wise (Hadamard) multiplication."""
        var result = Self(self.rows, self.cols)
        alias width = simdwidthof[dtype]()
        
        @parameter
        fn mul_simd[w: Int](i: Int):
            var a = self.data.load[width=w](i)
            var b = other.data.load[width=w](i)
            result.data.store(i, a * b)
        
        vectorize[mul_simd, width](self.rows * self.cols)
        return result
    
    fn scale(self, scalar: Scalar[dtype]) -> Self:
        """Multiply all elements by a scalar."""
        var result = Self(self.rows, self.cols)
        alias width = simdwidthof[dtype]()
        var scalar_simd = SIMD[dtype, width](scalar)
        
        @parameter
        fn scale_simd[w: Int](i: Int):
            var a = self.data.load[width=w](i)
            result.data.store(i, a * SIMD[dtype, w](scalar))
        
        vectorize[scale_simd, width](self.rows * self.cols)
        return result
    
    fn transpose(self) -> Self:
        """Return the transpose of the matrix."""
        var result = Self(self.cols, self.rows)
        
        @parameter
        fn transpose_row(i: Int):
            for j in range(self.cols):
                result[j, i] = self[i, j]
        
        parallelize[transpose_row](self.rows)
        return result
    
    # Matrix multiplication implementations
    fn matmul_naive(self, other: Self) -> Self:
        """Naive O(n³) matrix multiplication."""
        var result = Self(self.rows, other.cols, Scalar[dtype](0))
        
        for i in range(self.rows):
            for j in range(other.cols):
                var sum: Scalar[dtype] = 0
                for k in range(self.cols):
                    sum += self[i, k] * other[k, j]
                result[i, j] = sum
        
        return result
    
    fn matmul_vectorized(self, other: Self) -> Self:
        """Vectorized matrix multiplication."""
        var result = Self(self.rows, other.cols, Scalar[dtype](0))
        alias width = simdwidthof[dtype]()
        
        @parameter
        fn compute_row(i: Int):
            for j in range(other.cols):
                var sum: Scalar[dtype] = 0
                
                @parameter
                fn dot_simd[w: Int](k: Int):
                    var a = self.data.load[width=w](i * self.cols + k)
                    var b = SIMD[dtype, w]()
                    for kk in range(w):
                        b[kk] = other[k + kk, j]
                    sum += (a * b).reduce_add()
                
                vectorize[dot_simd, width](self.cols)
                result[i, j] = sum
        
        parallelize[compute_row](self.rows)
        return result
    
    fn matmul_tiled(self, other: Self) -> Self:
        """Tiled matrix multiplication for cache efficiency."""
        alias TILE_SIZE = 32  # Adjust based on cache size
        
        var result = Self(self.rows, other.cols, Scalar[dtype](0))
        
        # Tile over output matrix
        for i0 in range(0, self.rows, TILE_SIZE):
            var i_end = min(i0 + TILE_SIZE, self.rows)
            
            for j0 in range(0, other.cols, TILE_SIZE):
                var j_end = min(j0 + TILE_SIZE, other.cols)
                
                for k0 in range(0, self.cols, TILE_SIZE):
                    var k_end = min(k0 + TILE_SIZE, self.cols)
                    
                    # Compute tile
                    for i in range(i0, i_end):
                        for j in range(j0, j_end):
                            var sum = result[i, j]
                            for k in range(k0, k_end):
                                sum += self[i, k] * other[k, j]
                            result[i, j] = sum
        
        return result
    
    # Reduction operations
    fn sum(self) -> Scalar[dtype]:
        """Sum all elements."""
        alias width = simdwidthof[dtype]()
        var total = SIMD[dtype, width](0)
        
        @parameter
        fn sum_simd[w: Int](i: Int):
            total += self.data.load[width=w](i)
        
        vectorize[sum_simd, width](self.rows * self.cols)
        return total.reduce_add()
    
    fn mean(self) -> Scalar[dtype]:
        """Compute the mean of all elements."""
        return self.sum() / Scalar[dtype](self.size())
    
    fn max(self) -> Scalar[dtype]:
        """Find the maximum element."""
        var max_val = self.data[0]
        for i in range(1, self.size()):
            if self.data[i] > max_val:
                max_val = self.data[i]
        return max_val
    
    fn min(self) -> Scalar[dtype]:
        """Find the minimum element."""
        var min_val = self.data[0]
        for i in range(1, self.size()):
            if self.data[i] < min_val:
                min_val = self.data[i]
        return min_val
    
    fn frobenius_norm(self) -> Scalar[dtype]:
        """Compute the Frobenius norm (sqrt of sum of squares)."""
        var sq_sum = self.multiply_elementwise(self).sum()
        return sqrt(sq_sum)
    
    fn print_matrix(self, name: String = "Matrix"):
        """Pretty print the matrix."""
        print(name, "(", self.rows, "x", self.cols, "):")
        for i in range(min(self.rows, 10)):
            var row = String("  [")
            for j in range(min(self.cols, 10)):
                row += String(self[i, j])
                if j < min(self.cols, 10) - 1:
                    row += ", "
            if self.cols > 10:
                row += ", ..."
            row += "]"
            print(row)
        if self.rows > 10:
            print("  ...")


fn benchmark_matmul(size: Int) -> Tuple[Float64, Float64, Float64]:
    """Benchmark different matrix multiplication implementations.
    
    Returns (naive_time, vectorized_time, tiled_time) in milliseconds.
    """
    var A = Matrix[DTYPE](size, size)
    var B = Matrix[DTYPE](size, size)
    A.fill_random()
    B.fill_random()
    
    # Naive
    var start = perf_counter_ns()
    var C_naive = A.matmul_naive(B)
    var naive_time = Float64(perf_counter_ns() - start) / 1e6
    
    # Vectorized
    start = perf_counter_ns()
    var C_vec = A.matmul_vectorized(B)
    var vec_time = Float64(perf_counter_ns() - start) / 1e6
    
    # Tiled
    start = perf_counter_ns()
    var C_tiled = A.matmul_tiled(B)
    var tiled_time = Float64(perf_counter_ns() - start) / 1e6
    
    return (naive_time, vec_time, tiled_time)


fn main() raises:
    print("=" * 60)
    print("Mojo Matrix Operations Library")
    print("=" * 60)
    print()
    
    # Basic operations demo
    print("1. Basic Operations:")
    print("-" * 40)
    
    var A = Matrix[DTYPE](3, 3, 1.0)
    var B = Matrix[DTYPE](3, 3, 2.0)
    
    A.print_matrix("Matrix A")
    B.print_matrix("Matrix B")
    
    var C = A.add(B)
    C.print_matrix("A + B")
    
    var D = A.scale(3.0)
    D.print_matrix("A * 3")
    
    print()
    
    # Identity and transpose
    print("2. Identity and Transpose:")
    print("-" * 40)
    
    var I = Matrix[DTYPE].identity(3)
    I.print_matrix("Identity")
    
    var M = Matrix[DTYPE](2, 3)
    M[0, 0] = 1; M[0, 1] = 2; M[0, 2] = 3
    M[1, 0] = 4; M[1, 1] = 5; M[1, 2] = 6
    M.print_matrix("M")
    
    var Mt = M.transpose()
    Mt.print_matrix("M^T")
    
    print()
    
    # Matrix multiplication benchmark
    print("3. Matrix Multiplication Benchmark:")
    print("-" * 40)
    
    alias SIZES = List[Int](64, 128, 256)
    
    print("Size\tNaive(ms)\tVectorized(ms)\tTiled(ms)\tSpeedup")
    
    for i in range(len(SIZES)):
        var size = SIZES[i]
        var times = benchmark_matmul(size)
        var naive = times.get[0, Float64]()
        var vec = times.get[1, Float64]()
        var tiled = times.get[2, Float64]()
        var speedup = naive / min(vec, tiled)
        
        print(
            size, "\t",
            naive, "\t",
            vec, "\t",
            tiled, "\t",
            speedup, "x"
        )
    
    print()
    
    # Statistics
    print("4. Statistics Operations:")
    print("-" * 40)
    
    var stats_mat = Matrix[DTYPE](100, 100)
    stats_mat.fill_random()
    
    print("100x100 Random Matrix Statistics:")
    print("  Sum:", stats_mat.sum())
    print("  Mean:", stats_mat.mean())
    print("  Min:", stats_mat.min())
    print("  Max:", stats_mat.max())
    print("  Frobenius Norm:", stats_mat.frobenius_norm())
    
    print()
    print("=" * 60)
