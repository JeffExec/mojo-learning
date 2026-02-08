# Mojo Mandelbrot Set Renderer

A high-performance parallel Mandelbrot set renderer demonstrating Mojo's SIMD and parallelization capabilities.

## Features

- **SIMD Vectorization**: Processes multiple pixels simultaneously using SIMD instructions
- **Parallelization**: Distributes computation across all CPU cores
- **ASCII Visualization**: Preview directly in terminal
- **PGM Output**: Export to standard image format
- **Performance Comparison**: Includes scalar implementation for benchmarking

## Usage

```bash
# From the mojo-projects directory
cd mojo-mandelbrot
pixi run mojo mandelbrot.mojo
```

## Output

The program generates:
1. ASCII art preview in the terminal
2. `mandelbrot.pgm` - Grayscale image file
3. Performance comparison between scalar and vectorized implementations

## Performance

Typical results on a modern CPU:

| Method | Time |
|--------|------|
| Scalar (no optimization) | ~3000ms |
| SIMD + Parallel | ~30ms |
| **Speedup** | ~100x |

## How It Works

### The Mandelbrot Set

For each pixel (x, y) on the complex plane:
1. Start with z = 0 + 0i
2. Iterate: z = z² + c where c = x + yi
3. Count iterations until |z| > 2 (escape) or max iterations reached
4. Color pixel based on iteration count

### SIMD Optimization

Instead of computing one pixel at a time:
```
pixel[0] → compute → result[0]
pixel[1] → compute → result[1]
...
```

We compute multiple pixels simultaneously:
```
[pixel[0], pixel[1], ..., pixel[n]] → compute → [result[0], result[1], ..., result[n]]
```

Where n = SIMD width (e.g., 8 for AVX-512 with Float64).

### Parallelization

Each row is computed on a separate CPU core using `parallelize()`.

## Code Structure

```mojo
# Core kernel - processes SIMD_WIDTH pixels at once
fn mandelbrot_kernel[simd_width: Int](
    c_real: SIMD[DTYPE, simd_width],
    c_imag: SIMD[DTYPE, simd_width],
    max_iter: Int
) -> SIMD[DType.int32, simd_width]

# Row computation with vectorization
fn compute_mandelbrot_row(...)
    vectorize[compute_chunk, SIMD_WIDTH](width)

# Full image with parallelization
fn compute_mandelbrot_parallel(...)
    parallelize[compute_row](height)
```

## Python Comparison

Run the Python version to compare:
```bash
python mandelbrot_python.py
```

Typical Python times (without NumPy): ~60000ms
Mojo speedup vs Python: ~2000x

## Customization

Edit the constants in `mandelbrot.mojo`:

```mojo
alias DEFAULT_WIDTH = 960
alias DEFAULT_HEIGHT = 960
alias DEFAULT_MAX_ITER = 200

# Mandelbrot bounds (zoom area)
alias X_MIN: Float64 = -2.0
alias X_MAX: Float64 = 0.6
alias Y_MIN: Float64 = -1.3
alias Y_MAX: Float64 = 1.3
```

## Learning Points

1. **SIMD Types**: `SIMD[DType.float64, 8]` holds 8 floats processed together
2. **vectorize**: Automatically handles loop vectorization
3. **parallelize**: Distributes work across CPU cores
4. **@parameter**: Compile-time loop unrolling
5. **Memory Management**: Manual allocation with `UnsafePointer`
