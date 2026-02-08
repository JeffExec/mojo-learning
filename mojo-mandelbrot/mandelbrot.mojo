"""
Parallel Mandelbrot Set Renderer in Mojo

Demonstrates SIMD operations and compile-time metaprogramming.

Note: The API changes frequently in Mojo nightly builds.
This version uses basic SIMD operations that work with Mojo 0.26+.
"""

from time import perf_counter_ns

# Configuration
comptime DEFAULT_WIDTH = 80
comptime DEFAULT_HEIGHT = 40
comptime DEFAULT_MAX_ITER = 100

# Mandelbrot bounds
comptime X_MIN: Float64 = -2.0
comptime X_MAX: Float64 = 0.6
comptime Y_MIN: Float64 = -1.3
comptime Y_MAX: Float64 = 1.3


fn mandelbrot_scalar(c_real: Float64, c_imag: Float64, max_iter: Int) -> Int:
    """Compute Mandelbrot iterations for a single point.
    
    Uses the escape-time algorithm:
    z = 0
    for each iteration:
        z = z² + c
        if |z| > 2: escape
    """
    var z_real: Float64 = 0.0
    var z_imag: Float64 = 0.0
    
    for i in range(max_iter):
        var z_real_sq = z_real * z_real
        var z_imag_sq = z_imag * z_imag
        
        # Escape condition: |z|² > 4
        if z_real_sq + z_imag_sq > 4.0:
            return i
        
        # z = z² + c
        var z_real_imag = z_real * z_imag
        z_real = z_real_sq - z_imag_sq + c_real
        z_imag = 2.0 * z_real_imag + c_imag
    
    return max_iter


fn mandelbrot_simd(
    c_real: SIMD[DType.float64, 4], 
    c_imag: SIMD[DType.float64, 4], 
    max_iter: Int
) -> SIMD[DType.int32, 4]:
    """SIMD version: compute 4 points at once."""
    var z_real = SIMD[DType.float64, 4](0.0)
    var z_imag = SIMD[DType.float64, 4](0.0)
    var iters = SIMD[DType.int32, 4](0)
    comptime escape_threshold: Float64 = 4.0
    
    for _ in range(max_iter):
        var z_real_sq = z_real * z_real
        var z_imag_sq = z_imag * z_imag
        var magnitude_sq = z_real_sq + z_imag_sq
        
        # Check which points have not escaped
        # Update iteration count for non-escaped points
        @parameter
        for j in range(4):
            if magnitude_sq[j] <= escape_threshold:
                iters[j] = iters[j] + 1
        
        # Update z
        var z_real_imag = z_real * z_imag
        z_real = z_real_sq - z_imag_sq + c_real
        z_imag = 2.0 * z_real_imag + c_imag
    
    return iters


fn compute_mandelbrot(
    width: Int,
    height: Int,
    max_iter: Int,
) -> Float64:
    """Compute Mandelbrot set and return time in ms."""
    var x_scale = (X_MAX - X_MIN) / Float64(width)
    var y_scale = (Y_MAX - Y_MIN) / Float64(height)
    
    # ASCII characters for visualization (dark to bright)
    comptime CHARS = " .:-=+*#%@"
    
    var start = perf_counter_ns()
    
    for row in range(height):
        var y = Y_MIN + Float64(row) * y_scale
        var line = String("")
        
        var col = 0
        while col < width:
            # Process 4 points at a time with SIMD when possible
            if col + 4 <= width:
                # Create SIMD vectors for 4 consecutive x values
                var c_real = SIMD[DType.float64, 4](
                    X_MIN + Float64(col) * x_scale,
                    X_MIN + Float64(col + 1) * x_scale,
                    X_MIN + Float64(col + 2) * x_scale,
                    X_MIN + Float64(col + 3) * x_scale
                )
                var c_imag = SIMD[DType.float64, 4](y)
                
                var iters = mandelbrot_simd(c_real, c_imag, max_iter)
                
                @parameter
                for j in range(4):
                    var char_idx = (Int(iters[j]) * (len(CHARS) - 1)) // max_iter
                    line += CHARS[char_idx]
                
                col += 4
            else:
                # Scalar fallback for remaining points
                var x = X_MIN + Float64(col) * x_scale
                var iters = mandelbrot_scalar(x, y, max_iter)
                var char_idx = (iters * (len(CHARS) - 1)) // max_iter
                line += CHARS[char_idx]
                col += 1
        
        print(line)
    
    var end = perf_counter_ns()
    return Float64(end - start) / 1e6


fn compute_mandelbrot_scalar_only(
    width: Int,
    height: Int,
    max_iter: Int,
) -> Float64:
    """Scalar-only version for comparison (no output)."""
    var x_scale = (X_MAX - X_MIN) / Float64(width)
    var y_scale = (Y_MAX - Y_MIN) / Float64(height)
    
    var start = perf_counter_ns()
    var total_iters = 0
    
    for row in range(height):
        var y = Y_MIN + Float64(row) * y_scale
        for col in range(width):
            var x = X_MIN + Float64(col) * x_scale
            total_iters += mandelbrot_scalar(x, y, max_iter)
    
    var end = perf_counter_ns()
    return Float64(end - start) / 1e6


fn main():
    var width = DEFAULT_WIDTH
    var height = DEFAULT_HEIGHT
    var max_iter = DEFAULT_MAX_ITER
    
    print("=" * 60)
    print("Mojo Mandelbrot Set Renderer")
    print("=" * 60)
    print()
    print("Configuration:")
    print("  Resolution:", width, "x", height)
    print("  Max iterations:", max_iter)
    print()
    
    print("Rendering with SIMD (4 points at a time):")
    print("-" * width)
    var time_simd = compute_mandelbrot(width, height, max_iter)
    print("-" * width)
    print()
    print("SIMD render time:", time_simd, "ms")
    print()
    
    # Scalar comparison (higher resolution for better timing)
    comptime BENCH_WIDTH = 200
    comptime BENCH_HEIGHT = 100
    
    print("Benchmarking scalar vs SIMD at", BENCH_WIDTH, "x", BENCH_HEIGHT, ":")
    
    # Scalar benchmark
    var scalar_start = perf_counter_ns()
    var scalar_total = 0
    var x_scale = (X_MAX - X_MIN) / Float64(BENCH_WIDTH)
    var y_scale = (Y_MAX - Y_MIN) / Float64(BENCH_HEIGHT)
    
    for row in range(BENCH_HEIGHT):
        var y = Y_MIN + Float64(row) * y_scale
        for col in range(BENCH_WIDTH):
            var x = X_MIN + Float64(col) * x_scale
            scalar_total += mandelbrot_scalar(x, y, max_iter)
    var scalar_time = Float64(perf_counter_ns() - scalar_start) / 1e6
    
    # SIMD benchmark
    var simd_start = perf_counter_ns()
    var simd_total = 0
    
    for row in range(BENCH_HEIGHT):
        var y = Y_MIN + Float64(row) * y_scale
        var col = 0
        while col + 4 <= BENCH_WIDTH:
            var c_real = SIMD[DType.float64, 4](
                X_MIN + Float64(col) * x_scale,
                X_MIN + Float64(col + 1) * x_scale,
                X_MIN + Float64(col + 2) * x_scale,
                X_MIN + Float64(col + 3) * x_scale
            )
            var c_imag = SIMD[DType.float64, 4](y)
            var iters = mandelbrot_simd(c_real, c_imag, max_iter)
            simd_total += Int(iters[0]) + Int(iters[1]) + Int(iters[2]) + Int(iters[3])
            col += 4
        
        while col < BENCH_WIDTH:
            var x = X_MIN + Float64(col) * x_scale
            simd_total += mandelbrot_scalar(x, y, max_iter)
            col += 1
    var simd_time = Float64(perf_counter_ns() - simd_start) / 1e6
    
    print()
    print("=" * 60)
    print("Results:")
    print("  Scalar time:", scalar_time, "ms")
    print("  SIMD time:  ", simd_time, "ms")
    print("  Speedup:    ", scalar_time / simd_time, "x")
    print("=" * 60)
