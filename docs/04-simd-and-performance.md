# Mojo SIMD and Performance

## What is SIMD?

**SIMD** = Single Instruction, Multiple Data

SIMD allows a single CPU instruction to process multiple data elements simultaneously:

```
Traditional (scalar):     SIMD (vector):
  1 → ×10 → 10            [1,2,3,4] → ×10 → [10,20,30,40]
  2 → ×10 → 20            (one instruction!)
  3 → ×10 → 30
  4 → ×10 → 40
  (four instructions)
```

---

## Mojo is SIMD-First

All numerical types in Mojo are SIMD-backed:

```mojo
# These are all equivalent:
var x: Float64 = 3.14
var y: Scalar[DType.float64] = 3.14
var z: SIMD[DType.float64, 1] = 3.14
```

The `Scalar` type is just `SIMD[size=1]`:
```mojo
alias Scalar = SIMD[size=1]
```

---

## SIMD System Properties

```mojo
from sys.info import simdbitwidth, simdbytewidth, simdwidthof

fn main():
    print(simdbitwidth())                    # e.g., 512 (bits)
    print(simdbytewidth())                   # e.g., 64 (bytes)
    print(simdwidthof[DType.float64]())      # e.g., 8 (values)
    print(simdwidthof[DType.float32]())      # e.g., 16 (values)
```

| Function | Returns |
|----------|---------|
| `simdbitwidth()` | Total bits in SIMD register (e.g., 512) |
| `simdbytewidth()` | Total bytes in SIMD register (e.g., 64) |
| `simdwidthof[T]()` | How many values of type T fit |

---

## SIMD Vectors

### Creating SIMD Vectors

```mojo
# Explicit creation
var v1 = SIMD[DType.uint8, 4](1, 2, 3, 4)

# Type annotation style
var v2: SIMD[DType.uint8, 4] = [1, 2, 3, 4]

# All zeros
var zeros = SIMD[DType.float32, 4]()  # [0, 0, 0, 0]

# Fill with single value
var ones = SIMD[DType.int32, 8](1)    # [1, 1, 1, 1, 1, 1, 1, 1]

# Sequential values with iota
from math import iota
var seq = iota[DType.int32, 8](0)     # [0, 1, 2, 3, 4, 5, 6, 7]
```

**Note:** SIMD size must be a power of 2 (1, 2, 4, 8, 16, ...)

### Optimal Size

```mojo
from sys.info import simdwidthof

alias dtype = DType.float32
alias optimal_width = simdwidthof[dtype]()

# This vector fills the SIMD register perfectly
var optimized = SIMD[dtype, optimal_width]()
```

---

## SIMD Operations

All operations apply to **all elements simultaneously**:

```mojo
var a = SIMD[DType.int32, 4](1, 2, 3, 4)
var b = SIMD[DType.int32, 4](10, 20, 30, 40)

# Arithmetic (elementwise)
print(a + b)   # [11, 22, 33, 44]
print(a * b)   # [10, 40, 90, 160]
print(a ** 2)  # [1, 4, 9, 16]

# Scalar operations
print(a * 10)  # [10, 20, 30, 40]

# Math functions
from math import sqrt, cos
var c = SIMD[DType.float64, 4](1.0, 4.0, 9.0, 16.0)
print(sqrt(c)) # [1.0, 2.0, 3.0, 4.0]
```

### Element Access

```mojo
var v = SIMD[DType.int32, 4](10, 20, 30, 40)
print(v[0])    # 10
print(v[2])    # 30
v[1] = 25
print(v)       # [10, 25, 30, 40]
print(len(v))  # 4
```

### Comparison and Reduction

```mojo
var x = SIMD[DType.int32, 4](1, 5, 3, 7)

# Element-wise comparison returns SIMD[Bool, 4]
var mask = x > 4  # [False, True, False, True]

# Reductions
from math import reduce_add, reduce_max, reduce_min
print(reduce_add(x))  # 16 (sum)
print(reduce_max(x))  # 7
print(reduce_min(x))  # 1

# All/any checks
if all(x > 0):
    print("All positive")
if any(x > 5):
    print("At least one > 5")
```

### Manipulation

```mojo
var a = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
var b = SIMD[DType.float32, 4](5.0, 6.0, 7.0, 8.0)

# Join (concatenate)
print(a.join(b))      # [1, 2, 3, 4, 5, 6, 7, 8]

# Interleave
print(a.interleave(b)) # [1, 5, 2, 6, 3, 7, 4, 8]

# Shuffle
var c = SIMD[DType.int8, 4](10, 20, 30, 40)
print(c.shuffle[2, 0, 3, 1]())  # [30, 10, 40, 20]
```

---

## Vectorization with `vectorize`

The `vectorize` function automatically loops with SIMD:

```mojo
from algorithm import vectorize
from sys.info import simdwidthof
from memory import UnsafePointer

alias dtype = DType.float32
alias simd_width = simdwidthof[dtype]()

fn add_scalar_to_array(
    data: UnsafePointer[Scalar[dtype]], 
    size: Int, 
    scalar: Scalar[dtype]
):
    @parameter
    fn add_simd[width: Int](i: Int):
        var chunk = data.load[width=width](i)
        chunk += scalar
        data.store(i, chunk)
    
    vectorize[add_simd, simd_width](size)
```

### How Vectorize Works

```
vectorize[func, 8](100) with SIMD width 8:

Iteration 0:  func[8](0)   → process indices 0-7
Iteration 1:  func[8](8)   → process indices 8-15
Iteration 2:  func[8](16)  → process indices 16-23
...
Iteration 11: func[8](88)  → process indices 88-95
Iteration 12: func[4](96)  → process indices 96-99 (remainder)
```

---

## Parallelization with `parallelize`

Run across multiple CPU cores:

```mojo
from algorithm import parallelize

fn parallel_work():
    var results = UnsafePointer[Int].alloc(100)
    
    @parameter
    fn worker(i: Int):
        results[i] = i * i
    
    parallelize[worker](100)  # Runs on all available cores
    
    # Clean up
    results.free()
```

### Combining SIMD + Parallelization

```mojo
from algorithm import vectorize, parallelize
from sys.info import simdwidthof, num_physical_cores

alias dtype = DType.float32
alias simd_width = simdwidthof[dtype]()
alias num_cores = num_physical_cores()

fn process_large_array(data: UnsafePointer[Scalar[dtype]], size: Int):
    alias chunk_size = size // num_cores
    
    @parameter
    fn parallel_chunk(core_id: Int):
        var start = core_id * chunk_size
        var end = start + chunk_size
        
        @parameter
        fn simd_process[width: Int](i: Int):
            var idx = start + i
            var values = data.load[width=width](idx)
            values = values * 2.0  # Example operation
            data.store(idx, values)
        
        vectorize[simd_process, simd_width](chunk_size)
    
    parallelize[parallel_chunk](num_cores)
```

---

## System Information

```mojo
from sys.info import (
    num_physical_cores,
    num_logical_cores,
    has_avx, has_avx2, has_avx512f,
    has_neon,
    has_nvidia_gpu_accelerator,
    os_is_linux, os_is_macos, os_is_windows,
    _current_arch
)

fn print_system_info():
    print("Physical cores:", num_physical_cores())
    print("Logical cores:", num_logical_cores())
    print("Architecture:", String(_current_arch()))
    
    print("SIMD capabilities:")
    if has_avx(): print("  AVX")
    if has_avx2(): print("  AVX2")
    if has_avx512f(): print("  AVX-512")
    if has_neon(): print("  NEON (ARM)")
    
    if has_nvidia_gpu_accelerator():
        print("NVIDIA GPU available")
```

---

## Performance Best Practices

### 1. Use Optimal SIMD Width

```mojo
# Bad: arbitrary width
var v = SIMD[DType.float32, 7](...)  # Not power of 2!

# Good: use system width
alias width = simdwidthof[DType.float32]()
var v = SIMD[DType.float32, width](...)
```

### 2. Align Data to SIMD Width

```mojo
# Ensure array size is multiple of SIMD width
alias simd_width = simdwidthof[DType.float32]()
var aligned_size = ((original_size + simd_width - 1) // simd_width) * simd_width
```

### 3. Minimize Memory Access

```mojo
# Bad: load/store in inner loop
for i in range(n):
    var x = data.load(i)
    x = process(x)
    data.store(i, x)

# Good: batch operations
@parameter
fn batch[width: Int](i: Int):
    var chunk = data.load[width=width](i)
    chunk = process_simd(chunk)
    data.store(i, chunk)
vectorize[batch, simd_width](n)
```

### 4. Use Compile-Time Constants

```mojo
# Use alias for compile-time values
alias SIZE = 1024
alias DTYPE = DType.float32
alias WIDTH = simdwidthof[DTYPE]()

# Use @parameter for compile-time loops
@parameter
for i in range(WIDTH):
    # Unrolled at compile time
    ...
```

### 5. Avoid Branches in SIMD Code

```mojo
# Bad: branches prevent vectorization
for i in range(n):
    if data[i] > threshold:
        data[i] = threshold

# Good: use SIMD select/mask
@parameter
fn clamp[w: Int](i: Int):
    var v = data.load[width=w](i)
    var mask = v > threshold
    v = mask.select(threshold, v)
    data.store(i, v)
```

---

## Benchmarking

```mojo
from time import perf_counter_ns

fn benchmark[func: fn() -> None]() -> Float64:
    var start = perf_counter_ns()
    func()
    var end = perf_counter_ns()
    return (end - start) / 1e6  # milliseconds

fn main():
    var time_ms = benchmark[my_function]()
    print("Time:", time_ms, "ms")
```

---

## Quick Reference

| Operation | Scalar | SIMD |
|-----------|--------|------|
| Add | `a + b` | `simd_a + simd_b` |
| Multiply | `a * b` | `simd_a * simd_b` |
| Load | `ptr.load(i)` | `ptr.load[width=w](i)` |
| Store | `ptr.store(i, v)` | `ptr.store(i, simd_v)` |
| Reduce sum | loop | `reduce_add(simd)` |
| Apply func | loop | `vectorize[func, w](n)` |

---

## Next Steps

- [05-metaprogramming.md](05-metaprogramming.md) - Compile-time programming
- [06-gpu-programming.md](06-gpu-programming.md) - GPU acceleration
