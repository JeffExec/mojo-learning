# Mojo GPU Programming

## Overview

Mojo provides unified GPU programming through the `gpu` package:
- Hardware-agnostic: Works on NVIDIA, AMD, and Apple GPUs
- Same syntax as CPU code
- Built on MLIR for optimal code generation

---

## Check GPU Availability

```mojo
from sys import has_accelerator, has_nvidia_gpu_accelerator, 
               has_amd_gpu_accelerator, has_apple_gpu_accelerator

fn main():
    @parameter
    if has_accelerator():
        print("GPU available!")
        
        @parameter
        if has_nvidia_gpu_accelerator():
            print("  NVIDIA GPU")
        elif has_amd_gpu_accelerator():
            print("  AMD GPU")
        elif has_apple_gpu_accelerator():
            print("  Apple GPU")
    else:
        print("No GPU - falling back to CPU")
```

---

## GPU Programming Model

1. **Initialize** data in host (CPU) memory
2. **Allocate** device (GPU) memory
3. **Copy** data from host to device
4. **Execute** kernel function on GPU
5. **Copy** results back to host
6. **Synchronize** to ensure completion

---

## DeviceContext

The main interface for GPU operations:

```mojo
from gpu.host import DeviceContext

fn main() raises:
    # Default GPU (device 0)
    var ctx = DeviceContext()
    
    # Specific GPU by index
    var ctx1 = DeviceContext(device_id=1)
    
    # Specific vendor API
    var nvidia_ctx = DeviceContext(api="cuda")
    var amd_ctx = DeviceContext(api="hip")
    var apple_ctx = DeviceContext(api="metal")
    
    # Number of available GPUs
    print("GPUs:", DeviceContext.number_of_devices())
```

---

## Memory Management

### Device Buffer (GPU Memory)

```mojo
from gpu.host import DeviceContext

fn main():
    var ctx = DeviceContext()
    
    # Create buffer on GPU
    var device_buf = ctx.enqueue_create_buffer[DType.float32](1024)
    
    # Buffer is automatically freed when out of scope
```

### Host Buffer (CPU Memory)

```mojo
var ctx = DeviceContext()

# Create buffer on CPU (pinned for fast transfers)
var host_buf = ctx.enqueue_create_host_buffer[DType.float32](1024)
ctx.synchronize()  # Wait for creation

# Initialize data
for i in range(1024):
    host_buf[i] = Float32(i)
```

### Copying Data

```mojo
var ctx = DeviceContext()
var host_buf = ctx.enqueue_create_host_buffer[DType.float32](1024)
var device_buf = ctx.enqueue_create_buffer[DType.float32](1024)
ctx.synchronize()

# Initialize host data...

# Host → Device
ctx.enqueue_copy(src_buf=host_buf, dst_buf=device_buf)

# ... run kernel ...

# Device → Host
ctx.enqueue_copy(src_buf=device_buf, dst_buf=host_buf)
ctx.synchronize()

# Alternative syntax
host_buf.enqueue_copy_to(dst=device_buf)
device_buf.enqueue_copy_to(dst=host_buf)
```

### Quick Testing with `map_to_host`

```mojo
# For prototyping (not production - involves copies)
with device_buf.map_to_host() as host_view:
    for i in range(len(host_view)):
        host_view[i] = Float32(i)
# Changes automatically copied back when exiting `with`
```

---

## Kernel Functions

Kernels are functions that run on the GPU:

```mojo
from gpu import block_dim, block_idx, thread_idx, global_idx

# Kernel function must be non-raising (fn, not def)
fn add_scalar(
    data: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
    scalar: Float32,
):
    # Calculate global thread index
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    # Bounds check (essential!)
    if idx < UInt(size):
        data[idx] += scalar
```

### Thread Organization

| Value | Description |
|-------|-------------|
| `grid_dim` | Dimensions of the grid (number of blocks) |
| `block_dim` | Dimensions of each block (threads per block) |
| `block_idx` | Index of current block in grid |
| `thread_idx` | Index of current thread in block |
| `global_idx` | Global thread index (computed) |

```
global_idx.x = block_dim.x * block_idx.x + thread_idx.x
global_idx.y = block_dim.y * block_idx.y + thread_idx.y
global_idx.z = block_dim.z * block_idx.z + thread_idx.z
```

### Grid and Block Dimensions

```mojo
# 1D: 1024 threads total
ctx.enqueue_function(
    kernel,
    grid_dim=4,      # 4 blocks
    block_dim=256,   # 256 threads per block
)

# 2D: 32x32 = 1024 threads per block, 16x16 blocks
ctx.enqueue_function(
    kernel,
    grid_dim=(16, 16),
    block_dim=(32, 32),
)

# 3D
ctx.enqueue_function(
    kernel,
    grid_dim=(4, 4, 4),
    block_dim=(8, 8, 8),
)
```

---

## Complete Example

```mojo
from sys import has_accelerator, exit
from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx
from math import iota

comptime NUM_ELEMENTS = 1024
comptime BLOCK_SIZE = 256

fn vector_add(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    result: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx < UInt(size):
        result[idx] = a[idx] + b[idx]

def main():
    @parameter
    if not has_accelerator():
        print("No GPU available")
        return
    
    var ctx = DeviceContext()
    
    # Create buffers
    var host_a = ctx.enqueue_create_host_buffer[DType.float32](NUM_ELEMENTS)
    var host_b = ctx.enqueue_create_host_buffer[DType.float32](NUM_ELEMENTS)
    var host_result = ctx.enqueue_create_host_buffer[DType.float32](NUM_ELEMENTS)
    
    var dev_a = ctx.enqueue_create_buffer[DType.float32](NUM_ELEMENTS)
    var dev_b = ctx.enqueue_create_buffer[DType.float32](NUM_ELEMENTS)
    var dev_result = ctx.enqueue_create_buffer[DType.float32](NUM_ELEMENTS)
    
    ctx.synchronize()
    
    # Initialize data
    for i in range(NUM_ELEMENTS):
        host_a[i] = Float32(i)
        host_b[i] = Float32(i * 2)
    
    # Copy to device
    ctx.enqueue_copy(src_buf=host_a, dst_buf=dev_a)
    ctx.enqueue_copy(src_buf=host_b, dst_buf=dev_b)
    
    # Compile and run kernel
    var num_blocks = (NUM_ELEMENTS + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    ctx.enqueue_function[vector_add, vector_add](
        dev_a,
        dev_b,
        dev_result,
        NUM_ELEMENTS,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )
    
    # Copy results back
    ctx.enqueue_copy(src_buf=dev_result, dst_buf=host_result)
    ctx.synchronize()
    
    # Verify
    print("First 5 results:")
    for i in range(5):
        print(f"  {host_a[i]} + {host_b[i]} = {host_result[i]}")
```

---

## Compiling Kernels

### One-Shot (Compile + Run)

```mojo
# Kernel is compiled and executed together
ctx.enqueue_function[my_kernel, my_kernel](
    arg1, arg2,
    grid_dim=blocks,
    block_dim=threads,
)
```

### Pre-Compiled (Reusable)

```mojo
# Compile once
var compiled = ctx.compile_function[my_kernel, my_kernel]()

# Run multiple times
for i in range(100):
    ctx.enqueue_function(
        compiled,
        arg1, arg2,
        grid_dim=blocks,
        block_dim=threads,
    )
```

---

## DevicePassable Types

Only certain types can be passed to kernels:

| Host Type | Device Type | Description |
|-----------|-------------|-------------|
| `Int` | `Int` | Integer |
| `SIMD[dtype, width]` | `SIMD[dtype, width]` | SIMD vector |
| `DeviceBuffer[dtype]` | `UnsafePointer[...]` | Memory buffer |
| `LayoutTensor` | `LayoutTensor` | Multi-dim array |

---

## Synchronization

GPU operations are asynchronous:

```mojo
# Enqueue operations (returns immediately)
ctx.enqueue_copy(src_buf=host, dst_buf=device)
ctx.enqueue_function[kernel, kernel](...)
ctx.enqueue_copy(src_buf=device, dst_buf=host)

# Block until all operations complete
ctx.synchronize()

# Now safe to access host buffer
print(host[0])
```

---

## Debugging Kernels

```mojo
fn debug_kernel(data: UnsafePointer[Float32, MutAnyOrigin]):
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    # Print from kernel (not on Apple GPUs)
    print(
        "block:", block_idx.x,
        "thread:", thread_idx.x,
        "global:", idx,
        "value:", data[idx]
    )
```

**Note:** Printing from kernels is not supported on Apple GPUs.

---

## Best Practices

### 1. Always Check Bounds

```mojo
fn kernel(data: UnsafePointer[Float32, MutAnyOrigin], size: Int):
    var idx = global_idx.x
    if idx < UInt(size):  # Essential!
        data[idx] = ...
```

### 2. Choose Block Size Wisely

```mojo
# Common choices: 128, 256, 512
# Should be multiple of warp size (32 for NVIDIA, 64 for AMD)
comptime BLOCK_SIZE = 256
```

### 3. Minimize Host-Device Transfers

```mojo
# Bad: Transfer for each operation
for i in range(100):
    ctx.enqueue_copy(host, device)
    ctx.enqueue_function(kernel)
    ctx.enqueue_copy(device, host)

# Good: Batch operations on device
ctx.enqueue_copy(host, device)
for i in range(100):
    ctx.enqueue_function(kernel)
ctx.enqueue_copy(device, host)
```

### 4. Coalesce Memory Access

```mojo
# Good: Adjacent threads access adjacent memory
fn good_kernel(data: UnsafePointer[Float32, MutAnyOrigin]):
    var idx = global_idx.x
    data[idx] = ...  # Coalesced access

# Bad: Strided access
fn bad_kernel(data: UnsafePointer[Float32, MutAnyOrigin], stride: Int):
    var idx = global_idx.x * stride  # Non-coalesced!
    data[idx] = ...
```

### 5. Use Shared Memory for Reused Data

```mojo
# Shared memory within block (future Mojo feature)
# For now, structure algorithms to maximize cache usage
```

---

## Error Handling

```mojo
fn main():
    try:
        var ctx = DeviceContext()
        # ... GPU operations ...
    except e:
        print("GPU error:", e)
        # Fallback to CPU implementation
```

---

## GPU Requirements

### NVIDIA (CUDA)
- Driver version 580+
- Compute capability 7.0+ (Volta, Turing, Ampere, Ada, Hopper, Blackwell)

### AMD (HIP/ROCm)
- Driver version 6.3.3+
- CDNA3/CDNA4 (MI300X, MI325X, MI355X)
- RDNA3/RDNA4 (limited)

### Apple (Metal)
- M1/M2/M3/M4/M5 (limited support)

---

## Next Steps

- Explore the `gpu` module documentation
- Study the MAX kernels library for advanced patterns
- Build GPU-accelerated projects
