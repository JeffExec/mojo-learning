# Mojo Metaprogramming

## Parameters vs Arguments

In Mojo, these terms have specific meanings:

| Term | When | How |
|------|------|-----|
| **Parameter** | Compile-time | In square brackets `[...]` |
| **Argument** | Runtime | In parentheses `(...)` |

```mojo
fn process[T: DType, size: Int](data: SIMD[T, size]) -> SIMD[T, size]:
#         ^^^^^^^^^^^^^^^^ parameters (compile-time)
#                                ^^^^^^^^^^^^^^^^^ argument (runtime)
    return data * 2
```

---

## Parameterized Functions

```mojo
# Function with compile-time parameter
fn repeat[count: Int](msg: String):
    @parameter  # Unroll at compile time
    for i in range(count):
        print(msg)

# Usage - parameter in brackets
repeat[3]("Hello")  # Prints "Hello" 3 times
```

The compiler generates specialized code for each unique parameter value:
```mojo
# repeat[3] becomes equivalent to:
fn repeat_3(msg: String):
    print(msg)
    print(msg)
    print(msg)
```

---

## Parameterized Structs

```mojo
struct Array[T: Copyable, size: Int]:
    var data: InlineArray[T, size]
    
    fn __init__(out self, fill: T):
        self.data = InlineArray[T, size](fill)
    
    fn __getitem__(self, i: Int) -> T:
        return self.data[i]
    
    fn __setitem__(mut self, i: Int, value: T):
        self.data[i] = value

# Usage
var arr = Array[Int, 10](0)
arr[5] = 42
print(arr[5])  # 42
```

### Accessing Struct Parameters

```mojo
# On type
print(SIMD[DType.float32, 4].size)   # 4
print(SIMD[DType.float32, 4].dtype)  # float32

# On instance
var v = SIMD[DType.int32, 8](1)
print(v.dtype)  # int32
print(v.size)   # 8
```

---

## The `@parameter` Decorator

### Parametric If

```mojo
from sys.info import has_avx512f

fn optimized_compute():
    @parameter
    if has_avx512f():
        # AVX-512 optimized code
        print("Using AVX-512")
    else:
        # Fallback code
        print("Using scalar")
```

Only the **true branch** is compiled into the binary.

### Parametric For

```mojo
@parameter
for i in range(4):
    print(i)  # Unrolled: 4 print statements in binary

# With compile-time constant
comptime LIMIT = 8

@parameter
for i in range(LIMIT):
    process(i)
```

### Parametric Closure

```mojo
fn use_closure[func: fn(Int) capturing [_] -> Int](num: Int) -> Int:
    return func(num)

fn create_closure():
    var x = 1
    
    @parameter
    fn add(i: Int) -> Int:
        return x + i  # Captures x
    
    var result = use_closure[add](2)  # Pass closure as parameter
    print(result)  # 3
```

---

## `comptime` Values

### Constants

```mojo
comptime PI = 3.14159
comptime BLOCK_SIZE = 256
comptime DEBUG = True
```

### Type Aliases

```mojo
comptime Float16 = SIMD[DType.float16, 1]
comptime Vec4f = SIMD[DType.float32, 4]
comptime IntList = List[Int]
```

### Computed Values

```mojo
fn calculate_size() -> Int:
    return 64 * 64

comptime SIZE = calculate_size()  # Runs at compile time!
```

### Parametric comptime

```mojo
comptime Square[n: Int]: Int = n * n
comptime Clamp[val: Int, lo: Int, hi: Int]: Int = max(lo, min(val, hi))

comptime nine = Square[3]           # 9
comptime clamped = Clamp[50, 0, 10] # 10
```

---

## Type Parameters and Generics

### Generic Functions

```mojo
fn swap[T: Copyable](mut a: T, mut b: T):
    var temp = a.copy()
    a = b
    b = temp

var x = 1
var y = 2
swap(x, y)  # T inferred as Int
print(x, y)  # 2 1
```

### Generic Structs

```mojo
struct Pair[T: Copyable]:
    var first: T
    var second: T
    
    fn __init__(out self, first: T, second: T):
        self.first = first
        self.second = second

var ints = Pair[Int](1, 2)
var strs = Pair[String]("hello", "world")
```

### Multiple Type Parameters

```mojo
struct Map[K: KeyElement, V: Copyable]:
    # K must be hashable and comparable
    # V just needs to be copyable
    ...
```

---

## Parameter Inference

Mojo can infer parameters from arguments:

```mojo
fn double[T: DType](x: Scalar[T]) -> Scalar[T]:
    return x * 2

var result = double(3.14)  # T inferred as float64
# Same as: double[DType.float64](3.14)
```

### Infer-Only Parameters

Parameters before `//` are infer-only:

```mojo
fn dependent[dtype: DType, //, value: Scalar[dtype]]():
    print("Type:", dtype)
    print("Value:", value)

dependent[Float64(3.14)]()  # dtype inferred from value type
# Can also specify: dependent[DType.float64, Float64(3.14)]()
```

---

## Variadic Parameters

```mojo
fn sum_all[*values: Int]() -> Int:
    comptime list = VariadicList(values)
    var total = 0
    for v in list:
        total += v
    return total

var result = sum_all[1, 2, 3, 4, 5]()  # 15
```

---

## Optional Parameters

```mojo
fn greet[name: String = "World", loud: Bool = False]():
    var msg = "Hello, " + name
    if loud:
        msg += "!"
    print(msg)

greet()                    # "Hello, World"
greet["Mojo"]()            # "Hello, Mojo"
greet[loud=True]()         # "Hello, World!"
greet["Mojo", True]()      # "Hello, Mojo!"
```

---

## Dependent Types

Return type can depend on parameters:

```mojo
fn concat[
    dtype: DType, left_size: Int, right_size: Int
](
    left: SIMD[dtype, left_size], 
    right: SIMD[dtype, right_size]
) -> SIMD[dtype, left_size + right_size]:  # Size is sum!
    var result = SIMD[dtype, left_size + right_size]()
    
    @parameter
    for i in range(left_size):
        result[i] = left[i]
    
    @parameter
    for i in range(right_size):
        result[left_size + i] = right[i]
    
    return result

var a = SIMD[DType.int32, 2](1, 2)
var b = SIMD[DType.int32, 3](3, 4, 5)
var c = concat(a, b)  # SIMD[DType.int32, 5]
print(c)  # [1, 2, 3, 4, 5]
```

---

## Compile-Time Recursion

```mojo
fn fibonacci[n: Int]() -> Int:
    @parameter
    if n <= 1:
        return n
    else:
        return fibonacci[n - 1]() + fibonacci[n - 2]()

comptime fib10 = fibonacci[10]()  # 55, computed at compile time!
```

---

## Conditional Conformance

Methods that work only with certain parameter types:

```mojo
struct Container[T: Movable]:
    var value: T
    
    fn __init__(out self, var value: T):
        self.value = value^
    
    # Only available when T is Writable
    fn print_value[WT: Writable & Movable](self: Container[WT]):
        print(self.value)
    
    # Only available when T is Comparable
    fn is_greater[CT: Comparable & Movable](
        self: Container[CT], other: Container[CT]
    ) -> Bool:
        return self.value > other.value
```

---

## SIMD Type Case Study

```mojo
# How SIMD uses parameters
struct SIMD[dtype: DType, size: Int]:
    var value: ...
    
    # Type-safe operations
    fn __add__(self, rhs: Self) -> Self: ...
    
    # Cast between types
    fn cast[target: DType](self) -> SIMD[target, size]: ...
    
    # Static factory
    @staticmethod
    fn splat(x: Scalar[dtype]) -> Self: ...

# Usage
var v = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
var doubled = v + v
var as_int = v.cast[DType.int32]()
var filled = SIMD[DType.float32, 8].splat(42.0)
```

---

## Best Practices

1. **Use `alias`/`comptime`** for compile-time constants
2. **Prefer parameter inference** when types are obvious
3. **Use `@parameter if`** for conditional compilation
4. **Use `@parameter for`** for loop unrolling
5. **Put dependent parameters after `//`** as infer-only
6. **Use traits** to constrain type parameters
7. **Document parameters** with docstrings

---

## Common Patterns

### Type Selection

```mojo
comptime SelectedType = @parameter if use_float else Int
```

### Feature Flags

```mojo
comptime DEBUG = True

fn log(msg: String):
    @parameter
    if DEBUG:
        print("[DEBUG]", msg)
```

### Platform-Specific Code

```mojo
from sys.info import os_is_linux, os_is_macos

fn get_path() -> String:
    @parameter
    if os_is_linux():
        return "/usr/local/bin"
    elif os_is_macos():
        return "/usr/local/bin"
    else:
        return "C:\\Program Files"
```

---

## Next Steps

- [06-gpu-programming.md](06-gpu-programming.md) - GPU programming
- Build projects to practice metaprogramming
