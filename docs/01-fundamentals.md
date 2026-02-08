# Mojo Language Fundamentals

> **Mojo** is a systems programming language designed for high-performance AI infrastructure and heterogeneous hardware. It combines Pythonic syntax with systems-level performance.

## Core Philosophy

- **Python syntax & interop**: Adopts Python's syntax and integrates with existing Python code
- **Struct-based types**: All types (including `Int`, `String`) are defined as structs
- **Zero-cost traits**: Compile-time type checking with no runtime cost
- **Value semantics**: Defaults to value semantics for predictable behavior
- **Value ownership**: Rust-like ownership system without garbage collection
- **MLIR foundation**: Built on modern compiler infrastructure for heterogeneous hardware

---

## Basic Syntax

### Hello World

```mojo
def main():
    print("Hello, world!")
```

Every Mojo program requires a `main()` function as the entry point.

### Variables

```mojo
# Implicit declaration (type inferred from value)
x = 10
y = x * x

# Explicit declaration with var
var x: Int = 10
var sum: Int  # Can declare without initializing

# Variables are statically typed - type is set at compile time
x = 10
x = "Foo"  # Error: Cannot convert "StringLiteral" to "Int"
```

### Comments

```mojo
# Single-line comment

var message = "Hello"  # Inline comment

fn print(x: String):
    """Docstring for API documentation.
    
    Args:
        x: The string to print.
    """
    ...
```

---

## Data Types (DType)

All numerical types in Mojo fall under the `DType` umbrella and are SIMD-backed:

| Type | Description |
|------|-------------|
| `Bool` | Boolean |
| `Int8`, `Int16`, `Int32`, `Int64`, `Int128`, `Int256` | Signed integers |
| `UInt8`, `UInt16`, `UInt32`, `UInt64`, `UInt128`, `UInt256` | Unsigned integers |
| `Float16`, `Float32`, `Float64` | Floating point |
| `BFloat16` | Brain floating point |

```mojo
# DType.float64 means same as Float64
var x: Float64 = 3.14
var y: SIMD[DType.float64, 1] = 3.14  # Equivalent

# Scalar is SIMD vector of size 1
alias Scalar = SIMD[size=1]
var s: Scalar[DType.int32] = 42
```

---

## Functions: `def` vs `fn`

### `def` Functions (Dynamic)

```mojo
def greet(name: String) -> String:
    return "Hello, " + name + "!"

# def functions are always treated as raising
# Arguments can be modified (implicit copy)
```

### `fn` Functions (Strict)

```mojo
fn greet(name: String) -> String:
    return "Hello, " + name + "!"

# fn functions cannot raise unless explicitly declared
fn risky_operation() raises:
    raise Error("Something went wrong")
```

### Key Differences

| Feature | `def` | `fn` |
|---------|-------|------|
| Error handling | Always treated as raising | Must declare `raises` explicitly |
| Argument mutability | Can modify copies | Arguments are `read` by default |
| Use case | Prototyping, Python interop | Performance-critical code, APIs |

---

## Argument Conventions

```mojo
# read (default): Immutable reference
fn print_value(value: Int):  # read is implicit
    print(value)
    # value += 1  # Error: cannot modify

# mut: Mutable reference
fn increment(mut value: Int):
    value += 1

# var: Transfer ownership
fn consume(var text: String):
    text += "!"
    print(text)

# ref: Parametric mutability (advanced)
fn process[is_mutable: Bool](ref[is_mutable] data: List[Int]):
    ...

# out: For constructors and named results
fn __init__(out self, name: String):
    self.name = name
```

### Transfer Sigil (^)

```mojo
var message = "Hello"
consume(message^)  # Transfer ownership, message is now invalid
# print(message)   # Error: use of uninitialized value
```

---

## Control Flow

### If/Else

```mojo
fn check_value(x: Int):
    if x > 0:
        print("positive")
    elif x < 0:
        print("negative")
    else:
        print("zero")
```

### For Loops

```mojo
# Range-based
for i in range(10):
    print(i)

# Iterating collections
var items = List[Int](1, 2, 3)
for item in items:
    print(item[])  # Note: item is a reference
```

### While Loops

```mojo
var count = 0
while count < 5:
    print(count)
    count += 1
```

---

## Compile-Time Values (`comptime`)

```mojo
# Constants
comptime rows = 512
comptime block_size = _calculate_block_size()

# Type aliases
comptime Float16 = SIMD[DType.float16, 1]
comptime UInt8 = SIMD[DType.uint8, 1]

# Parametric comptime values
comptime AddOne[a: Int]: Int = a + 1
comptime nine = AddOne[8]  # 9
```

---

## Error Handling

```mojo
fn safe_divide(a: Int, b: Int) raises -> Int:
    if b == 0:
        raise Error("Division by zero")
    return a // b

fn main():
    try:
        var result = safe_divide(10, 0)
        print(result)
    except e:
        print("Error:", e)
```

---

## Python Integration

```mojo
from python import Python

def main():
    var np = Python.import_module("numpy")
    var ar = np.arange(15).reshape(3, 5)
    print(ar)
    print(ar.shape)
```

---

## Quick Reference: Operators

| Category | Operators |
|----------|-----------|
| Arithmetic | `+`, `-`, `*`, `/`, `//`, `%`, `**` |
| Comparison | `==`, `!=`, `<`, `>`, `<=`, `>=` |
| Logical | `and`, `or`, `not` |
| Bitwise | `&`, `\|`, `^`, `~`, `<<`, `>>` |
| Assignment | `=`, `+=`, `-=`, `*=`, `/=`, etc. |

---

## Next Steps

- [02-structs-and-traits.md](02-structs-and-traits.md) - Structs, traits, and OOP
- [03-ownership-and-memory.md](03-ownership-and-memory.md) - Ownership model
- [04-simd-and-performance.md](04-simd-and-performance.md) - SIMD and vectorization
- [05-metaprogramming.md](05-metaprogramming.md) - Parameters and compile-time programming
