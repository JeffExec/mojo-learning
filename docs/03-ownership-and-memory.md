# Mojo Ownership and Memory Management

## Overview

Mojo uses an **ownership model** (similar to Rust) to ensure memory safety without garbage collection:

- **Every value has exactly one owner**
- **When the owner's lifetime ends, the value is destroyed**
- **References can temporarily borrow values**

---

## Stack vs Heap

### Stack
- Fixed-size local variables
- Automatic management (push/pop with function calls)
- Fast access

### Heap
- Dynamically-sized values
- Programmer-managed (via standard library types)
- Accessible from anywhere in call stack

```mojo
fn example():
    var x: Int = 42         # Stack: fixed size
    var s: String = "hello" # Stack: pointer + metadata
                            # Heap: actual string data
```

---

## Argument Conventions

### `read` (Default): Immutable Reference

```mojo
fn print_list(list: List[Int]):  # read is implicit
    print(list.__str__())
    # list.append(5)  # Error: cannot mutate

def main():
    var values = [1, 2, 3, 4]
    print_list(values)  # No copy, just a reference
```

**Key points:**
- Function cannot modify the value
- No copy is made (efficient)
- Like `const&` in C++ but with lifetime checking

### `mut`: Mutable Reference

```mojo
fn append_value(mut list: List[Int], value: Int):
    list.append(value)  # Modifies original

def main():
    var values = [1, 2, 3]
    append_value(values, 4)
    print(values)  # [1, 2, 3, 4]
```

**Key points:**
- Function can modify the original value
- Changes are visible after function returns
- Original must already be mutable

### `var`: Ownership Transfer

```mojo
fn consume(var text: String):
    text += "!"
    print(text)
    # text is destroyed here (unless transferred)

def main():
    var message = "Hello"
    consume(message^)  # Transfer ownership with ^
    # print(message)   # Error: message is uninitialized
```

**Key points:**
- Function receives exclusive ownership
- Without `^`, a copy is made (if Copyable)
- With `^`, original becomes invalid

### `out`: Uninitialized Output

```mojo
# Used in constructors
fn __init__(out self, value: Int):
    self.data = value  # Must initialize before return

# Used for named results
fn create_pair(out result: MyPair):
    result = MyPair(1, 2)
```

### `ref`: Parametric Mutability

```mojo
fn process[mut: Bool](ref[mut] data: List[Int]) -> ref[mut] Int:
    return data[0]

# Can work with both mutable and immutable references
```

---

## The Transfer Sigil (`^`)

The `^` operator ends a variable's lifetime and transfers ownership:

```mojo
var a = String("Hello")
var b = a^  # 'a' is now invalid, 'b' owns the value

# Without ^, this would copy (if Copyable):
var c = String("World")
var d = c     # 'c' is copied to 'd' (if ImplicitlyCopyable)
var e = c^    # 'c' transferred to 'e', 'c' now invalid
```

---

## Argument Exclusivity

Mojo enforces that mutable references have no aliases:

```mojo
fn append_twice(mut s: String, other: String):
    s += other
    s += other

fn main():
    var my_string = "o"
    # append_twice(my_string, my_string)  # Error!
    # Can't pass same value as both mut and read
    
    # Solution: make a copy
    var other = my_string
    append_twice(my_string, other)  # OK
```

---

## Copy vs Move

### Copy (Deep Copy)

```mojo
struct HeapArray(Copyable):
    var data: UnsafePointer[Int]
    var size: Int
    
    fn __copyinit__(out self, existing: Self):
        # Deep copy: allocate new memory and copy data
        self.size = existing.size
        self.data = UnsafePointer[Int].alloc(self.size)
        for i in range(self.size):
            (self.data + i).init_pointee_copy(existing.data[i])
```

### Move (Ownership Transfer)

```mojo
struct UniqueArray(Movable):
    var data: UnsafePointer[Int]
    var size: Int
    
    fn __moveinit__(out self, deinit existing: Self):
        # Just transfer the pointer, no copy
        self.data = existing.data
        self.size = existing.size
        # existing's destructor is NOT called
```

---

## Value Semantics vs Reference Semantics

### Value Semantics (Mojo Default)

```mojo
var a = [1, 2, 3]
var b = a.copy()  # b is independent copy
b.append(4)
print(a)  # [1, 2, 3] - unchanged
print(b)  # [1, 2, 3, 4]
```

### Reference Semantics (Explicit)

```mojo
fn modify_in_place(mut list: List[Int]):
    list.append(99)  # Modifies original

var data = [1, 2, 3]
modify_in_place(data)
print(data)  # [1, 2, 3, 99]
```

---

## Lifetime Rules

### Rule 1: One Owner

```mojo
var a = String("Hello")
var b = a^  # a's lifetime ends, b is new owner
# a is invalid here
```

### Rule 2: References Extend Lifetime

```mojo
fn process(s: String):  # s borrows the value
    print(s)

var msg = String("Hello")
process(msg)  # msg stays valid (just borrowed)
print(msg)    # Still works
```

### Rule 3: Mutable Reference = Exclusive Access

```mojo
fn modify(mut s: String):
    s += "!"

var text = String("Hi")
modify(text)  # text is exclusively borrowed
# No other access to text during modify()
```

---

## UnsafePointer for Manual Memory

For low-level control:

```mojo
from memory import UnsafePointer

fn manual_memory():
    # Allocate
    var ptr = UnsafePointer[Int].alloc(10)
    
    # Initialize
    for i in range(10):
        (ptr + i).init_pointee_copy(i * 2)
    
    # Use
    print(ptr[5])  # 10
    
    # Clean up
    for i in range(10):
        (ptr + i).destroy_pointee()
    ptr.free()
```

---

## Common Patterns

### Factory Functions with Named Results

```mojo
# For types that can't be moved/copied
fn create_resource(out result: ImmovableResource):
    result = ImmovableResource(setup_config())
    result.configure()  # Can modify before return

var resource = create_resource()
```

### Optional Ownership

```mojo
fn maybe_consume(var text: String, should_consume: Bool):
    if should_consume:
        print("Consuming:", text)
        # text destroyed here
    else:
        print("Keeping:", text)
        # What to do with text?
```

### RAII Pattern

```mojo
struct FileHandle:
    var fd: Int
    
    fn __init__(out self, path: String) raises:
        self.fd = open_file(path)
    
    fn __del__(deinit self):
        close_file(self.fd)  # Auto-cleanup

fn use_file():
    var f = FileHandle("data.txt")
    # use f...
    # f automatically closed when scope ends
```

---

## Memory Safety Guarantees

Mojo prevents:

| Error | Prevention |
|-------|------------|
| Use-after-free | Lifetime tracking |
| Double-free | Single ownership |
| Memory leaks | Automatic destruction |
| Data races | Exclusivity rules |
| Null pointer | UnsafePointer is explicit |

---

## Best Practices

1. **Prefer `read`** (default) for function arguments
2. **Use `mut`** only when modification is needed
3. **Use `var`** for consuming functions
4. **Use `^`** to explicitly transfer ownership
5. **Implement `Copyable`** for value types
6. **Implement `Movable`** for resource types
7. **Avoid `UnsafePointer`** unless necessary

---

## Next Steps

- [04-simd-and-performance.md](04-simd-and-performance.md) - SIMD and vectorization
- [05-metaprogramming.md](05-metaprogramming.md) - Compile-time programming
