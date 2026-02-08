# Mojo Structs and Traits

## Structs Overview

Structs are Mojo's primary way to define custom types. They are:
- **Static**: Bound at compile-time (no dynamic dispatch)
- **Memory-safe**: All fields must be initialized
- **Performant**: Optimized code generation

```mojo
struct MyPair:
    var first: Int
    var second: Int
    
    fn __init__(out self, first: Int, second: Int):
        self.first = first
        self.second = second
    
    fn get_sum(self) -> Int:
        return self.first + self.second
```

---

## The `@fieldwise_init` Decorator

Generates a constructor that initializes all fields:

```mojo
@fieldwise_init
struct MyPair:
    var first: Int
    var second: Int
    # __init__ is auto-generated!

var pair = MyPair(2, 4)
print(pair.first)  # 2
```

With implicit conversion:
```mojo
@fieldwise_init("implicit")
struct Counter:
    var count: Int

var c: Counter = 5  # Implicitly converts Int to Counter
```

---

## Field Rules

```mojo
struct MyStruct:
    # Fields MUST use var keyword
    var count: Int          # ✓ Correct
    # value: Int            # ✗ Error: missing var
    
    # Fields MUST be initialized in constructor
    # var foo: Int = 10     # ✗ Error: no default values
    
    # comptime members ARE allowed for constants
    comptime bar = 10       # ✓ Correct
```

---

## Methods

### Instance Methods

```mojo
struct Counter:
    var value: Int
    
    fn __init__(out self):
        self.value = 0
    
    # Immutable self (read)
    fn get_value(self) -> Int:
        return self.value
    
    # Mutable self (mut)
    fn increment(mut self):
        self.value += 1
```

### Static Methods

```mojo
struct Logger:
    @staticmethod
    fn log_info(message: String):
        print("Info:", message)

Logger.log_info("Hello")  # Call on type
```

---

## Special Methods (Dunder Methods)

### Constructor and Destructor

```mojo
struct Resource:
    var data: UnsafePointer[Int]
    
    fn __init__(out self, size: Int):
        self.data = UnsafePointer[Int].alloc(size)
    
    fn __del__(deinit self):
        self.data.free()
```

### Copy and Move Constructors

```mojo
struct MyValue(Copyable, Movable):
    var data: Int
    
    fn __init__(out self, value: Int):
        self.data = value
    
    # Copy constructor (optional with Copyable trait)
    fn __copyinit__(out self, existing: Self):
        self.data = existing.data
    
    # Move constructor (optional with Movable trait)
    fn __moveinit__(out self, deinit existing: Self):
        self.data = existing.data
```

### Common Dunder Methods

| Method | Purpose |
|--------|---------|
| `__init__` | Constructor |
| `__del__` | Destructor |
| `__copyinit__` | Copy constructor |
| `__moveinit__` | Move constructor |
| `__str__` | String conversion |
| `__repr__` | Debug representation |
| `__eq__`, `__ne__` | Equality |
| `__lt__`, `__le__`, `__gt__`, `__ge__` | Comparison |
| `__add__`, `__sub__`, etc. | Arithmetic |
| `__getitem__`, `__setitem__` | Indexing |
| `__len__` | Length |
| `__iter__` | Iteration |

---

## Traits

Traits define a contract that types must implement:

```mojo
trait Quackable:
    fn quack(self):
        ...  # Not implemented (required)

@fieldwise_init
struct Duck(Quackable):
    fn quack(self):
        print("Quack!")

@fieldwise_init  
struct StealthCow(Quackable):
    fn quack(self):
        print("Moo!")
```

### Using Traits as Type Bounds

```mojo
# Long form with explicit parameter
fn make_it_quack[T: Quackable](maybe_a_duck: T):
    maybe_a_duck.quack()

# Short form with Some
fn make_it_quack2(maybe_a_duck: Some[Quackable]):
    maybe_a_duck.quack()

make_it_quack(Duck())       # Quack!
make_it_quack(StealthCow()) # Moo!
```

### Default Method Implementations

```mojo
trait Greetable:
    fn greet(self):
        print("Hello!")  # Default implementation
    
    fn custom_greet(self, name: String):
        ...  # Must be implemented

struct Person(Greetable):
    fn custom_greet(self, name: String):
        print("Hi,", name)
    # greet() uses default implementation
```

### Trait Inheritance

```mojo
trait Animal:
    fn make_sound(self): ...

trait Bird(Animal):  # Inherits from Animal
    fn fly(self): ...

# Must implement both make_sound() AND fly()
struct Parrot(Bird):
    fn make_sound(self):
        print("Squawk!")
    fn fly(self):
        print("Flap flap!")
```

### Trait Composition

```mojo
# Using & to combine traits
fn quack_and_fly[T: Quackable & Flyable](creature: T):
    creature.quack()
    creature.fly()

# Create type alias for composition
comptime DuckLike = Quackable & Flyable
```

---

## Built-in Traits

### Lifecycle Traits

| Trait | Purpose | Methods |
|-------|---------|---------|
| `Copyable` | Explicit copy support | `__copyinit__`, `copy()` |
| `ImplicitlyCopyable` | Implicit copy support | `__copyinit__` |
| `Movable` | Move support | `__moveinit__` |
| `Defaultable` | Default constructor | `__init__(out self)` |

### Value Traits

| Trait | Purpose | Methods |
|-------|---------|---------|
| `Stringable` | String conversion | `__str__()` |
| `Representable` | Debug repr | `__repr__()` |
| `Writable` | Stream writing | `write_to()` |
| `Sized` | Has length | `__len__()` |
| `Boolable` | Bool conversion | `__bool__()` |
| `Intable` | Int conversion | `__int__()` |
| `Hashable` | Hash support | `__hash__()` |

### Comparison Traits

| Trait | Purpose | Methods |
|-------|---------|---------|
| `Equatable` | Equality | `__eq__`, `__ne__` |
| `Comparable` | Ordering | `__lt__`, `__le__`, `__gt__`, `__ge__` |

---

## Making Structs Copyable and Movable

```mojo
# Basic copyable (generates __copyinit__)
@fieldwise_init
struct Point(Copyable):
    var x: Int
    var y: Int

var p1 = Point(1, 2)
var p2 = p1.copy()  # Explicit copy
# var p3 = p1       # Error: not implicitly copyable

# Implicitly copyable (allows var p3 = p1)
@fieldwise_init
struct SmallValue(ImplicitlyCopyable):
    var value: Int

var v1 = SmallValue(42)
var v2 = v1  # Implicit copy works

# Move only (cannot copy)
@fieldwise_init
struct UniqueResource(Movable):
    var data: Int

var r1 = UniqueResource(100)
var r2 = r1^  # Transfer ownership
# print(r1.data)  # Error: r1 is invalid
```

---

## The `@register_passable` Decorator

For types that can be passed in CPU registers:

```mojo
@register_passable
struct Coord:
    var x: UInt32
    var y: UInt32
    var z: UInt32

# "trivial" for simple types without custom lifecycle
@register_passable("trivial")
struct TrivialCoord:
    var x: UInt32
    var y: UInt32
```

---

## Conditional Conformance

Define methods that work only with specific type parameters:

```mojo
struct Container[T: Movable]:
    var element: T
    
    fn __init__(out self, var value: T):
        self.element = value^
    
    # Only available if T is also Writable
    def __str__[StrT: Writable & Movable](
        self: Container[StrT]
    ) -> String:
        return String(self.element)
```

---

## Structs vs Python Classes

| Feature | Mojo Structs | Python Classes |
|---------|--------------|----------------|
| Binding | Compile-time | Runtime |
| Dynamic dispatch | No | Yes |
| Monkey-patching | No | Yes |
| Inheritance | No (use traits) | Yes |
| Static data members | No (use comptime) | Yes |
| Field declaration | Required with `var` | Optional |
| Performance | High | Lower |

---

## Best Practices

1. **Use `@fieldwise_init`** for simple structs
2. **Add `Copyable`** only if copying makes sense
3. **Prefer traits** over inheritance for polymorphism
4. **Use `comptime`** for constants, not fields
5. **Document with docstrings** for API clarity
6. **Use `fn`** for performance-critical methods
7. **Keep structs focused** - single responsibility

---

## Next Steps

- [03-ownership-and-memory.md](03-ownership-and-memory.md) - Deep dive into ownership
- [04-simd-and-performance.md](04-simd-and-performance.md) - Performance optimization
