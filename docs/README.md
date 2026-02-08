# Mojo Learning Resources

Comprehensive documentation for learning the Mojo programming language.

## Contents

1. **[01-fundamentals.md](01-fundamentals.md)** - Core syntax, types, functions
2. **[02-structs-and-traits.md](02-structs-and-traits.md)** - OOP in Mojo
3. **[03-ownership-and-memory.md](03-ownership-and-memory.md)** - Memory safety
4. **[04-simd-and-performance.md](04-simd-and-performance.md)** - SIMD vectorization
5. **[05-metaprogramming.md](05-metaprogramming.md)** - Compile-time programming
6. **[06-gpu-programming.md](06-gpu-programming.md)** - GPU acceleration

## Quick Start

```bash
# Install Mojo via pixi
curl -fsSL https://pixi.sh/install.sh | bash
pixi init my-mojo-project -c https://conda.modular.com/max-nightly/ -c conda-forge
cd my-mojo-project
pixi add mojo
pixi shell

# Run a Mojo file
mojo hello.mojo
```

## Key Mojo Features

| Feature | Description |
|---------|-------------|
| **Python Syntax** | Familiar syntax for Python developers |
| **Static Typing** | Compile-time type checking |
| **SIMD-First** | All numerics are SIMD-backed |
| **Ownership** | Rust-like memory safety without GC |
| **Zero-Cost Traits** | Compile-time polymorphism |
| **Metaprogramming** | Powerful compile-time code generation |
| **GPU Support** | Unified programming for CPUs and GPUs |
| **Python Interop** | Import and use Python libraries |

## Learning Path

1. Start with [fundamentals](01-fundamentals.md)
2. Understand [structs and traits](02-structs-and-traits.md)
3. Master [ownership](03-ownership-and-memory.md) for memory safety
4. Learn [SIMD optimization](04-simd-and-performance.md)
5. Explore [metaprogramming](05-metaprogramming.md)
6. (Optional) Dive into [GPU programming](06-gpu-programming.md)

## Resources

- [Official Mojo Manual](https://docs.modular.com/mojo/manual/)
- [Mojo API Reference](https://docs.modular.com/mojo/lib)
- [GitHub Examples](https://github.com/modular/modular/tree/main/mojo/examples)
- [Modular Discord](https://discord.gg/modular)

## Projects

See [/projects/mojo-projects/](../../projects/mojo-projects/) for practical examples.
