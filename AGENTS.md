# Agent Guide - ring

## Build & Test Commands
- Build: `cargo build`
- Test: `cargo test`
- Single Test: `cargo test -- <test_name>`
- Lint: `cargo clippy`
- Format: `cargo fmt`

## Code Style Guidelines
- **Formatting**: Max width 80 chars. Follow `rustfmt.toml`.
- **No Std**: Use core only
- **No Dependencies**: Only rustix; do not add anymore.
- **Imports**: Group imports: `core`, then external (`rustix`), then `std`.
- **Naming**: `PascalCase` for types, `snake_case` for functions/variables.
- **Errors**: Custom `enum` for initialization; `rustix::io::Result` for I/O.
- **Safety**: Document `unsafe` blocks. Minimize use outside of FFI/mmap.
- **Architecture**: Inspired by Zig's `IoUring`. Prefer low-level control.
- **Efficiency**: Use single `mmap` for SQ/CQ rings (Kernel 5.4+).
