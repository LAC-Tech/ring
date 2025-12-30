⚠️ **WIP, not all tests have been implemented yet** ⚠️

[docs](https://docs.rs/hringas/latest/hringas/)
[examples](/examples/readme.rs)

Rust io_uring bindings.

- pure rust
- `no_std`
- does not link libc
- simple API

It's heavily based off of std.os.linux.IoUring.zig, but also takes some pointers (lol) from liburing.

## Contribute

I welcome pull requests, code reviews, etc. I am especially looking for help in establishing where the "safe" and "unsafe" boundaries should be.

## Why make more bindings when others exist?

- tokio-rs/io-uring has an overly cutesy, opionated and verbose interface. It also depends on libc
- rustix-uring rightfully just depends on rustix, but copies the tokio-rs/io-uring interface
- axboe-liburing depends on libc, and also compiles C code
