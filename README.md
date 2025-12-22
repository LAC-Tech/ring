Pure rust, io_uring bindings that do not link libc. Inspired by the bindings in the Zig standard library, and also the C library liburing.

## Why make more bindings when others exist?

- tokio-rs/io-uring has an overly cutesy, opionated and verbose interface. It also depends on libc
- rustix-uring rightfully just depends on rustix, but copies the tokio-rs/io-uring interface
- axboe-liburing depends on libc, and also compiles C code
