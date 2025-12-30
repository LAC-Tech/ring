⚠️ **WIP, not all tests have been implemented yet** ⚠️

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

## Example

```rust
#![no_std]
use ring::rustix::fs::{openat, Mode, OFlags, CWD};
use ring::rustix::io_uring::io_uring_cqe;
use ring::{prep, IoUring};

fn main() {
    let mut ring = IoUring::new(8).unwrap();

    let fd = openat(CWD, "README.md", OFlags::RDONLY, Mode::empty()).unwrap();
    let mut buf = [0; 1024];

    let sqe = ring.get_sqe().unwrap();
    prep::read(sqe, 0x42, fd, &mut buf, 0);

    // Note that the developer needs to ensure
    // that the entry pushed into submission queue is valid (e.g. fd, buffer).
    let io_uring_cqe { user_data, res, .. } = unsafe {
        ring.submit_and_wait(1).expect("completion queue is empty");
        ring.copy_cqe().unwrap()
    };

    assert_eq!(user_data.u64_(), 0x42);
    assert!(res >= 0, "read error: {}", res);
}
```
