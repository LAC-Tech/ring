[docs](https://docs.rs/hringas/latest/hringas/)

[example](/examples/readme.rs)

⚠️ **WIP, not all tests have been implemented yet** ⚠️

| IoUring.zig test                               | Passes |
| ---------------------------------------------- | ------ |
| structs/offsets/entries                        | ✅     |
| nop                                            | ✅     |
| readv                                          | ❌     |
| writev/fsync/readv                             | ✅     |
| write/read                                     | ❌     |
| splice/read                                    | ❌     |
| write_fixed/read_fixed                         | ❌     |
| openat                                         | ❌     |
| close                                          | ❌     |
| accept/connect/send/recv                       | ❌     |
| sendmsg/recvmsg                                | ❌     |
| timeout (after a relative time)                | ❌     |
| timeout (after a number of completions)        | ❌     |
| timeout_remove                                 | ❌     |
| accept/connect/recv/link_timeout               | ❌     |
| fallocate                                      | ❌     |
| statx                                          | ❌     |
| accept/connect/recv/cancel                     | ❌     |
| register_files_update                          | ❌     |
| shutdown                                       | ❌     |
| renameat                                       | ❌     |
| unlinkat                                       | ❌     |
| mkdirat                                        | ❌     |
| symlinkat                                      | ❌     |
| linkat                                         | ❌     |
| provide_buffers: read                          | ❌     |
| remove_buffers                                 | ❌     |
| provide_buffers: accept/connect/send/recv      | ❌     |
| accept multishot                               | ❌     |
| accept/connect/send_zc/recv                    | ❌     |
| accept_direct                                  | ❌     |
| accept_multishot_direct                        | ❌     |
| socket                                         | ❌     |
| socket_direct/socket_direct_alloc/close_direct | ❌     |
| openat_direct/close_direct                     | ❌     |
| waitid                                         | ❌     |
| BufferGroup                                    | ❌     |
| ring mapped buffers recv                       | ❌     |
| ring mapped buffers multishot recv             | ❌     |
| copy_cqes with wrapping sq.cqes buffer         | ❌     |
| bind/listen/connect                            | ❌     |

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
