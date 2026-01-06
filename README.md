# ᚻᚱᛁᚾᚷᚪᛋ

Low-level IoUring bindings in pure rust.

- `no_std`
- does not link libc
- simple API

It's heavily based off of std.os.linux.IoUring.zig, but also takes some pointers (lol) from liburing.

[docs](https://docs.rs/hringas/latest/hringas/)

[example](/examples/readme.rs)

⚠️ **WIP, not all tests have been implemented yet** ⚠️

| IoUring.zig test                               | Passes |
| ---------------------------------------------- | ------ |
| structs/offsets/entries                        | ✅     |
| nop                                            | ✅     |
| readv                                          | ✅     |
| writev/fsync/readv                             | ✅     |
| write/read                                     | ✅     |
| splice/read                                    | ✅     |
| write_fixed/read_fixed                         | ✅     |
| openat                                         | ✅     |
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

## Contribute

Pull requests welcome.

Constructive, specific, and actionable feedback is also appreciated.

## Why make more bindings when others exist?

I wanted a simpler interface than tokio-rs/io-uring. I also did not want to link libc, or compile C code. No other library satisifed this goal, so I made my own.
