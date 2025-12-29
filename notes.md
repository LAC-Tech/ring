Some notes for my own benefit; though they may be of use to others.

I am closefly following IoUring.zig` in the Zig Standard Library, version 0.15.2.

## Atomic Access of Shared Memory

We can see atomic operations used to access parts of the main mmap:

| Zig Name   | `io_uring_params` | reads      | writes  |
| ---------- | ----------------- | ---------- | ------- |
| `sq.flags` | `sq_off.flags`    | unordered  | N/A     |
| `sq.head`  | `sq_off.head`     | acquire    | N/A     |
| `sq.tail`  | `sq_off.tail`     | non-atomic | release |
| `cq.head`  | `cq_off.head`     | non-atomic | release |
| `cq.tail`  | `cq_off.tail`     | acquire    | N/A     |

This seems to roughly match what is described in `man 7 io_uring`:

> You add SQEs to the tail of the SQ. The kernel reads SQEs off the head of the queue.
> The kernel adds CQEs to the tail of the CQ. You read CQEs off the head of the queue.

## Zig vs Rust Atomics

Zig atomic orders are not well documented; there's a TODO in the main docs. [Their values are](https://ziglang.org/documentation/0.15.2/std/#std.builtin.AtomicOrder):

They have the same name as those in [C](https://en.cppreference.com/w/c/atomic/memory_order.html). TODO: confirm liburing uses them the same way as Zig.

- `unordered`
- `monotonic`
- `acquire`
- `release`
- `acq_rel`
- `seq_cst`

Rust Atomics [are explicilty documented as following the same rules as C++20 atomics](https://doc.rust-lang.org/core/sync/atomic/enum.Ordering.html):

- `Relaxed`
- `Release`
- `Acquire`
- `AcqRel`
- `SeqCst`

### Correspondence between Zig and Rust orderings

Limiting ourselves to the orderings found in `IoUring.zig`. I will quote cppreference, and the rust docs in order

#### `unordered` -> `Relaxed`

> there are no synchronization or ordering constraints imposed on other reads or writes, only this operation's atomicity is guaranteed

> No ordering constraints, only atomic operations.

#### `acquire` -> `Acquire`

> A load operation with this memory order performs the acquire operation on the affected memory location: no reads or writes in the current thread can be reordered before this load. All writes in other threads that release the same atomic variable are visible in the current thread

> When coupled with a load, if the loaded value was written by a store operation with Release (or stronger) ordering, then all subsequent operations become ordered after that store. In particular, all subsequent loads will see data written before the store.

#### `release` -> `Release`

> A store operation with this memory order performs the release operation: no reads or writes in the current thread can be reordered after this store. All writes in the current thread are visible in other threads that acquire the same atomic variable ... and writes that carry a dependency into the atomic variable become visible in other threads that consume the same atomic

> When coupled with a store, all previous operations become ordered before any load of this value with Acquire (or stronger) ordering. In particular, all previous writes become visible to all threads that perform an Acquire (or stronger) load of this value.

## Safety

### Safe boundary around Shared Memory

If we follow the correct offsets, and atomic reads/writes, of the shared memory, perhaps that could form a safe boundary lower down in the code

### SQE methods should be unsafe

We can't track the lifetime of buffers and fd's submitted to the kernel for writing. These should be unsafe... or is it just unsafe once we submit them!?
