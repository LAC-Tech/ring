//! Pure Rust `io_uring` bindings that do not rely on libc, or the rust standard
//! library. The api is designed to be simple and straight-forward.
//!
//! Start with [`IoUring`] - this is the primary entry point.
//!
//! This is a systems programming library at the same level of abstraction as
//! `liburing`. while `unsafe` methods are kept  to a minimum,  submitting
//! things to the linux kernel to be modified asynchronously is inherently
//! outside rust's concept of "safety".
//!
//! It is based on the [iouring implementation in the zig standard library](https://codeberg.org/ziglang/zig/src/branch/0.15.x/lib/std/os/linux/iouring.zig),
//! though the design is beginning to diverge. The only dependency in rustix,
//! which is roughly the equivalent of zig's `std.os`.
//!
//! # Example
//!
//! ```no_run
#![doc = include_str!("../examples/readme.rs")]
//! ```
//! 
//! Or, in `no_std`:
//! ```no_run
#![doc = include_str!("../examples/readme_no_std.rs")]
//! ```

pub mod entry;
mod mmap;

pub use crate::entry::*;
pub use rustix;

// These form part of the public API
pub use rustix::fd::BorrowedFd;
pub use rustix::io::Errno;
pub use rustix::io_uring::{
    io_uring_params, iovec, IoringEnterFlags, IoringOp, IoringSqeFlags,
    ReadWriteFlags,
};

use core::ptr::null;
use core::{assert, assert_eq, assert_ne, cmp};
use rustix::fd::{AsFd, OwnedFd};
use rustix::io;
use rustix::io_uring::{
    io_uring_enter, io_uring_register, io_uring_setup, IoringFeatureFlags,
    IoringRegisterOp, IoringSetupFlags, IoringSqFlags,
};

/// The main entry point to the library.
#[derive(Debug)]
pub struct IoUring {
    fd: OwnedFd,
    shared: mmap::Ioring,
    flags: IoringSetupFlags,
    features: IoringFeatureFlags,
    sq: SubmissionQueue,
    cq: CompletionQueue,
}

impl IoUring {
    /// A friendly way to setup an io_uring, with default linux.io_uring_params.
    /// `entries` must be a power of two between 1 and 32768, although the
    /// kernel will make the final call on how many entries the submission
    /// and completion queues will ultimately have, see <https://github.com/torvalds/linux/blob/v5.8/fs/io_uring.c#L8027-L8050>.
    /// Matches the interface of io_uring_queue_init() in liburing.
    pub fn new(entries: u32) -> Result<Self, err::Init> {
        let mut params = io_uring_params::default();
        params.sq_thread_idle = 1000;
        Self::new_with_params(entries, &mut params)
    }
    /// A powerful way to setup an `io_uring`, if you want to tweak
    /// `io_uring_params` such as submission queue thread cpu affinity
    /// or thread idle timeout (the kernel and our default is 1 second).
    /// `params` is passed by reference because the kernel needs to modify the
    /// parameters. Matches the interface of `io_uring_queue_init_params()` in
    /// liburing.
    ///
    /// # Errors
    ///
    /// An [`err::Init::Os`] can be:
    /// - [`Errno::FAULT`] Parameters are outside accessible address space.
    /// - [`Errno::INVAL`] The resv array contains non-zero data, p.flags
    ///   contains an unsupported flag, entries out of bounds,
    ///   [`IoringSetupFlags::SQ_AFF`] was specified without
    ///   [`IoringSetupFlags::SQPOLL`], or [`IoringSetupFlags::CQSIZE`] was
    ///   specified but linux.io_uring_params.cq_entries was invalid.
    /// - [`Errno::MFILE`] Process file descriptor quota exceeded.
    /// - [`Errno::NFILE`] System file descriptor quota exceeded.
    /// - [`Errno::NOMEM`] System resources.
    /// - [`Errno::PERM`] [`IoringSetupFlags::SQPOLL`] was specified but
    ///   effective user ID lacks sufficient privileges, or a container seccomp
    ///   policy prohibits io_uring syscalls.
    /// - [`Errno::NOSYS`] System is outdated (kernel < 5.4).
    pub fn new_with_params(
        entries: u32,
        p: &mut io_uring_params,
    ) -> Result<Self, err::Init> {
        use err::Init::*;
        if entries == 0 {
            return Err(EntriesZero);
        }
        if !entries.is_power_of_two() {
            return Err(EntriesNotPowerOfTwo);
        }

        assert_eq!(p.sq_entries, 0);
        assert!(
            p.cq_entries == 0 || p.flags.contains(IoringSetupFlags::CQSIZE)
        );
        assert!(p.features.is_empty());
        assert!(p.wq_fd == 0 || p.flags.contains(IoringSetupFlags::ATTACH_WQ));
        assert_eq!(p.resv, [0, 0, 0]);

        // SAFETY: `entries` and `p4 have been validated.
        // The kernel will initialize the fields of `p`.
        let res = unsafe { io_uring_setup(entries, p) };
        let fd = res.map_err(Os)?;

        // Kernel versions 5.4 and up use only one mmap() for the submission and
        // completion queues. This is not an optional feature for us...
        // if the kernel does it, we have to do it. The thinking on this
        // by the kernel developers was that both the submission and the
        // completion queue rings have sizes just over a power of two, but the
        // submission queue ring is significantly smaller with u32
        // slots. By bundling both in a single mmap, the kernel gets the
        // submission queue ring for free. See https://patchwork.kernel.org/patch/11115257 for the kernel patch.
        // We do not support the double mmap() done before 5.4, because we want
        // to keep the init/deinit mmap paths simple and because
        // io_uring has had many bug fixes even since 5.4.
        if !p.features.contains(IoringFeatureFlags::SINGLE_MMAP) {
            return Err(SingleMmapUnsupported);
        }

        if p.flags.contains(IoringSetupFlags::CQE32) {
            return Err(CQE32Unsupported);
        }

        // Check that the kernel has actually set params and that "impossible is
        // nothing".
        assert_ne!(p.sq_entries, 0);
        assert_ne!(p.cq_entries, 0);
        assert!(p.cq_entries >= p.sq_entries);

        let (shared, ring_masks) =
            mmap::Ioring::new(fd.as_fd(), p).map_err(Os)?;

        let sqes = mmap::Sqes::new(fd.as_fd(), p).map_err(Os)?;

        let sq = SubmissionQueue {
            entries: p.sq_entries,
            sqes,
            mask: ring_masks.sq,
            sqe_head: 0,
            sqe_tail: 0,
        };
        let cq = CompletionQueue { mask: ring_masks.cq, entries: p.cq_entries };

        Ok(Self { fd, shared, flags: p.flags, features: p.features, sq, cq })
    }

    pub fn fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }

    /// Returns a reference to a vacant SQE, or an error if the submission
    /// queue is full. We follow the implementation (and atomics) of
    /// liburing's `io_uring_get_sqe()`, EXCEPT that the fields are zeroed out.
    pub fn get_sqe(&mut self) -> Result<&mut Sqe, err::GetSqe> {
        let sqe = self.get_sqe_raw()?;
        *sqe = Sqe::default();
        Ok(sqe)
    }

    /// Returns a reference to a vacant SQE, or an error if the submission
    /// queue is full. We follow the implementation (and atomics) of
    /// liburing's `io_uring_get_sqe()` exactly.
    pub fn get_sqe_raw(&mut self) -> Result<&mut Sqe, err::GetSqe> {
        let head = self.shared.sq_head();
        // Remember that these head and tail offsets wrap around every four
        // billion operations. We must therefore use wrapping addition
        // and subtraction to avoid a runtime crash.
        let next = self.sq.sqe_tail.wrapping_add(1);
        if next.wrapping_sub(head) > self.sq.entries {
            return Err(err::GetSqe::SubmissionQueueFull);
        }

        let index = (self.sq.sqe_tail & self.sq.mask) as usize;
        let sqe = &mut self.sq.sqes[index];
        self.sq.sqe_tail = next;
        Ok(sqe)
    }

    /// Submits the SQEs acquired via [`Self::get_sqe()`] to the kernel. You can
    /// call this once after you have called [`Self::get_sqe()`] multiple times
    /// to setup multiple I/O requests. Returns the number of SQEs
    /// submitted, if not used alongside `IORING_SETUP_SQPOLL`. If the
    /// `io_uring` instance is uses `IORING_SETUP_SQPOLL`, the value returned
    /// on success is not guaranteed to match the amount of actually
    /// submitted sqes during this call. A value higher or lower, including
    /// 0, may be returned. Matches the implementation of
    /// `io_uring_submit()` in liburing.
    ///
    /// # Safety
    ///
    /// See [`Self::enter`].
    pub unsafe fn submit(&mut self) -> io::Result<u32> {
        self.submit_and_wait(0)
    }

    /// Like [`Self::submit()`], but allows waiting for events as well.
    /// Returns the number of SQEs submitted.
    /// Matches the implementation of `io_uring_submit_and_wait()` in liburing.
    ///
    /// # Safety
    ///
    /// See [`Self::enter`].
    pub unsafe fn submit_and_wait(&mut self, wait_nr: u32) -> io::Result<u32> {
        let submitted = self.flush_sq();
        let mut flags = IoringEnterFlags::empty();

        if self.sq_ring_needs_enter(&mut flags) || wait_nr > 0 {
            if wait_nr > 0 || self.flags.contains(IoringSetupFlags::IOPOLL) {
                flags.set(IoringEnterFlags::GETEVENTS, true);
            }
            return self.enter(submitted, wait_nr, flags);
        }

        Ok(submitted)
    }

    /// Tell the kernel we have submitted SQEs and/or want to wait for CQEs.
    /// Returns the number of SQEs submitted.
    ///
    /// # Safety
    ///
    /// The caller must ensure that any buffers or file descriptors referenced
    /// by the SQEs remain valid until the kernel has completed the
    /// requested operations.
    ///
    /// # Errors
    ///
    /// An [`io::Errno`] can be:
    /// - [`io::Errno::AGAIN`] The kernel was unable to allocate memory or ran
    ///   out of resources for the request. The application should wait for some
    ///   completions and try again.
    /// - [`Errno::BADF`] The SQE `fd` is invalid, or
    ///   [`IoringSqeFlags::FIXED_FILE`] was set but no files were registered.
    /// - [`Errno::BADFD`] The file descriptor is valid, but the ring is not in
    ///   the right state. See io_uring_register(2) for how to enable the ring.
    /// - [`Errno::BUSY`] The application attempted to overcommit the number of
    ///   requests it can have pending. The application should wait for some
    ///   completions and try again.
    /// - [`Errno::INVAL`] The SQE is invalid, or valid but the ring was setup
    ///   with [`IoringSetupFlags::IOPOLL`].
    /// - [`Errno::FAULT`] The buffer is outside the process' accessible address
    ///   space, or [`IoringOp::ReadFixed`] or [`IoringOp::WriteFixed`] was
    ///   specified but no buffers were registered, or the range described by
    ///   `addr` and `len` is not within the buffer registered at `buf_index`.
    /// - [`Errno::NXIO`] Ring is shutting down.
    /// - [`Errno::OPNOTSUPP`] The kernel believes our `self.fd` does not refer
    ///   to an io_uring instance, or the opcode is valid but not supported by
    ///   this kernel (more likely).
    /// - [`Errno::INTR`] The operation was interrupted by a delivery of a
    ///   signal before it could complete. This can happen while waiting for
    ///   events with [`IoringEnterFlags::GETEVENTS`].
    pub unsafe fn enter(
        &mut self,
        to_submit: u32,
        min_complete: u32,
        flags: IoringEnterFlags,
    ) -> io::Result<u32> {
        io_uring_enter(self.fd.as_fd(), to_submit, min_complete, flags)
    }

    /// Sync internal state with kernel ring state on the SQ side.
    /// Returns the number of all pending events in the SQ ring, for the shared
    /// ring. This return value includes previously flushed SQEs, as per
    /// liburing. The rationale is to suggest that an `io_uring_enter()` call
    /// is needed rather than not. Matches the implementation of
    /// `__io_uring_flush_sq()` in liburing.
    pub fn flush_sq(&mut self) -> u32 {
        if self.sq.sqe_head != self.sq.sqe_tail {
            // Fill in SQEs that we have queued up, adding them to the
            // kernel ring.
            let to_submit = self.sq.sqe_tail.wrapping_sub(self.sq.sqe_head);
            let mut new_tail = self.shared.sq_tail();

            let array = self.shared.sqe_indices();

            for _ in 0..to_submit {
                array[(new_tail & self.sq.mask) as usize] =
                    self.sq.sqe_head & self.sq.mask;
                new_tail = new_tail.wrapping_add(1);
                self.sq.sqe_head = self.sq.sqe_head.wrapping_add(1);
            }

            // Ensure that the kernel can actually see the SQE updates when
            // it sees the tail update.
            self.shared.sq_tail_write(new_tail);
        }

        self.sq_ready()
    }

    /// Returns true if we are not using an SQ thread (thus nobody submits but
    /// us), or if `IORING_SQ_NEED_WAKEUP` is set and the SQ thread must be
    /// explicitly awakened. For the latter case, we set the SQ thread
    /// wakeup flag. Matches the implementation of `sq_ring_needs_enter()` in
    /// liburing.
    pub fn sq_ring_needs_enter(
        &mut self,
        flags: &mut IoringEnterFlags,
    ) -> bool {
        if !self.flags.contains(IoringSetupFlags::SQPOLL) {
            return true;
        }

        if self.shared.sq_flag_contains(IoringSqFlags::NEED_WAKEUP) {
            flags.set(IoringEnterFlags::SQ_WAKEUP, true);
            return true;
        }
        false
    }

    /// Returns the number of flushed and unflushed SQEs pending in the///
    /// submission queue. In other words, this is the number of SQEs in the
    /// submission queue, i.e. its length. These are SQEs that the kernel is
    /// yet to consume. Matches the implementation of `io_uring_sq_ready` in
    /// liburing.
    pub fn sq_ready(&mut self) -> u32 {
        self.sq.sqe_tail.wrapping_sub(self.shared.sq_head())
    }

    /// Returns the number of CQEs in the completion queue, i.e. its length.
    /// These are CQEs that the application is yet to consume.
    /// Matches the implementation of `io_uring_cq_ready` in liburing.
    pub fn cq_ready(&mut self) -> u32 {
        self.shared.cq_tail().wrapping_sub(self.shared.cq_head())
    }

    /// Copies as many CQEs as are ready, and that can fit into the destination
    /// `cqes` slice. If none are available, enters into the kernel to wait
    /// for at most `wait_nr` CQEs. Returns the number of CQEs copied,
    /// advancing the CQ ring. Provides all the wait/peek methods found in
    /// liburing, but with batching and a single method. The rationale for
    /// copying CQEs rather than copying pointers is that pointers are 8 bytes
    /// whereas CQEs are not much more at only 16 bytes, and this provides a
    /// safer faster interface. Safer, because you no longer need to call
    /// [`Self::cqe_seen`], avoiding idempotency bugs. Faster, because we can
    /// now amortize the atomic store release to `cq.head` across the batch. See <https://github.com/axboe/liburing/issues/103#issuecomment-686665007>.
    /// Matches the implementation of `io_uring_peek_batch_cqe()` in liburing,
    /// but supports waiting.
    ///
    /// # Safety
    ///
    /// May call [`Self::enter`].
    pub unsafe fn copy_cqes(
        &mut self,
        cqes: &mut [Cqe],
        wait_nr: u32,
    ) -> io::Result<u32> {
        let count = self.copy_cqes_ready(cqes);
        if count > 0 {
            return Ok(count);
        }
        if self.cq_ring_needs_flush() || wait_nr > 0 {
            self.enter(0, wait_nr, IoringEnterFlags::GETEVENTS)?;
            return Ok(self.copy_cqes_ready(cqes));
        }

        Ok(0)
    }

    /// Returns a copy of an I/O completion, waiting for it if necessary, and
    /// advancing the CQ ring. A convenience method for `copy_cqes()` for
    /// when you don't need to batch or peek.
    ///
    /// # Safety
    ///
    /// May call [`Self::enter`].
    pub unsafe fn copy_cqe(&mut self) -> io::Result<Cqe> {
        let mut cqes = [Cqe::default(); 1];
        loop {
            let count = self.copy_cqes(&mut cqes, 1)?;
            if count > 0 {
                return Ok(cqes[0]);
            }
        }
    }

    fn copy_cqes_ready(&mut self, cqes: &mut [Cqe]) -> u32 {
        let ready = self.cq_ready();
        let count = cmp::min(cqes.len(), ready as usize);
        let head = self.shared.cq_head() & self.cq.mask;

        // before wrapping
        let n = cmp::min((self.cq.entries - head) as usize, count);

        let ring_cqes = self.shared.cqes();

        {
            let head = head as usize;
            cqes[..n].clone_from_slice(&ring_cqes[head..head + n]);
        }

        if count as usize > n {
            // wrap self.cq.cqes
            let w = count as usize - n;
            cqes[n..n + w].clone_from_slice(&ring_cqes[0..w]);
        }

        let count: u32 = count
            .try_into()
            .expect("io_uring expects all counts and offsets to fit into u32");
        self.cq_advance(count);
        count
    }

    /// Matches the implementation of `cq_ring_needs_flush()` in liburing.
    pub fn cq_ring_needs_flush(&mut self) -> bool {
        self.shared.sq_flag_contains(IoringSqFlags::CQ_OVERFLOW)
    }

    /// For advanced use cases only that implement custom completion queue
    /// methods. If you use [`Self::copy_cqes`] or [`Self::copy_cqe`] you must
    /// not call [`Self::cqe_seen()`] or [`Self::cq_advance()`]. Must be called
    /// exactly once after a zero-copy CQE has been processed by your
    /// application. Not idempotent, calling more than once will result in
    /// other CQEs being lost. Matches the implementation of `cqe_seen()` in
    /// liburing.
    // TODO: const pointer param? not using it? review this..
    pub fn cqe_seen(&mut self, _cqe: *const Cqe) {
        self.shared.cq_advance(1);
    }

    /// For advanced use cases only that implement custom completion queue
    /// methods. Matches the implementation of `cq_advance()` in liburing.
    pub fn cq_advance(&mut self, count: u32) {
        self.shared.cq_advance(count);
    }

    /// Registers an array of buffers for use with [`Sqe::prep_readv_fixed`]
    /// and [`Sqe::prep_writev_fixed`].
    ///
    /// # Safety
    ///
    /// The caller must ensure that the file descriptors remain valid until they
    /// have been unregistered.
    ///
    /// # Panics
    ///
    /// If the length of the `fds` cannot fit in a `u32`.
    ///
    /// # Errors
    ///
    /// An [`io::Errno`] can be:
    /// - [`Errno::BADF`] One or more fds in the array are invalid, or the
    ///   kernel does not support sparse sets.
    /// - [`Errno::BUSY`] Files are already registered.
    /// - [`Errno::INVAL`] Files array is empty.
    /// - [`Errno::MFILE`] Adding file references would exceed the maximum
    ///   allowed number of files the user is allowed to have according to the
    ///   per-user RLIMIT_NOFILE resource limit and the CAP_SYS_RESOURCE
    ///   capability is not set, or exceeds the maximum allowed for a fixed file
    ///   set (older kernels have a limit of 1024 files vs 64K files).
    /// - [`Errno::NOMEM`] Insufficient kernel resources, or the caller had a
    ///   non-zero RLIMIT_MEMLOCK soft resource limit but tried to lock more
    ///   memory than the limit permitted (not enforced when the process is
    ///   privileged with CAP_IPC_LOCK).
    /// - [`Errno::NXIO`] Attempt to register files on a ring already
    ///   registering files or being torn down.
    pub unsafe fn register_files(&self, fds: &[BorrowedFd]) -> io::Result<u32> {
        io_uring_register(
            self.fd(),
            IoringRegisterOp::RegisterFiles,
            fds.as_ptr().cast(),
            fds.len().try_into().expect("length of fds must fit in a u32"),
        )
    }

    /// Unregisters all registered file descriptors previously associated with
    /// the ring.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no pending SQEs reference the registered
    /// indices.
    pub unsafe fn unregister_files(&mut self) -> io::Result<u32> {
        io_uring_register(
            self.fd(),
            IoringRegisterOp::UnregisterFiles,
            null(),
            0,
        )
    }

    /// Registers an array of buffers for use with `read_fixed` and
    /// `write_fixed`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no pending SQEs reference the registered
    /// buffers.
    pub unsafe fn register_buffers(
        &mut self,
        buffers: &[iovec],
    ) -> io::Result<u32> {
        io_uring_register(
            self.fd(),
            IoringRegisterOp::RegisterFiles,
            buffers.as_ptr().cast(),
            buffers
                .len()
                .try_into()
                .expect("length of buffers must fit in a u32"),
        )
    }
}

// Unlike the Zig version, we do not store the mmap; as it is used by the
// CompletionQueue as well.
// We do store mmap_entries however, as it is exclusively used by this
// struct
#[derive(Debug)]
struct SubmissionQueue {
    // Contains the array with the actual sq entries
    sqes: mmap::Sqes,
    entries: u32,
    mask: u32,
    // We use `sqe_head` and `sqe_tail` in the same way as liburing:
    // We increment `sqe_tail` (but not `tail`) for each call to
    // `get_sqe()`. We then set `tail` to `sqe_tail` once, only
    // when these events are actually submitted. This allows us to
    // amortize the cost of the @atomicStore to `tail` across multiple
    // SQEs.
    sqe_head: u32,
    sqe_tail: u32,
}

#[derive(Debug)]
struct CompletionQueue {
    mask: u32,
    entries: u32,
}

pub mod err {
    use crate::rustix::io::Errno;
    #[derive(Debug, Eq, PartialEq)]
    pub enum Init {
        EntriesZero,
        EntriesNotPowerOfTwo,
        CQE32Unsupported,
        SingleMmapUnsupported,
        Os(Errno),
    }

    #[derive(Debug, Eq, PartialEq)]
    pub enum GetSqe {
        SubmissionQueueFull,
    }
}

#[cfg(test)]
mod test_ioring_op_uring_cmd {
    use super::*;
    use err::*;
    use pretty_assertions::{assert_eq, assert_ne};

    // Testing it at least prevents us getting in this situation
    #[test]
    fn panics_on_cqe_c32_setup_flag() {
        let mut params = io_uring_params::default();
        params.flags.set(IoringSetupFlags::CQE32, true);

        let ring = IoUring::new_with_params(2, &mut params);

        assert!(
            matches!(ring, Err(Init::CQE32Unsupported)),
            "attempt to construct ring with CQE32 flag not caught"
        );
    }

    // This broken in Zig too:
    // https://codeberg.org/ziglang/zig/issues/30649
    #[test]
    #[ignore]
    fn handles_32bit_sqes() {
        let mut params = io_uring_params::default();
        params.flags.set(IoringSetupFlags::CQE32, true);

        let mut ring = IoUring::new_with_params(2, &mut params).unwrap();

        ring.get_sqe().unwrap().prep_nop(0);
        ring.get_sqe().unwrap().prep_nop(1);
        let submitted = unsafe { ring.submit_and_wait(2) };
        assert_eq!(submitted, Ok(2));

        let cqe1 = unsafe { ring.copy_cqe() }.unwrap();
        let cqe2 = unsafe { ring.copy_cqe() }.unwrap();
        assert_ne!(
            cqe1.user_data.u64_(),
            cqe2.user_data.u64_(),
            "both submitted sqes had different user data"
        );
    }
}

// From IoUring.zig
#[cfg(test)]
mod zig_tests {
    use super::*;
    use err::*;
    use pretty_assertions::assert_eq;
    use rustix::fd::AsRawFd;
    use rustix::fs::{openat, Mode, OFlags, CWD};
    use rustix::{
        // TODO: the only place we use these constants, is in these tests?
        io_uring::{
            io_uring_ptr, ioprio_union, IoringCqeFlags, IoringOp,
            IoringSqeFlags,
        },
    };
    use tempfile::{tempdir, TempDir};

    // It's just a u16 in the C struct
    fn ioprio_to_u16(i: ioprio_union) -> u16 {
        // 100% safe, 'cause Stone Cold said so
        unsafe { core::mem::transmute::<_, u16>(i) }
    }

    fn temp_file(dir: &TempDir, path: &'static str) -> OwnedFd {
        openat(
            CWD,
            dir.path().join(path),
            OFlags::CREATE | OFlags::RDWR | OFlags::TRUNC,
            Mode::RUSR | Mode::WUSR,
        )
        .unwrap()
    }

    /// Helper to verify clean ring state
    fn assert_ring_clean(ring: &mut IoUring) {
        assert_eq!(
            ring.sq_ready(),
            0,
            "Submission queue has {} pending SQEs",
            ring.sq_ready()
        );
        assert_eq!(
            ring.cq_ready(),
            0,
            "Completion queue has {} pending CQEs",
            ring.cq_ready()
        );
        assert_eq!(
            ring.sq.sqe_head, ring.sq.sqe_tail,
            "SQE head/tail mismatch: {} != {}",
            ring.sq.sqe_head, ring.sq.sqe_tail
        );
    }

    // Much of what is tested here in Zig, we do statically
    #[test]
    fn entries() {
        assert_eq!(Init::EntriesZero, IoUring::new(0).unwrap_err());
        assert_eq!(Init::EntriesNotPowerOfTwo, IoUring::new(3).unwrap_err());
    }

    #[test]
    fn nop() {
        let mut ring = IoUring::new(1).unwrap();
        let sqe = ring.get_sqe().unwrap();
        sqe.prep_nop(0xaaaaaaaa);

        assert_eq!(sqe.opcode, IoringOp::Nop);
        assert_eq!(sqe.flags, IoringSqeFlags::empty());
        assert_eq!(
            ioprio_to_u16(sqe.ioprio),
            ioprio_to_u16(ioprio_union::default())
        );
        assert_eq!(sqe.fd, 0);
        assert_eq!(sqe.off(), 0);
        assert_eq!(sqe.addr(), io_uring_ptr::null());
        assert_eq!(
            // This isn't even a union in the C struct??
            unsafe { sqe.len.len },
            0
        );
        assert_eq!(unsafe { sqe.op_flags.rw_flags }, ReadWriteFlags::empty());
        assert_eq!(sqe.user_data.u64_(), 0xaaaaaaaa);
        assert_eq!(unsafe { sqe.buf.buf_index }, 0);
        assert_eq!(sqe.personality, 0);
        assert_eq!(
            unsafe { sqe.splice_fd_in_or_file_index_or_addr_len.splice_fd_in },
            0
        );
        assert_eq!(unsafe { sqe.addr3_or_cmd.addr3.addr3 }, 0);
        // TODO: rustix struct lacks resv...should be fine?

        assert_eq!(ring.sq.sqe_head, 0);
        assert_eq!(ring.sq.sqe_tail, 1);
        assert_eq!(ring.shared.sq_tail(), 0);
        assert_eq!(ring.shared.cq_head() & ring.cq.mask, 0);
        assert_eq!(ring.sq_ready(), 1);
        assert_eq!(ring.cq_ready(), 0);

        assert_eq!(unsafe { ring.submit() }, Ok(1));
        assert_eq!(ring.sq.sqe_head, 1);
        assert_eq!(ring.sq.sqe_tail, 1);
        assert_eq!(ring.shared.sq_tail(), 1);
        assert_eq!(ring.shared.cq_head(), 0);
        assert_eq!(ring.sq_ready(), 0);

        let cqe = unsafe { ring.copy_cqe().unwrap() };
        assert_eq!(cqe.user_data.u64_(), 0xaaaaaaaa);
        assert_eq!(cqe.res, 0);
        assert_eq!(cqe.flags, IoringCqeFlags::empty());
        assert_eq!(ring.shared.cq_head(), 1);
        assert_eq!(ring.cq_ready(), 0);

        let sqe_barrier = ring.get_sqe().unwrap();
        sqe_barrier.prep_nop(0xbbbbbbbb);
        sqe_barrier.flags.set(IoringSqeFlags::IO_DRAIN, true);
        assert_eq!(unsafe { ring.submit() }, Ok(1));
        let cqe = unsafe { ring.copy_cqe().unwrap() };
        assert_eq!(cqe.user_data.u64_(), 0xbbbbbbbb);
        assert_eq!(cqe.res, 0);
        assert_eq!(cqe.flags, IoringCqeFlags::empty());
        assert_eq!(ring.sq.sqe_head, 2);
        assert_eq!(ring.sq.sqe_tail, 2);
        assert_eq!(ring.shared.sq_tail(), 2);
        assert_eq!(ring.shared.cq_head(), 2);

        assert_ring_clean(&mut ring);
    }

    #[test]
    fn readv() {
        let mut ring = IoUring::new(1).unwrap();
        let fd =
            openat(CWD, "/dev/zero", OFlags::RDONLY, Mode::empty()).unwrap();

        // Linux Kernel 5.4 supports IORING_REGISTER_FILES but not sparse fd
        // sets (i.e. an fd of -1). Linux Kernel 5.5 adds support for
        // sparse fd sets. Compare:
        // https://github.com/torvalds/linux/blob/v5.4/fs/io_uring.c#L3119-L3124 vs
        // https://github.com/torvalds/linux/blob/v5.8/fs/io_uring.c#L6687-L6691
        // We therefore avoid stressing sparse fd sets here:
        let registered_fds = [fd.as_fd(); 1];
        let fd_index = 0;
        let registered =
            unsafe { ring.register_files(&registered_fds).unwrap() };

        let mut buffer = [42u8; 128];
        let iovecs = [iovec {
            iov_base: buffer.as_mut_ptr().cast(),
            iov_len: buffer.len(),
        }];
        let sqe = ring.get_sqe().unwrap();
        sqe.prep_readv_fixed(0xcccccccc, fd_index, &iovecs, 0);
        assert_eq!(sqe.opcode, IoringOp::Readv);
        assert!(sqe.flags.contains(IoringSqeFlags::FIXED_FILE));

        // Hack because Ok branch does not implement debug..
        assert!(
            matches!(ring.get_sqe(), Err(GetSqe::SubmissionQueueFull)),
            "expected submission queue to be full"
        );
        assert_eq!(unsafe { ring.submit() }, Ok(1));
        let cqe = unsafe { ring.copy_cqe() }.unwrap();
        assert_eq!(cqe.user_data.u64_(), 0xcccccccc);
        assert_eq!(cqe.res, buffer.len().try_into().unwrap());
        assert_eq!(cqe.flags, IoringCqeFlags::empty());
        assert!(&buffer.iter().all(|&n| n == 0));

        let unregistered = unsafe { ring.unregister_files().unwrap() };
        assert_eq!(registered, unregistered);
        assert_ring_clean(&mut ring);
    }

    #[test]
    fn writev_fsync_readv() {
        let mut ring = IoUring::new(4).unwrap();

        let tmp = tempdir().unwrap();
        let fd = temp_file(&tmp, "test_io_uring_writev_fsync_readv");

        let mut buffer_write: [u8; 128] = [42; 128];
        let iovecs_write = [iovec {
            iov_base: buffer_write.as_mut_ptr().cast(),
            iov_len: buffer_write.len(),
        }];

        let mut buffer_read = [0u8; 128];
        let mut iovecs_read = [iovec {
            iov_base: buffer_read.as_mut_ptr().cast(),
            iov_len: buffer_read.len(),
        }];

        {
            let sqe = ring.get_sqe().unwrap();
            sqe.prep_writev(0xdddddddd, fd.as_fd(), &iovecs_write, 17);
            assert_eq!(sqe.opcode, IoringOp::Writev);
            assert_eq!(sqe.off(), 17);
            sqe.flags.set(IoringSqeFlags::IO_LINK, true);
        }
        {
            let sqe = ring.get_sqe().unwrap();
            sqe.prep_fsync(0xeeeeeeee, fd.as_fd(), ReadWriteFlags::empty());
            assert_eq!(sqe.opcode, IoringOp::Fsync);
            assert_eq!(sqe.fd, fd.as_raw_fd());
            sqe.flags.set(IoringSqeFlags::IO_LINK, true);
        }
        {
            let sqe = ring.get_sqe().unwrap();
            sqe.prep_readv(0xffffffff, fd.as_fd(), &mut iovecs_read, 17);
            assert_eq!(sqe.opcode, IoringOp::Readv);
            assert_eq!(sqe.off(), 17);
        }

        assert_eq!(ring.sq_ready(), 3);
        assert_eq!(unsafe { ring.submit_and_wait(3) }, Ok(3));
        assert_eq!(ring.sq_ready(), 0);
        assert_eq!(ring.cq_ready(), 3);

        let cqe = unsafe { ring.copy_cqe() }.unwrap();
        assert_eq!(cqe.user_data.u64_(), 0xdddddddd);
        assert_eq!(cqe.res, buffer_write.len().try_into().unwrap());
        assert_eq!(cqe.flags, IoringCqeFlags::empty());
        assert_eq!(ring.cq_ready(), 2);

        let cqe = unsafe { ring.copy_cqe() }.unwrap();
        assert_eq!(cqe.user_data.u64_(), 0xeeeeeeee);
        assert_eq!(cqe.res, 0);
        assert_eq!(cqe.flags, IoringCqeFlags::empty());
        assert_eq!(ring.cq_ready(), 1);

        let cqe = unsafe { ring.copy_cqe() }.unwrap();
        assert_eq!(cqe.user_data.u64_(), 0xffffffff);
        assert_eq!(cqe.res, buffer_read.len().try_into().unwrap());
        assert_eq!(cqe.flags, IoringCqeFlags::empty());
        assert_eq!(ring.cq_ready(), 0);

        assert_eq!(&buffer_write[0..], &buffer_read[0..]);
        assert_ring_clean(&mut ring);
    }

    #[test]
    fn write_read() {
        use tempfile::tempdir;
        let mut ring = IoUring::new(2).unwrap();
        let tmp = tempdir().unwrap();
        let fd = temp_file(&tmp, "test_io_uring_write_read");

        const BUFFER_WRITE: [u8; 20] = [97; 20];
        let mut buffer_read = [98u8; 20];

        let sqe_write = ring.get_sqe().unwrap();
        sqe_write.prep_write(0x11111111, fd.as_fd(), &BUFFER_WRITE, 10);
        assert_eq!(sqe_write.opcode, IoringOp::Write);
        assert_eq!(sqe_write.off(), 10);
        sqe_write.flags.set(IoringSqeFlags::IO_LINK, true);

        let sqe_read = ring.get_sqe().unwrap();
        sqe_read.prep_read(0x22222222, fd.as_fd(), &mut buffer_read, 10);
        assert_eq!(sqe_read.opcode, IoringOp::Read);
        assert_eq!(sqe_read.off(), 10);

        assert_eq!(unsafe { ring.submit() }, Ok(2));

        let cqe_write = unsafe { ring.copy_cqe() }.unwrap();
        let cqe_read = unsafe { ring.copy_cqe() }.unwrap();

        assert_eq!(cqe_write.user_data.u64_(), 0x11111111);
        assert_eq!(cqe_write.res, BUFFER_WRITE.len() as i32);
        assert_eq!(cqe_write.flags, IoringCqeFlags::empty());

        assert_eq!(cqe_read.user_data.u64_(), 0x22222222);
        assert_eq!(cqe_read.res, buffer_read.len() as i32);
        assert_eq!(cqe_read.flags, IoringCqeFlags::empty());

        assert_eq!(BUFFER_WRITE, buffer_read);
        assert_ring_clean(&mut ring);
    }

    // TODO: this is slow (5+ seconds) when spawned from a thread, but instant
    // when spawned from the main thread - see examples/
    // Same story when run from liburing - see foreign_examples/
    //
    // This is a bug in the linux kernel:
    // https://lore.kernel.org/io-uring/f98f318f-0c3b-4b01-afb2-2b276f3fe6cd@kernel.dk/
    #[test]
    fn splice_read() {
        let mut ring = IoUring::new(4).unwrap();

        let tmp = tempdir().unwrap();
        let fd_src = temp_file(&tmp, "test_io_uring_splice_src");
        let fd_dst = temp_file(&tmp, "test_io_uring_splice_dst");

        let buffer_write = [97u8; 20];
        let mut buffer_read = [98u8; 20];
        assert_eq!(
            rustix::io::write(fd_src.as_fd(), &buffer_write),
            Ok(buffer_write.len())
        );

        let (reading_fd_pipe, writing_fd_pipe) = rustix::pipe::pipe().unwrap();
        let pipe_offset = u64::MAX;

        {
            let sqe = ring.get_sqe().unwrap();
            sqe.prep_splice(
                0x11111111,
                fd_src.as_fd(),
                0,
                writing_fd_pipe.as_fd(),
                pipe_offset,
                buffer_write.len(),
            );
            assert_eq!(sqe.opcode, IoringOp::Splice);
            assert_eq!(sqe.addr(), io_uring_ptr::null());
            assert_eq!(sqe.off(), pipe_offset);
            sqe.flags.set(IoringSqeFlags::IO_LINK, true);
        }
        {
            let sqe = ring.get_sqe().unwrap();
            sqe.prep_splice(
                0x22222222,
                reading_fd_pipe.as_fd(),
                pipe_offset,
                fd_dst.as_fd(),
                10,
                buffer_write.len(),
            );
            assert_eq!(sqe.opcode, IoringOp::Splice);
            assert_eq!(
                unsafe { sqe.addr_or_splice_off_in.splice_off_in },
                pipe_offset
            );
            assert_eq!(sqe.off(), 10);
            sqe.flags.set(IoringSqeFlags::IO_LINK, true);
        }

        {
            let sqe = ring.get_sqe().unwrap();
            sqe.prep_read(0x33333333, fd_dst.as_fd(), &mut buffer_read, 10);
            assert_eq!(sqe.opcode, IoringOp::Read);
            assert_eq!(sqe.off(), 10);
        }

        // TODO: this slows everything down!!! why is it so slow?
        assert_eq!(unsafe { ring.submit() }, Ok(3));

        let cqe_splice_to_pipe = unsafe { ring.copy_cqe() }.unwrap();
        let cqe_splice_from_pipe = unsafe { ring.copy_cqe() }.unwrap();
        let cqe_read = unsafe { ring.copy_cqe() }.unwrap();

        assert_eq!(cqe_splice_to_pipe.user_data.u64_(), 0x11111111);
        assert_eq!(cqe_splice_to_pipe.res, buffer_write.len() as i32);
        assert_eq!(cqe_splice_to_pipe.flags, IoringCqeFlags::empty());

        assert_eq!(cqe_splice_from_pipe.user_data.u64_(), 0x22222222);
        assert_eq!(cqe_splice_from_pipe.res, buffer_write.len() as i32);
        assert_eq!(cqe_splice_from_pipe.flags, IoringCqeFlags::empty());

        assert_eq!(cqe_read.user_data.u64_(), 0x33333333);
        assert_eq!(cqe_read.res, buffer_read.len() as i32);
        assert_eq!(cqe_read.flags, IoringCqeFlags::empty());

        assert_eq!(&buffer_write, &buffer_read);
        assert_ring_clean(&mut ring);
    }

    #[test]
    fn write_fixed_read_fixed() {
        let mut ring = IoUring::new(2).unwrap();

        let tmp = tempdir().unwrap();
        let fd = temp_file(&tmp, "test_io_uring_write_read_fixed");

        let mut raw_buffers: [[u8; 11]; 2] = [[0; 11]; 2];
        // First buffer will be written to the file.
        raw_buffers[0].fill(b'z');
        raw_buffers[0][.."foobar".len()].copy_from_slice(b"foobar");

        let buffers = [
            iovec {
                iov_base: raw_buffers[0].as_mut_ptr().cast(),
                iov_len: raw_buffers[0].len(),
            },
            iovec {
                iov_base: raw_buffers[1].as_mut_ptr().cast(),
                iov_len: raw_buffers[1].len(),
            },
        ];

        let registered = unsafe { ring.register_buffers(&buffers).unwrap() };
    }
}
