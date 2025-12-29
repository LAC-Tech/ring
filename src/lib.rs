// https://codeberg.org/ziglang/zig/src/branch/0.15.x/lib/std/os/linux/IoUring.zig
// Licensed under MIT License:
// https://codeberg.org/ziglang/zig/src/branch/0.15.x/LICENSE

#![cfg_attr(not(test), no_std)]
// I need to CONSTATLY cast things between usize and u32
const _: () = assert!(usize::BITS >= 32);

pub use rustix;

use core::ffi::c_void;
use core::sync::atomic::{AtomicU32, Ordering};
use core::{assert, assert_eq, assert_ne, cmp, mem, ptr};
use rustix::fd::{AsFd, BorrowedFd, OwnedFd};
use rustix::io;
use rustix::io_uring::{
    io_cqring_offsets, io_sqring_offsets, io_uring_cqe, io_uring_enter,
    io_uring_params, io_uring_ptr, io_uring_register, io_uring_setup,
    io_uring_sqe, iovec, IoringEnterFlags, IoringFeatureFlags, IoringOp,
    IoringSetupFlags, IoringSqFlags, IORING_OFF_SQES, IORING_OFF_SQ_RING,
};

use rustix::mm;

#[derive(Debug)]
pub struct IoUring {
    pub fd: OwnedFd,
    // A single mmap call that contains both the sq and cq
    mmap: RwMmap,
    pub flags: IoringSetupFlags,
    pub features: IoringFeatureFlags,
    sq: SubmissionQueue,
    cq: CompletionQueue,
}

impl IoUring {
    /// A friendly way to setup an io_uring, with default linux.io_uring_params.
    /// `entries` must be a power of two between 1 and 32768, although the
    /// kernel will make the final call on how many entries the submission
    /// and completion queues will ultimately have, see https://github.com/torvalds/linux/blob/v5.8/fs/io_uring.c#L8027-L8050.
    /// Matches the interface of io_uring_queue_init() in liburing.
    pub fn new(entries: u32) -> Result<Self, err::Init> {
        let mut params = io_uring_params::default();
        params.sq_thread_idle = 1000;
        Self::new_with_params(entries, &mut params)
    }

    /// A powerful way to setup an io_uring, if you want to tweak
    /// linux.io_uring_params such as submission queue thread cpu affinity
    /// or thread idle timeout (the kernel and our default is 1 second).
    /// `params` is passed by reference because the kernel needs to modify the
    /// parameters. Matches the interface of io_uring_queue_init_params() in
    /// liburing.
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

        use io::Errno;

        let res = unsafe { io_uring_setup(entries, p) };
        let fd = res.map_err(|errno| match errno {
            Errno::FAULT => ParamsOutsideAccessibleAddressSpace,
            Errno::INVAL => ArgumentsInvalid,
            Errno::MFILE => ProcessFdQuotaExceeded,
            Errno::NFILE => SystemFdQuotaExceeded,
            Errno::NOMEM => SystemResources,
            Errno::PERM => PermissionDenied,
            Errno::NOSYS => SystemOutdated,
            _ => UnexpectedErrno(errno),
        })?;

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
            return Err(SystemOutdated);
        }

        // Check that the kernel has actually set params and that "impossible is
        // nothing".
        assert_ne!(p.sq_entries, 0);
        assert_ne!(p.cq_entries, 0);
        assert!(p.cq_entries >= p.sq_entries);

        let size = cmp::max::<u32>(
            p.sq_off.array + p.sq_entries * mem::size_of::<u32>() as u32,
            p.cq_off.cqes
                + p.cq_entries * mem::size_of::<io_uring_cqe>() as u32,
        ) as usize;

        let mmap = RwMmap::new(
            size,
            fd.as_fd(),
            mm::MapFlags::SHARED | mm::MapFlags::POPULATE,
            IORING_OFF_SQ_RING,
        )
        .map_err(UnexpectedErrno)?;

        let sq_mask = unsafe { mmap.u32_at(p.sq_off.ring_mask) };
        let sq = SubmissionQueue::new(p, fd.as_fd(), sq_mask)
            .map_err(UnexpectedErrno)?;
        let cq = unsafe { CompletionQueue::new(&p, &mmap) };

        // Check that our starting state is as we expect.
        assert_eq!(unsafe { sq.read_head(&mmap) }, 0);
        assert_eq!(unsafe { sq.read_tail(&mmap) }, 0);
        assert_eq!(sq.mask, p.sq_entries - 1);
        // Allow flags.* to be non-zero, since the kernel may set
        // IORING_SQ_NEED_WAKEUP at any time.
        assert_eq!(unsafe { mmap.u32_at(p.sq_off.dropped) }, 0);
        assert_eq!(sq.sqe_head, 0);
        assert_eq!(sq.sqe_tail, 0);

        assert_eq!(unsafe { cq.read_head(&mmap) }, 0);
        assert_eq!(unsafe { cq.read_tail(&mmap) }, 0);
        assert_eq!(cq.mask, p.cq_entries - 1);
        assert_eq!(unsafe { mmap.u32_at(p.cq_off.overflow) }, 0);

        // We expect the kernel copies p.sq_entries to the u32 pointed to by
        // p.sq_off.ring_entries, see https://github.com/torvalds/linux/blob/v5.8/fs/io_uring.c#L7843-L7844.
        assert_eq!(unsafe { mmap.u32_at(p.sq_off.ring_entries) }, p.sq_entries);

        Ok(Self { fd, mmap, flags: p.flags, features: p.features, sq, cq })
    }

    /// Returns a reference to a vacant SQE, or an error if the submission
    /// queue is full. We follow the implementation (and atomics) of
    /// liburing's `io_uring_get_sqe()`, EXCEPT that the fields are zeroed out.
    pub unsafe fn get_sqe(&mut self) -> Result<*mut io_uring_sqe, err::GetSqe> {
        let sqe = self.sq.get_sqe(&self.mmap)?;
        *sqe = io_uring_sqe::default();
        Ok(sqe)
    }

    /// Returns a reference to a vacant SQE, or an error if the submission
    /// queue is full. We follow the implementation (and atomics) of
    /// liburing's `io_uring_get_sqe()` exactly.
    /// TODO: is it reasonable/possible to make this "safe" and return a
    /// reference?
    pub unsafe fn get_sqe_raw(
        &mut self,
    ) -> Result<*mut io_uring_sqe, err::GetSqe> {
        self.sq.get_sqe(&self.mmap)
    }

    /// Submits the SQEs acquired via get_sqe() to the kernel. You can call this
    /// once after you have called get_sqe() multiple times to setup
    /// multiple I/O requests. Returns the number of SQEs submitted, if not
    /// used alongside IORING_SETUP_SQPOLL. If the io_uring instance is uses
    /// IORING_SETUP_SQPOLL, the value returned on success is not guaranteed
    /// to match the amount of actually submitted sqes during this call. A value
    /// higher or lower, including 0, may be returned.
    /// Matches the implementation of io_uring_submit() in liburing.
    pub unsafe fn submit(&mut self) -> Result<u32, err::Enter> {
        self.submit_and_wait(0)
    }

    /// Like submit(), but allows waiting for events as well.
    /// Returns the number of SQEs submitted.
    /// Matches the implementation of io_uring_submit_and_wait() in liburing.
    pub unsafe fn submit_and_wait(
        &mut self,
        wait_nr: u32,
    ) -> Result<u32, err::Enter> {
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
    pub unsafe fn enter(
        &mut self,
        to_submit: u32,
        min_complete: u32,
        flags: IoringEnterFlags,
    ) -> Result<u32, err::Enter> {
        use err::Enter::*;
        use rustix::io::Errno;

        io_uring_enter(self.fd.as_fd(), to_submit, min_complete, flags).map_err(
            |err| match err {
                Errno::AGAIN => SystemResources,
                Errno::BADF => FileDescriptorInvalid,
                Errno::BADFD => FileDescriptorInBadState,
                Errno::BUSY => CompletionQueueOvercommitted,
                Errno::INVAL => SubmissionQueueEntryInvalid,
                Errno::FAULT => BufferInvalid,
                Errno::NXIO => RingShuttingDown,
                Errno::OPNOTSUPP => OpcodeNotSupported,
                Errno::INTR => SignalInterrupt,
                errno => UnexpectedErrno(errno),
            },
        )
    }

    /// Sync internal state with kernel ring state on the SQ side.
    /// Returns the number of all pending events in the SQ ring, for the shared
    /// ring. This return value includes previously flushed SQEs, as per
    /// liburing. The rationale is to suggest that an io_uring_enter() call
    /// is needed rather than not. Matches the implementation of
    /// __io_uring_flush_sq() in liburing.
    pub unsafe fn flush_sq(&mut self) -> u32 {
        self.sq.flush(&mut self.mmap)
    }

    /// Returns true if we are not using an SQ thread (thus nobody submits but
    /// us), or if IORING_SQ_NEED_WAKEUP is set and the SQ thread must be
    /// explicitly awakened. For the latter case, we set the SQ thread
    /// wakeup flag. Matches the implementation of sq_ring_needs_enter() in
    /// liburing.
    pub unsafe fn sq_ring_needs_enter(
        &mut self,
        flags: &mut IoringEnterFlags,
    ) -> bool {
        assert!(flags.is_empty());
        if !self.flags.contains(IoringSetupFlags::SQPOLL) {
            return true;
        }

        if self.sq.flag_contains(&self.mmap, IoringSqFlags::NEED_WAKEUP) {
            flags.set(IoringEnterFlags::SQ_WAKEUP, true);
            return true;
        }
        false
    }

    /// Returns the number of flushed and unflushed SQEs pending in the
    /// submission queue. In other words, this is the number of SQEs in the
    /// submission queue, i.e. its length. These are SQEs that the kernel is
    /// yet to consume. Matches the implementation of io_uring_sq_ready in
    /// liburing.
    pub unsafe fn sq_ready(&mut self) -> u32 {
        self.sq.ready(&mut self.mmap)
    }

    /// Returns the number of CQEs in the completion queue, i.e. its length.
    /// These are CQEs that the application is yet to consume.
    /// Matches the implementation of io_uring_cq_ready in liburing.
    pub unsafe fn cq_ready(&mut self) -> u32 {
        self.cq.ready(&self.mmap)
    }

    /// Copies as many CQEs as are ready, and that can fit into the destination
    /// `cqes` slice. If none are available, enters into the kernel to wait
    /// for at most `wait_nr` CQEs. Returns the number of CQEs copied,
    /// advancing the CQ ring. Provides all the wait/peek methods found in
    /// liburing, but with batching and a single method. The rationale for
    /// copying CQEs rather than copying pointers is that pointers are 8 bytes
    /// whereas CQEs are not much more at only 16 bytes, and this provides a
    /// safer faster interface. Safer, because you no longer need to call
    /// cqe_seen(), avoiding idempotency bugs. Faster, because we can now
    /// amortize the atomic store release to `cq.head` across the batch. See https://github.com/axboe/liburing/issues/103#issuecomment-686665007.
    /// Matches the implementation of io_uring_peek_batch_cqe() in liburing, but
    /// supports waiting.
    pub unsafe fn copy_cqes(
        &mut self,
        cqes: &mut [io_uring_cqe],
        wait_nr: u32,
    ) -> Result<u32, err::Enter> {
        let count = self.cq.copy_ready_events(&mut self.mmap, cqes);
        if count > 0 {
            return Ok(count);
        }
        if self.cq_ring_needs_flush() || wait_nr > 0 {
            self.enter(0, wait_nr, IoringEnterFlags::GETEVENTS)?;
            return Ok(self.cq.copy_ready_events(&mut self.mmap, cqes));
        }

        Ok(0)
    }

    /// Returns a copy of an I/O completion, waiting for it if necessary, and
    /// advancing the CQ ring. A convenience method for `copy_cqes()` for
    /// when you don't need to batch or peek.
    pub unsafe fn copy_cqe(&mut self) -> Result<io_uring_cqe, err::Enter> {
        let mut cqes = [io_uring_cqe::default(); 1];
        loop {
            let count = self.copy_cqes(&mut cqes, 1)?;
            if count > 0 {
                return Ok(core::ptr::read(&cqes[0]));
            }
        }
    }

    /// Matches the implementation of cq_ring_needs_flush() in liburing.
    pub unsafe fn cq_ring_needs_flush(&mut self) -> bool {
        self.sq.flag_contains(&self.mmap, IoringSqFlags::CQ_OVERFLOW)
    }

    /// For advanced use cases only that implement custom completion queue
    /// methods. If you use copy_cqes() or copy_cqe() you must not call
    /// cqe_seen() or cq_advance(). Must be called exactly once after a
    /// zero-copy CQE has been processed by your application.
    /// Not idempotent, calling more than once will result in other CQEs being
    /// lost. Matches the implementation of cqe_seen() in liburing.
    pub unsafe fn cqe_seen(&mut self, _cqe: *const io_uring_cqe) {
        self.cq.advance(&mut self.mmap, 1);
    }

    /// For advanced use cases only that implement custom completion queue
    /// methods. Matches the implementation of cq_advance() in liburing.
    pub unsafe fn cq_advance(&mut self, count: u32) {
        self.cq.advance(&mut self.mmap, count);
    }
}

/// Prepares SQEs to perform various syscalls when they are submitted
/// These all only set the relevant fields, they do not zero out everything
// I originally thought about making this a trait, but there's weird semantics
// with raw pointers I do not want to deal with.
// And honestly it's six of one, half a dozen of the other in terms of DX
pub mod prep {
    use core::ffi::c_void;
    use rustix::io_uring::{io_uring_ptr, io_uring_sqe, iovec, IoringOp};

    /// A no-op is more useful than may appear at first glance. For example, you
    /// could call `drain_previous_sqes()` on the returned SQE, to use the no-op
    /// to know when the ring is idle before acting on a kill signal.
    pub unsafe fn nop(sqe: *mut io_uring_sqe, user_data: u64) {
        (*sqe).opcode = IoringOp::Nop;
        (*sqe).user_data = user_data.into();
    }

    pub unsafe fn read(
        sqe: *mut io_uring_sqe,
        user_data: u64,
        fd: i32,
        buffer: &mut [u8],
        offset: u64,
    ) {
        (*sqe).opcode = IoringOp::Read;
        (*sqe).fd = fd;
        (*sqe).addr_or_splice_off_in.addr =
            io_uring_ptr::new(buffer.as_mut_ptr() as *mut c_void);
        (*sqe).len.len = buffer.len() as u32;
        (*sqe).off_or_addr2.off = offset;
        (*sqe).user_data.u64_ = user_data;
    }

    pub unsafe fn readv(
        sqe: *mut io_uring_sqe,
        user_data: u64,
        fd: i32,
        // const in libc in zig; they point to mutable buffers
        iovecs: &[iovec],
        offset: u64,
    ) {
        (*sqe).opcode = IoringOp::Readv;
        (*sqe).fd = fd;
        (*sqe).addr_or_splice_off_in.addr =
            io_uring_ptr::new(iovecs.as_ptr() as *mut c_void);
        (*sqe).len.len = iovecs.len() as u32;
        (*sqe).off_or_addr2.off = offset;
        (*sqe).user_data.u64_ = user_data;
    }
}

// Unlike the Zig version, we do not store the mmap; as it is used by the
// CompletionQueue as well.
// We do store mmap_entries however, as it is exclusively used by this
// struct
#[derive(Debug)]
pub struct SubmissionQueue {
    // Contains the array with the actual sq entries
    mmap_entries: RwMmap,
    off: io_sqring_offsets,
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

impl SubmissionQueue {
    pub fn new(
        p: &io_uring_params,
        fd: BorrowedFd<'_>,
        mask: u32,
    ) -> io::Result<Self> {
        /*
         * TODO: wtf does this mean:
         * "The motivation for the `sqes` and `array` indirection is to
         * make it possible for the application to preallocate static
         * linux.io_uring_sqe entries and then replay them when needed."
         */
        let size_sqes =
            (p.sq_entries * mem::size_of::<io_uring_sqe>() as u32) as usize;

        let mmap_entries = RwMmap::new(
            size_sqes,
            fd,
            mm::MapFlags::SHARED | mm::MapFlags::POPULATE,
            IORING_OFF_SQES,
        )?;
        // TODO: usless assert?
        assert_eq!(mmap_entries.len, size_sqes);

        let sq = Self {
            entries: p.sq_entries,
            off: p.sq_off,
            mmap_entries,
            mask,
            sqe_head: 0,
            sqe_tail: 0,
        };

        Ok(sq)
    }

    // man 7 io_uring:
    // - "You add SQEs to the tail of the SQ. The kernel reads SQEs off the head
    //   of the queue."
    unsafe fn read_head(&self, mmap: &RwMmap) -> u32 {
        mmap.atomic_load_u32_at(self.off.head, Ordering::Acquire)
    }

    unsafe fn flag_contains(&self, mmap: &RwMmap, flag: IoringSqFlags) -> bool {
        let bits = mmap.atomic_load_u32_at(self.off.flags, Ordering::Relaxed);
        let flags = IoringSqFlags::from_bits_retain(bits);
        flags.contains(flag)
    }

    // This method helped me catch a bug. I do not care if it is shallow.
    unsafe fn read_tail(&self, mmap: &RwMmap) -> u32 {
        mmap.u32_at(self.off.tail)
    }

    unsafe fn get_sqe(
        &mut self,
        mmap: &RwMmap,
    ) -> Result<*mut io_uring_sqe, err::GetSqe> {
        let head = self.read_head(mmap);
        // Remember that these head and tail offsets wrap around every four
        // billion operations. We must therefore use wrapping addition
        // and subtraction to avoid a runtime crash.
        let next = self.sqe_tail.wrapping_add(1);
        if next.wrapping_sub(head) > self.entries {
            return Err(err::GetSqe::SubmissionQueueFull);
        }

        let sqe = self.mmap_entries.mut_ptr_at(self.sqe_tail & self.mask);

        self.sqe_tail = next;
        Ok(sqe)
    }

    unsafe fn flush(&mut self, mmap: &mut RwMmap) -> u32 {
        if self.sqe_head != self.sqe_tail {
            // Fill in SQEs that we have queued up, adding them to the
            // kernel ring.
            let to_submit = self.sqe_tail.wrapping_sub(self.sqe_head);
            let mut tail = self.read_tail(mmap);

            for _ in 0..to_submit {
                let sqe: *mut u32 =
                    mmap.mut_ptr_at(self.off.array + (tail & self.mask));
                *sqe = self.sqe_head & self.mask;
                tail = tail.wrapping_add(1);
                self.sqe_head = self.sqe_head.wrapping_add(1);
            }

            // Ensure that the kernel can actually see the SQE updates when
            // it sees the tail update.
            mmap.atomic_u32_at(self.off.tail).store(tail, Ordering::Release);
        }

        self.ready(mmap)
    }

    unsafe fn ready(&mut self, mmap: &mut RwMmap) -> u32 {
        // Always use the shared ring state (i.e. not self.sqe_head) to
        // avoid going out of sync, see https://github.com/axboe/liburing/issues/92.
        self.sqe_tail.wrapping_sub(self.read_head(mmap))
    }
}

// Again, we do not store the mmap, as this is shared
#[derive(Debug)]
pub struct CompletionQueue {
    off: io_cqring_offsets,
    mask: u32,
    entries: u32,
}

impl CompletionQueue {
    unsafe fn new(p: &io_uring_params, mmap: &RwMmap) -> Self {
        Self {
            off: p.cq_off,
            mask: mmap.u32_at(p.cq_off.ring_mask),

            entries: p.cq_entries,
        }
    }

    unsafe fn read_head(&self, mmap: &RwMmap) -> u32 {
        mmap.u32_at(self.off.head)
    }

    unsafe fn read_tail(&self, mmap: &RwMmap) -> u32 {
        mmap.atomic_load_u32_at(self.off.tail, Ordering::Acquire)
    }

    unsafe fn ready(&mut self, mmap: &RwMmap) -> u32 {
        self.read_tail(mmap).wrapping_sub(self.read_head(mmap))
    }

    unsafe fn advance(&mut self, mmap: &mut RwMmap, count: u32) {
        if count > 0 {
            mmap.atomic_u32_at(self.off.head)
                .fetch_add(count, Ordering::Release);
        }
    }

    unsafe fn copy_ready_events(
        &mut self,
        mmap: &mut RwMmap,
        cqes: &mut [io_uring_cqe],
    ) -> u32 {
        let ready = self.ready(mmap);
        let count = cmp::min(cqes.len() as u32, ready);
        let head = mmap.u32_at(self.off.head) & self.mask;

        // before wrapping
        let n = cmp::min(self.entries - head, count) as usize;

        let ring_cqes =
            mmap.slice_at::<io_uring_cqe>(self.off.cqes, self.entries);

        // io_uring_cqe is not copyable; it has an array field "big_cqe"
        // since big_cqe never seems to be read we shall copy it anyway
        {
            let head = head as usize;
            let src = ring_cqes[head..head + n].as_ptr();
            let dst = cqes.as_mut_ptr();
            core::ptr::copy_nonoverlapping(src, dst, n);
        }

        if count as usize > n {
            // wrap self.cq.cqes
            let w = count as usize - n;
            {
                let src = ring_cqes[0..w].as_ptr();
                let dst = cqes[n..n + w].as_mut_ptr();
                core::ptr::copy_nonoverlapping(src, dst, w);
            }
        }

        self.advance(mmap, count);
        count
    }
}

#[derive(Debug)]
struct RwMmap {
    ptr: *mut c_void,
    len: usize,
}

impl RwMmap {
    fn new(
        size: usize,
        fd: BorrowedFd,
        flags: mm::MapFlags,
        offset: u64,
    ) -> rustix::io::Result<Self> {
        let prot_flags = mm::ProtFlags::READ | mm::ProtFlags::WRITE;
        let ptr = unsafe {
            mm::mmap(ptr::null_mut(), size, prot_flags, flags, fd, offset)?
        };

        Ok(Self { ptr, len: size })
    }
}

impl Drop for RwMmap {
    fn drop(&mut self) {
        unsafe {
            // If we fail while unmapping memory I don't know what to do.
            mm::munmap(self.ptr, self.len).expect("munmap failed")
        }
    }
}

// All our pointer accesses are by u32s..
impl RwMmap {
    unsafe fn ptr_at<T>(&self, byte_offset: u32) -> *const T {
        (self.ptr as *const u8).add(byte_offset as usize) as *const T
    }

    // just avoids a bit of line noise all the times I do this
    unsafe fn u32_at(&self, byte_offset: u32) -> u32 {
        *self.ptr_at(byte_offset)
    }

    unsafe fn mut_ptr_at<T>(&mut self, byte_offset: u32) -> *mut T {
        (self.ptr as *mut u8).add(byte_offset as usize) as *mut T
    }

    // Atomic needs a mutable ptr but we never mutate the value
    // So this method avoids &mut self when we just want to read
    unsafe fn atomic_load_u32_at(
        &self,
        byte_offset: u32,
        ordering: Ordering,
    ) -> u32 {
        let ptr = self.ptr_at::<u32>(byte_offset) as *mut u32;
        AtomicU32::from_ptr(ptr).load(ordering)
    }
    /// Use this when you need to do atomic writes
    unsafe fn atomic_u32_at(&mut self, byte_offset: u32) -> &AtomicU32 {
        AtomicU32::from_ptr(self.mut_ptr_at(byte_offset))
    }

    unsafe fn slice_at<T>(&self, byte_offset: u32, len: u32) -> &[T] {
        let ptr = self.ptr_at::<T>(byte_offset);
        &*ptr::slice_from_raw_parts(ptr, len as usize)
    }
}

mod err {
    #[derive(Debug, Eq, PartialEq)]
    pub enum Init {
        EntriesZero,
        EntriesNotPowerOfTwo,
        ParamsOutsideAccessibleAddressSpace,

        /// The resv array contains non-zero data, p.flags contains an
        /// unsupported flag, entries out of bounds,
        /// IORING_SETUP_SQ_AFF was specified without IORING_SETUP_SQPOLL,
        /// or IORING_SETUP_CQSIZE was specified but
        /// linux.io_uring_params.cq_entries was invalid:
        ArgumentsInvalid,
        ProcessFdQuotaExceeded,
        SystemFdQuotaExceeded,
        SystemResources,
        /// IORING_SETUP_SQPOLL was specified but effective user ID lacks
        /// sufficient privileges, or a container seccomp policy
        /// prohibits io_uring syscalls:
        PermissionDenied,
        SystemOutdated,
        UnexpectedErrno(rustix::io::Errno),
    }

    #[derive(Debug, Eq, PartialEq)]
    pub enum Enter {
        /// The kernel was unable to allocate memory or ran out of resources
        /// for the request. The application should waitufor some
        /// completions and try again:
        SystemResources,
        /// The SQE `fd` is invalid, or IOSQE_FIXED_FILE was set but no files
        /// were registered:
        FileDescriptorInvalid,
        /// The file descriptor is valid, but the ring is not in the right
        /// state. See io_uring_register(2) for how to enable the ring.
        FileDescriptorInBadState,
        /// The application attempted to overcommit the number of requests it
        /// can have pending. The application should wait for some
        /// completions and try again:
        CompletionQueueOvercommitted,
        /// The SQE is invalid, or valid but the ring was setup with
        /// IORING_SETUP_IOPOLL:
        SubmissionQueueEntryInvalid,
        /// The buffer is outside the process' accessible address space, or
        /// IORING_OP_READ_FIXED or IORING_OP_WRITE_FIXED was specified
        /// but no buffers were registered, or the range described by
        /// `addr` and `len` is not within the buffer registered at
        /// `buf_index`:
        BufferInvalid,
        RingShuttingDown,
        /// The kernel believes our `self.fd` does not refer to an io_uring
        /// instance, or the opcode is valid but not supported by this
        /// kernel (more likely):
        OpcodeNotSupported,
        /// The operation was interrupted by a delivery of a signal before it
        /// could complete. This can happen while waiting for events
        /// with IORING_ENTER_GETEVENTS:
        SignalInterrupt,
        UnexpectedErrno(rustix::io::Errno),
    }

    #[derive(Debug)]
    pub enum GetSqe {
        SubmissionQueueFull,
    }

    #[derive(Debug)]
    pub enum Register {
        /// One or more fds in the array are invalid, or the kernel does not
        /// support sparse sets:
        FileDescriptorInvalid,
        FilesAlreadyRegistered,
        FilesEmpty,
        /// Adding `nr_args` file references would exceed the maximum allowed
        /// number of files the user is allowed to have according to
        /// the per-user RLIMIT_NOFILE resource limit and
        /// the CAP_SYS_RESOURCE capability is not set, or `nr_args` exceeds
        /// the maximum allowed for a fixed file set (older kernels
        /// have a limit of 1024 files vs 64K files):
        UserFdQuotaExceeded,
        /// Insufficient kernel resources, or the caller had a non-zero
        /// RLIMIT_MEMLOCK soft resource limit but tried to lock more
        /// memory than the limit permitted (not enforced
        /// when the process is privileged with CAP_IPC_LOCK):
        SystemResources,
        // Attempt to register files on a ring already registering files or
        // being torn down:
        RingShuttingDownOrAlreadyRegisteringFiles,
        UnexpectedErrno(rustix::io::Errno),
    }

    #[derive(Debug)]
    pub enum RegisterBufRing {
        ArgumentsInvalid,
        UnexpectedErrno(rustix::io::Errno),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use err::*;
    use pretty_assertions::assert_eq;
    use rustix::{
        io::ReadWriteFlags,
        // TODO: the only place we use these constants, is in these tests?
        io_uring::{
            io_uring_ptr, ioprio_union, IoringCqeFlags, IoringOp,
            IoringSqeFlags, IORING_OFF_CQ_RING, IORING_OFF_SQES,
        },
    };

    #[test]
    fn structs_offsets_entries() {
        assert_eq!(120, mem::size_of::<io_uring_params>());
        assert_eq!(64, mem::size_of::<io_uring_sqe>());
        assert_eq!(16, mem::size_of::<io_uring_cqe>());

        assert_eq!(0, IORING_OFF_SQ_RING);
        assert_eq!(0x8000000, IORING_OFF_CQ_RING);
        assert_eq!(0x10000000, IORING_OFF_SQES);

        assert_eq!(Init::EntriesZero, IoUring::new(0).unwrap_err());
        assert_eq!(Init::EntriesNotPowerOfTwo, IoUring::new(3).unwrap_err());
    }

    // It's just a u16 in the C struct
    fn ioprio_to_u16(i: ioprio_union) -> u16 {
        // 100% safe, 'cause Stone Cold said so
        unsafe { mem::transmute::<_, u16>(i) }
    }

    #[test]
    fn nop() {
        let mut ring = IoUring::new(1).unwrap();
        //let sqe = unsafe { *ring.nop(0xaaaaaaaa).unwrap() };
        let sqe = unsafe {
            let sqe = ring.get_sqe().unwrap();
            prep::nop(sqe, 0xaaaaaaaa);
            *sqe
        };

        //
        // Asserting sqe, field by field, as it lacks `Debug`
        //
        assert_eq!(sqe.opcode, IoringOp::Nop);
        assert_eq!(sqe.flags, IoringSqeFlags::empty());
        assert_eq!(
            ioprio_to_u16(sqe.ioprio),
            ioprio_to_u16(ioprio_union::default())
        );
        assert_eq!(sqe.fd, 0);
        assert_eq!(unsafe { sqe.off_or_addr2.off }, 0);
        assert_eq!(
            unsafe { sqe.addr_or_splice_off_in.addr },
            io_uring_ptr::null()
        );
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
        unsafe {
            assert_eq!(ring.sq.read_tail(&mut ring.mmap), 0);
            assert_eq!(ring.cq.read_head(&mut ring.mmap), 0);
            assert_eq!(ring.sq_ready(), 1);
            assert_eq!(ring.cq_ready(), 0);
        }

        assert_eq!(unsafe { ring.submit() }, Ok(1));
        assert_eq!(ring.sq.sqe_head, 1);
        assert_eq!(ring.sq.sqe_tail, 1);
        assert_eq!(unsafe { ring.sq.read_tail(&mut ring.mmap) }, 1);
        assert_eq!(unsafe { ring.cq.read_head(&mut ring.mmap) }, 0);
        assert_eq!(unsafe { ring.sq_ready() }, 0);

        let cqe = unsafe { ring.copy_cqe().unwrap() };
        assert_eq!(cqe.user_data.u64_(), 0xaaaaaaaa);
        assert_eq!(cqe.res, 0);
        assert_eq!(cqe.flags, IoringCqeFlags::empty());
        assert_eq!(unsafe { ring.cq.read_head(&mut ring.mmap) }, 1);
        assert_eq!(unsafe { ring.cq_ready() }, 0);

        let sqe_barrier = unsafe { ring.get_sqe().unwrap() };
        unsafe { prep::nop(sqe_barrier, 0xbbbbbbbb) };
        unsafe { (*sqe_barrier).flags.set(IoringSqeFlags::IO_DRAIN, true) };
        assert_eq!(unsafe { ring.submit() }, Ok(1));
        let cqe = unsafe { ring.copy_cqe().unwrap() };
        assert_eq!(cqe.user_data.u64_(), 0xbbbbbbbb);
        assert_eq!(cqe.res, 0);
        assert_eq!(cqe.flags, IoringCqeFlags::empty());
        assert_eq!(ring.sq.sqe_head, 2);
        assert_eq!(ring.sq.sqe_tail, 2);
        assert_eq!(unsafe { ring.sq.read_tail(&mut ring.mmap) }, 2);
        assert_eq!(unsafe { ring.cq.read_head(&mut ring.mmap) }, 2);
    }
}
