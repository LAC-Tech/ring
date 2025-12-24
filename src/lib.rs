// Ported from Zig's standard library io_uring implementation
// Original: https://codeberg.org/ziglang/zig/src/branch/master/lib/std/os/linux/IoUring.zig
// Licensed under MIT License (see https://github.com/ziglang/zig/blob/master/LICENSE)
#![cfg_attr(not(test), no_std)]

// I need to CONSTATLY cast things between usize and u32
const _: () = assert!(usize::BITS >= 32);

use core::ffi::c_void;
use core::sync::atomic::{AtomicU32, Ordering};
use core::{assert, assert_eq, assert_ne, cmp, mem, ptr};
use rustix::fd::{AsFd, BorrowedFd, OwnedFd};
use rustix::io;
use rustix::io_uring::{
    io_cqring_offsets, io_sqring_offsets, io_uring_cqe, io_uring_enter,
    io_uring_params, io_uring_setup, io_uring_sqe, IoringEnterFlags,
    IoringFeatureFlags, IoringSetupFlags, IoringSqFlags, IORING_OFF_SQES,
    IORING_OFF_SQ_RING,
};

use rustix::mm;

pub struct IoUring {
    pub fd: OwnedFd,
    // A single mmap call that contains both the sq and cq
    mmap_rings: RwMmap,
    // Contains the array with the actual sq entries
    mmap_sq_entries: RwMmap,
    pub flags: IoringSetupFlags,
    pub features: IoringFeatureFlags,
    pub sq_off: io_sqring_offsets,
    pub cq_off: io_cqring_offsets,
    sqe_head: u32,
    sqe_tail: u32,
    sq_mask: u32,
}

impl IoUring {
    /// A friendly way to setup an io_uring, with default linux.io_uring_params.
    /// `entries` must be a power of two between 1 and 32768, although the
    /// kernel will make the final call on how many entries the submission
    /// and completion queues will ultimately have, see https://github.com/torvalds/linux/blob/v5.8/fs/io_uring.c#L8027-L8050.
    /// Matches the interface of io_uring_queue_init() in liburing.
    pub fn new(
        entries: u32,
        flags: IoringSetupFlags,
    ) -> Result<Self, err::Init> {
        let mut params = io_uring_params::default();
        params.flags = flags;
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

        let mmap_rings = RwMmap::new(
            size,
            fd.as_fd(),
            mm::MapFlags::SHARED | mm::MapFlags::POPULATE,
            IORING_OFF_SQ_RING,
        )
        .map_err(UnexpectedErrno)?;
        // TODO: usless assert?
        assert_eq!(mmap_rings.len, size);

        // TODO: wtf does this mean
        // "The motivation for the `sqes` and `array` indirection is to make it
        // possible for the
        // application to preallocate static linux.io_uring_sqe entries and
        // then replay them when needed."
        let size_sqes =
            (p.sq_entries * mem::size_of::<io_uring_sqe>() as u32) as usize;

        let mmap_sq_entries = RwMmap::new(
            size_sqes.try_into().unwrap(),
            fd.as_fd(),
            mm::MapFlags::SHARED | mm::MapFlags::POPULATE,
            IORING_OFF_SQES,
        )
        .map_err(UnexpectedErrno)?;
        // TODO: usless assert?
        assert_eq!(mmap_sq_entries.len, size_sqes);

        let sq_mask: u32 = unsafe { *mmap_rings.ptr_at(p.cq_off.ring_mask) };

        // We expect the kernel copies p.sq_entries to the u32 pointed to by
        // p.sq_off.ring_entries, see https://github.com/torvalds/linux/blob/v5.8/fs/io_uring.c#L7843-L7844.
        unsafe {
            assert_eq!(p.sq_entries, *mmap_rings.ptr_at(p.sq_off.ring_entries))
        };

        // Check that our starting state is as we expect.
        // TODO: make these little private functions if they are used more than
        // once
        unsafe {
            assert_eq!(*mmap_rings.ptr_at::<u32>(p.sq_off.head), 0);
            assert_eq!(*mmap_rings.ptr_at::<u32>(p.sq_off.tail), 0);
            assert_eq!(
                *mmap_rings.ptr_at::<u32>(p.sq_off.ring_mask),
                p.sq_entries - 1,
            );
            // Allow p.sq_off.flags to be non-zero, since the kernel may set
            // IORING_SQ_NEED_WAKEUP at any time.
            assert_eq!(*mmap_rings.ptr_at::<u32>(p.sq_off.dropped), 0);
        }
        unsafe {
            assert_eq!(*mmap_rings.ptr_at::<u32>(p.cq_off.head), 0);
            assert_eq!(*mmap_rings.ptr_at::<u32>(p.cq_off.tail), 0);
            assert_eq!(
                *mmap_rings.ptr_at::<u32>(p.cq_off.ring_mask),
                p.cq_entries - 1
            );
            assert_eq!(*mmap_rings.ptr_at::<u32>(p.cq_off.overflow), 0);
        }

        Ok(Self {
            fd,
            mmap_rings,
            mmap_sq_entries,
            flags: p.flags,
            features: p.features,
            sq_off: p.sq_off,
            cq_off: p.cq_off,
            sqe_head: 0,
            sqe_tail: 0,
            sq_mask,
        })
    }

    unsafe fn shared_sq_head(&mut self) -> u32 {
        self.mmap_rings.atomic_u32_at(self.sq_off.head).load(Ordering::Acquire)
    }

    unsafe fn shared_sq_flags(&mut self) -> IoringSqFlags {
        let atomic_u32 = self
            .mmap_rings
            .atomic_u32_at(self.sq_off.flags)
            .load(Ordering::Relaxed);

        IoringSqFlags::from_bits_retain(atomic_u32)
    }

    /// Returns a reference to a vacant SQE, or an error if the submission
    /// queue is full. We follow the implementation (and atomics) of
    /// liburing's `io_uring_get_sqe()` exactly.
    /// TODO: is it reasonable/possible to make this "safe" and return a
    /// reference?
    pub unsafe fn get_sqe(&mut self) -> Result<*mut io_uring_sqe, GetSqeErr> {
        let head = unsafe { self.shared_sq_head() };

        // Remember that these head and tail offsets wrap around every four
        // billion operations. We must therefore use wrapping addition
        // and subtraction to avoid a runtime crash.
        let next = self.sqe_head.wrapping_add(1);
        if next.wrapping_sub(head) > self.sq_off.ring_entries {
            return Err(GetSqeErr::SubmissionQueueFull);
        }

        let sqe = self.mmap_sq_entries.mut_ptr_at(self.sqe_tail & self.sq_mask);

        self.sqe_tail = next;
        Ok(sqe)
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
        if self.sqe_head != self.sqe_tail {
            // Fill in SQEs that we have queued up, adding them to the kernel
            // ring.
            let to_submit = self.sqe_tail.wrapping_sub(self.sqe_head);
            let tail = self.mmap_rings.mut_ptr_at::<u32>(self.sq_off.tail);

            for _ in 0..to_submit {
                let sqe: *mut u32 = self
                    .mmap_rings
                    .mut_ptr_at(self.sq_off.array + (*tail & self.sq_mask));

                *tail = *tail.wrapping_add(1);
                *sqe = self.sqe_head & self.sq_mask;
            }

            // Ensure that the kernel can actually see the SQE updates when it
            // sees the tail update.
            self.mmap_rings
                .atomic_u32_at(self.sq_off.tail)
                .store(*tail, Ordering::Release);
        }

        self.sq_ready()
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

        if self.shared_sq_flags().contains(IoringSqFlags::NEED_WAKEUP) {
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
        // Always use the shared ring state (i.e. not self.sqe_head) to
        // avoid going out of sync, see https://github.com/axboe/liburing/issues/92.
        self.sqe_tail.wrapping_sub(self.shared_sq_head())
    }
}

pub enum GetSqeErr {
    SubmissionQueueFull,
}

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

    unsafe fn mut_ptr_at<T>(&mut self, byte_offset: u32) -> *mut T {
        (self.ptr as *mut u8).add(byte_offset as usize) as *mut T
    }

    unsafe fn atomic_u32_at(&mut self, byte_offset: u32) -> &AtomicU32 {
        AtomicU32::from_ptr(self.mut_ptr_at(byte_offset))
    }
}

mod err {
    #[derive(Debug)]
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        let _uring =
            IoUring::new(8, IoringSetupFlags::empty()).expect("setup failed");
    }
}
