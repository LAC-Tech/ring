//! IO Uring library, based on zig's std.os.linux.IoUring
//! Based around the 3 syscalls provided by the linux kernel.

use core::sync::atomic;
use core::{assert, assert_eq, assert_ne, cmp, ffi, mem, ptr};
use rustix::fd::{AsRawFd, BorrowedFd, OwnedFd};
use rustix::io;
use rustix::io_uring::{
    io_cqring_offsets, io_sqring_offsets, io_uring_cqe, io_uring_params,
    io_uring_setup, io_uring_sqe, IoringFeatureFlags, IoringSetupFlags,
    IORING_OFF_SQES, IORING_OFF_SQ_RING,
};
use std::os::fd::AsFd;

use rustix::mm;

struct IoUring {
    fd: OwnedFd,
    sq: SubmissionQueue,
    cq: CompletionQueue,
    flags: u32,
    features: u32,
}

impl IoUring {
    /**
     * A friendly way to setup an io_uring, with default
     * rustix::io_uring::io_uring_params.
     * `entries` must be a power of two between 1 and 32768, although the kernel
     * will make the final call on how many entries the submission and
     * completion queues will ultimately have,
     * see https://github.com/torvalds/linux/blob/v5.8/fs/io_uring.c#L8027-L8050
     * Matches the interface of io_uring_queue_init() in liburing.
     */
    pub fn new(entries: u32, flags: IoringSetupFlags) -> Result<Self, InitErr> {
        let mut params = io_uring_params {
            flags,
            sq_thread_idle: 1000,
            ..Default::default()
        };

        Self::new_with_params(entries, &mut params)
    }

    /**
     * A powerful way to setup an io_uring, if you want to tweak
     * rustix::io_uring::io_uring_params such as submission queue thread cpu
     * affinity or thread idle timeout (the kernel and our default is 1 second).
     * `params` is passed by reference because the kernel needs to modify the
     * parameters.
     * Matches the interface of io_uring_queue_init_params() in liburing.
     */
    fn new_with_params(
        entries: u32,
        p: &mut io_uring_params,
    ) -> Result<Self, InitErr> {
        if entries == 0 {
            return Err(InitErr::EntriesZero);
        }
        if !entries.is_power_of_two() {
            return Err(InitErr::EntriesNotPowerOfTwo);
        }

        assert_eq!(p.sq_entries, 0);
        assert!(
            p.cq_entries == 0 || p.flags.contains(IoringSetupFlags::CQSIZE)
        );
        assert!(p.features.is_empty());
        assert!(p.wq_fd == 0 || p.flags.contains(IoringSetupFlags::ATTACH_WQ));
        assert_eq!(p.resv, [0, 0, 0]);

        use io::Errno;

        let fd = io_uring_setup(entries, p).map_err(|errno| match errno {
            Errno::FAULT => InitErr::ParamsOutsideAccessibleAddressSpace,
            // The resv array contains non-zero data, p.flags contains an
            // unsupported flag, entries out of bounds, IORING_SETUP_SQ_AFF was
            // specified without IORING_SETUP_SQPOLL, or IORING_SETUP_CQSIZE was
            // specified but linux.io_uring_params.cq_entries was invalid:
            Errno::INVAL => InitErr::ArgumentsInvalid,
            Errno::MFILE => InitErr::ProcessFdQuotaExceeded,
            Errno::NFILE => InitErr::SystemFdQuotaExceeded,
            Errno::NOMEM => InitErr::SystemResources,
            // IORING_SETUP_SQPOLL was specified but effective user ID lacks
            // sufficient privileges, or a container seccomp policy prohibits
            // io_uring syscalls:
            Errno::PERM => InitErr::PermissionDenied,
            Errno::NOSYS => InitErr::SystemOutdated,
            _ => InitErr::UnexpectedErrno(errno),
        })?;

        assert!(fd.as_raw_fd() >= 0); // Extra paranoid sanity check

        // Kernel versions 5.4 and up use only one mmap() for the submission
        // and completion queues. This is not an optional feature for us... if
        // the kernel does it, we have to do it.
        // The thinking on this by the kernel developers was that both the
        // submission and the completion queue rings have sizes just over a
        // power of two, but the submission queue ring is significantly smaller
        // with u32 slots. By bundling both in a single mmap, the kernel gets
        // the submission queue ring for free.
        // See https://patchwork.kernel.org/patch/11115257 for the kernel patch.
        // We do not support the double mmap() done before 5.4, because we want
        // to keep the init/deinit mmap paths simple and because io_uring has
        // had many bug fixes even since 5.4.
        if !p.features.contains(IoringFeatureFlags::SINGLE_MMAP) {
            return Err(InitErr::SystemOutdated);
        }

        // Check that the kernel has actually set params and that "impossible is
        // nothing".
        assert_ne!(p.sq_entries, 0);
        assert_ne!(p.cq_entries, 0);
        assert!(p.cq_entries >= p.sq_entries);

        let sq = SubmissionQueue::new(fd.as_fd(), *p)
            .map_err(InitErr::UnexpectedErrno)?;
        let cq = CompletionQueue::new(fd.as_fd(), *p, sq);

        // Check that our starting state is as we expect.
        assert_eq!(sq.offsets.head, 0);
        assert_eq!(sq.offsets.tail, 0);
        assert_eq!(sq.offsets.ring_mask, p.sq_entries - 1);
        // Allow flags.* to be non-zero, since the kernel may set
        // IORING_SQ_NEED_WAKEUP at any time.
        assert_eq!(sq.offsets.dropped, 0);
        assert_eq!(sq.offsets.array.len, p.sq_entries);
        assert_eq!(sq.sqes.len, p.sq_entries);
        assert_eq!(sq.sqe_head, 0);
        assert_eq!(sq.sqe_tail, 0);

        assert_eq!(cq.offsets.head, 0);
        assert_eq!(cq.offsets.tail, 0);
        assert_eq!(cq.offsets.ring_mask, p.cq_entries - 1);
        assert_eq!(cq.offsets.overflow, 0);

        Ok(Self { fd, sq, cq, flags: p.flags, features: p.features.bits() })
    }
}

enum InitErr {
    EntriesZero,
    EntriesNotPowerOfTwo,
    ParamsOutsideAccessibleAddressSpace,
    ArgumentsInvalid,
    ProcessFdQuotaExceeded,
    SystemFdQuotaExceeded,
    SystemResources,
    PermissionDenied,
    SystemOutdated,
    UnexpectedErrno(io::Errno),
}

struct SubmissionQueue {
    head: *const atomic::AtomicU32,
    tail: *const atomic::AtomicU32,
    mask: u32,
    flags: *const atomic::AtomicU32,
    dropped: *const atomic::AtomicU32,
    array: &[u32],
    sqes: &[io_uring_sqe],
    mmap: Mmap,
    mmap_sqes: Mmap,
}

impl SubmissionQueue {
    fn new(fd: BorrowedFd, p: io_uring_params) -> io::Result<Self> {
        assert!(fd.as_raw_fd() >= 0);
        assert!(p.features.contains(IoringFeatureFlags::SINGLE_MMAP));

        let size = cmp::max(
            p.sq_off.array as usize
                + p.sq_entries as usize * mem::size_of::<u32>(),
            p.cq_off.cqes as usize
                + p.cq_entries as usize * mem::size_of::<io_uring_cqe>(),
        );

        let mmap = Mmap::new(size, fd, IORING_OFF_SQ_RING)?;

        let sizes_sqe = p.sq_entries as usize * mem::size_of::<io_uring_sqe>();
        let mmap_sqes = Mmap::new(sizes_sqe, fd, IORING_OFF_SQES)?;

        let offsets = unsafe {
            io_sqring_offsets {
                head: *(mmap.ptr.add(p.sq_off.head as usize) as *const u32),
                tail: *(mmap.ptr.add(p.sq_off.tail as usize) as *const u32),
                ring_mask: *(mmap.ptr.add(p.sq_off.ring_mask as usize)
                    as *const u32),
                ring_entries: *(mmap.ptr.add(p.sq_off.ring_entries as usize)
                    as *const u32),
                flags: *(mmap.ptr.add(p.sq_off.flags as usize) as *const u32),
                dropped: *(mmap.ptr.add(p.sq_off.dropped as usize)
                    as *const u32),
                array: *(mmap.ptr.add(p.sq_off.array as usize) as *const u32),
                // Reserved fields are not set by the kernel
                resv1: 0,
                resv2: 0,
            }
        };

        Ok(Self { offsets, mmap, mmap_sqes })
    }
}

struct CompletionQueue {
    offsets: io_cqring_offsets,
}

impl CompletionQueue {
    fn new(fd: BorrowedFd, p: io_uring_params, sq: SubmissionQueue) -> Self {
        assert!(fd.as_raw_fd() >= 0);
        assert!(p.features.contains(IoringFeatureFlags::SINGLE_MMAP));

        let mmap = sq.mmap;

        let offsets = unsafe {
            io_cqring_offsets {
                head: *(mmap.ptr.add(p.cq_off.head as usize) as *const u32),
                tail: *(mmap.ptr.add(p.cq_off.tail as usize) as *const u32),
                ring_mask: *(mmap.ptr.add(p.cq_off.ring_mask as usize)
                    as *const u32),
                ring_entries: *(mmap.ptr.add(p.cq_off.ring_entries as usize)
                    as *const u32),
                overflow: *(mmap.ptr.add(p.cq_off.overflow as usize)
                    as *const u32),
                cqes: *(mmap.ptr.add(p.cq_off.cqes as usize) as *const u32),
                flags: *(mmap.ptr.add(p.cq_off.flags as usize) as *const u32),
                // Reserved fields are not set by the kernel
                resv1: 0,
                resv2: 0,
            }
        };

        assert!(p.cq_entries == offsets.ring_entries);

        Self { offsets }
    }
}

struct Mmap {
    ptr: *mut ffi::c_void,
    len: usize,
}

impl Mmap {
    fn new(
        size: usize,
        fd: BorrowedFd,
        offset: u64,
    ) -> rustix::io::Result<Self> {
        let ptr = unsafe {
            mm::mmap(
                ptr::null_mut(),
                size,
                mm::ProtFlags::READ | mm::ProtFlags::WRITE,
                mm::MapFlags::SHARED | mm::MapFlags::POPULATE,
                fd,
                offset,
            )?
        };

        Ok(Self { ptr, len: size })
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        unsafe {
            // If we fail while unmapping memory I don't know what to do.
            mm::munmap(self.ptr, self.len).unwrap();
        }
    }
}
