#![cfg_attr(not(test), no_std)]

// I need to CONSTATLY cast things between usize and u32
const _: () = assert!(usize::BITS >= 32);

use core::ffi::c_void;
use core::sync::atomic::{AtomicU32, Ordering};
use core::{assert, assert_eq, assert_ne, cmp, ffi, mem, ptr};
use rustix::fd::{AsFd, BorrowedFd, OwnedFd};
use rustix::io_uring::{
    io_cqring_offsets, io_sqring_offsets, io_uring_cqe, io_uring_enter,
    io_uring_params, io_uring_setup, io_uring_sqe, IoringEnterFlags,
    IoringFeatureFlags, IoringOp, IoringSetupFlags, IORING_OFF_SQES,
    IORING_OFF_SQ_RING,
};
use rustix::{io, io_uring};

use rustix::mm;

pub struct IoUring {
    fd: OwnedFd,
    // A single mmap call that contains both the sq and cq
    mmap_rings: RwMmap,
    // Contains the array with the actual sq entries
    mmap_sq_entries: RwMmap,
    pub flags: IoringSetupFlags,
    pub features: IoringFeatureFlags,
    pub sq_off: io_sqring_offsets,
    pub cq_off: io_cqring_offsets,
}

impl IoUring {
    /// A friendly way to setup an io_uring, with default linux.io_uring_params.
    /// `entries` must be a power of two between 1 and 32768, although the
    /// kernel will make the final call on how many entries the submission
    /// and completion queues will ultimately have, see https://github.com/torvalds/linux/blob/v5.8/fs/io_uring.c#L8027-L8050.
    /// Matches the interface of io_uring_queue_init() in liburing.
    pub fn new(entries: u32, flags: IoringSetupFlags) -> Result<Self, InitErr> {
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

        let res = unsafe { io_uring_setup(entries, p) };
        let fd = res.map_err(|errno| match errno {
            Errno::FAULT => InitErr::ParamsOutsideAccessibleAddressSpace,
            Errno::INVAL => InitErr::ArgumentsInvalid,
            Errno::MFILE => InitErr::ProcessFdQuotaExceeded,
            Errno::NFILE => InitErr::SystemFdQuotaExceeded,
            Errno::NOMEM => InitErr::SystemResources,
            Errno::PERM => InitErr::PermissionDenied,
            Errno::NOSYS => InitErr::SystemOutdated,
            _ => InitErr::UnexpectedErrno(errno),
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
            return Err(InitErr::SystemOutdated);
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

        let mmap_sq = RwMmap::new(
            size,
            fd.as_fd(),
            mm::MapFlags::SHARED | mm::MapFlags::POPULATE,
            IORING_OFF_SQ_RING,
        )
        .map_err(InitErr::UnexpectedErrno)?;
        assert_eq!(mmap_sq.len, size);

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
        .map_err(InitErr::UnexpectedErrno)?;
        assert_eq!(mmap_sq_entries.len, size_sqes);

        // Check that our starting state is as we expect.
        // TODO: make these little private functions if they are used more than
        // once
        unsafe {
            assert_eq!(*mmap_sq.ptr_at::<u32>(p.sq_off.head), 0);
            assert_eq!(*mmap_sq.ptr_at::<u32>(p.sq_off.tail), 0);
            assert_eq!(
                *mmap_sq.ptr_at::<u32>(p.sq_off.ring_mask),
                p.sq_entries - 1,
            );
            // Allow p.sq_off.flags to be non-zero, since the kernel may set
            // IORING_SQ_NEED_WAKEUP at any time.
            assert_eq!(*mmap_sq.ptr_at::<u32>(p.sq_off.dropped), 0);
        }
        unsafe {
            assert_eq!(*mmap_sq.ptr_at::<u32>(p.cq_off.head), 0);
            assert_eq!(*mmap_sq.ptr_at::<u32>(p.cq_off.tail), 0);
            assert_eq!(
                *mmap_sq.ptr_at::<u32>(p.cq_off.ring_mask),
                p.cq_entries - 1
            );
            assert_eq!(*mmap_sq.ptr_at::<u32>(p.cq_off.overflow), 0);
        }

        Ok(Self {
            fd,
            mmap_rings: mmap_sq,
            mmap_sq_entries,
            flags: p.flags,
            features: p.features,
            sq_off: p.sq_off,
            cq_off: p.cq_off,
        })
    }

    /*
        pub fn get_sqe(&mut self) -> Option<&mut io_uring_sqe> {
            let head = unsafe { (*self.mmap_rings.head).load(Ordering::Acquire) };
            let next = self.mmap_rings.sqe_tail.wrapping_add(1);
            if next.wrapping_sub(head) > self.mmap_rings.mask + 1 {
                return None;
            }
            let sqe = unsafe {
                let s = &mut *self.mmap_rings.sqes.add(
                    (self.mmap_rings.sqe_tail & self.mmap_rings.mask) as usize,
                );
                ptr::write_bytes(s, 0, 1);
                s
            };
            self.mmap_rings.sqe_tail = next;
            Some(sqe)
        }

        pub fn submit(&mut self) -> io::Result<u32> {
            self.submit_and_wait(0)
        }

        pub fn submit_and_wait(&mut self, wait_nr: u32) -> io::Result<u32> {
            let sq = &mut self.mmap_rings;
            let to_submit = sq.sqe_tail.wrapping_sub(sq.sqe_head);
            if to_submit > 0 {
                let mut tail = unsafe { (*sq.tail).load(Ordering::Acquire) };
                while sq.sqe_head != sq.sqe_tail {
                    unsafe {
                        *sq.array.add((tail & sq.mask) as usize) =
                            sq.sqe_head & sq.mask;
                    }
                    sq.sqe_head = sq.sqe_head.wrapping_add(1);
                    tail = tail.wrapping_add(1);
                }
                unsafe {
                    (*sq.tail).store(tail, Ordering::Release);
                }
            }

            let mut flags = IoringEnterFlags::empty();
            if wait_nr > 0 {
                flags |= IoringEnterFlags::GETEVENTS;
            }

            unsafe { io_uring_enter(self.fd.as_fd(), to_submit, wait_nr, flags) }
        }

        pub fn peek_cqe(&self) -> Option<&io_uring_cqe> {
            let cq = &self.cq;
            let head = unsafe { (*cq.head).load(Ordering::Acquire) };
            if head == unsafe { (*cq.tail).load(Ordering::Acquire) } {
                return None;
            }
            Some(unsafe { &*cq.cqes.add((head & cq.mask) as usize) })
        }

        pub fn cq_advance(&self, nr: u32) {
            let cq = &self.cq;
            unsafe {
                let head = (*cq.head).load(Ordering::Relaxed);
                (*cq.head).store(head.wrapping_add(nr), Ordering::Release);
            }
        }

        pub fn copy_cqes(
            &mut self,
            cqes: &mut [io_uring_cqe],
            wait_nr: u32,
        ) -> io::Result<u32> {
            let mut count: u32 = 0;
            while (count as usize) < cqes.len() {
                if let Some(cqe) = self.peek_cqe() {
                    cqes[count as usize] = unsafe { ptr::read(cqe as *const _) };
                    count += 1;
                    self.cq_advance(1);
                } else {
                    if count >= wait_nr {
                        break;
                    }
                    self.submit_and_wait(wait_nr - count)?;
                }
            }
            Ok(count)
        }
    */
}

#[derive(Debug)]
pub enum InitErr {
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
        let ptr = unsafe {
            mm::mmap(
                ptr::null_mut(),
                size,
                mm::ProtFlags::READ | mm::ProtFlags::WRITE,
                flags,
                fd,
                offset,
            )?
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

impl RwMmap {
    // All our pointer accesses are by u32s
    unsafe fn ptr_at<T>(&self, byte_offset: u32) -> *const T {
        (self.ptr as *const u8).add(byte_offset as usize) as *const T
    }

    unsafe fn slice_at<T>(&self, byte_offset: u32, count: usize) -> &[T] {
        let ptr = self.ptr_at::<T>(byte_offset);
        &*core::ptr::slice_from_raw_parts(ptr, count)
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

    /*
    #[test]
    fn test_nop() {
        let mut uring =
            IoUring::new(8, IoringSetupFlags::empty()).expect("setup failed");

        let sqe = uring.get_sqe().expect("get sqe failed");
        sqe.opcode = IoringOp::Nop;
        sqe.user_data = 0x42.into();

        uring.submit_and_wait(1).expect("submit failed");

        let mut cqes = [io_uring_cqe::default()];
        let count = uring.copy_cqes(&mut cqes, 1).expect("copy cqes failed");

        assert_eq!(count, 1);
        assert_eq!(cqes[0].user_data, 0x42.into());
    }
    */
}
