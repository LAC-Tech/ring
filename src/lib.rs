#![cfg_attr(not(test), no_std)]

use core::sync::atomic::{AtomicU32, Ordering};
use core::{assert, assert_eq, assert_ne, cmp, ffi, mem, ptr};
use rustix::fd::{BorrowedFd, OwnedFd};
use rustix::io;
use rustix::io_uring::{
    io_uring_cqe, io_uring_enter, io_uring_params, io_uring_setup,
    io_uring_sqe, IoringEnterFlags, IoringFeatureFlags, IoringOp,
    IoringSetupFlags, IORING_OFF_SQES, IORING_OFF_SQ_RING,
};
use std::os::fd::AsFd;

use rustix::mm;

pub struct IoUring {
    fd: OwnedFd,
    sq: SubmissionQueue,
    cq: CompletionQueue,
    _mmap: RwMmap,
    _mmap_sqes: RwMmap,
    pub flags: IoringSetupFlags,
    pub features: IoringFeatureFlags,
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
        let mut params = io_uring_params::default();
        params.flags = flags;
        params.sq_thread_idle = 1000;

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

        use io::Errno;

        let fd =
            unsafe { io_uring_setup(entries, p) }.map_err(
                |errno| match errno {
                    Errno::FAULT => {
                        InitErr::ParamsOutsideAccessibleAddressSpace
                    }
                    Errno::INVAL => InitErr::ArgumentsInvalid,
                    Errno::MFILE => InitErr::ProcessFdQuotaExceeded,
                    Errno::NFILE => InitErr::SystemFdQuotaExceeded,
                    Errno::NOMEM => InitErr::SystemResources,
                    Errno::PERM => InitErr::PermissionDenied,
                    Errno::NOSYS => InitErr::SystemOutdated,
                    _ => InitErr::UnexpectedErrno(errno),
                },
            )?;

        if !p.features.contains(IoringFeatureFlags::SINGLE_MMAP) {
            return Err(InitErr::SystemOutdated);
        }

        assert_ne!(p.sq_entries, 0);
        assert_ne!(p.cq_entries, 0);
        assert!(p.cq_entries >= p.sq_entries);

        let size = cmp::max(
            p.sq_off.array as usize
                + p.sq_entries as usize * mem::size_of::<u32>(),
            p.cq_off.cqes as usize
                + p.cq_entries as usize * mem::size_of::<io_uring_cqe>(),
        );

        let mmap = RwMmap::new(
            size,
            fd.as_fd(),
            mm::MapFlags::SHARED | mm::MapFlags::POPULATE,
            IORING_OFF_SQ_RING,
        )
        .map_err(InitErr::UnexpectedErrno)?;

        let sizes_sqe = p.sq_entries as usize * mem::size_of::<io_uring_sqe>();
        let mmap_sqes = RwMmap::new(
            sizes_sqe,
            fd.as_fd(),
            mm::MapFlags::SHARED | mm::MapFlags::POPULATE,
            IORING_OFF_SQES,
        )
        .map_err(InitErr::UnexpectedErrno)?;

        let sq = unsafe { SubmissionQueue::new(mmap.ptr, mmap_sqes.ptr, p) };
        let cq = unsafe { CompletionQueue::new(mmap.ptr, p) };

        unsafe {
            assert_eq!((*sq.head).load(Ordering::Acquire), 0);
            assert_eq!((*sq.tail).load(Ordering::Acquire), 0);
            assert_eq!(sq.mask, p.sq_entries - 1);
            assert_eq!((*sq.dropped).load(Ordering::Acquire), 0);

            assert_eq!((*cq.head).load(Ordering::Acquire), 0);
            assert_eq!((*cq.tail).load(Ordering::Acquire), 0);
            assert_eq!(cq.mask, p.cq_entries - 1);
            assert_eq!((*cq.overflow).load(Ordering::Acquire), 0);
        }

        Ok(Self {
            fd,
            sq,
            cq,
            _mmap: mmap,
            _mmap_sqes: mmap_sqes,
            flags: p.flags,
            features: p.features,
        })
    }

    pub fn get_sqe(&mut self) -> Option<&mut io_uring_sqe> {
        let head = unsafe { (*self.sq.head).load(Ordering::Acquire) };
        let next = self.sq.sqe_tail.wrapping_add(1);
        if next.wrapping_sub(head) > self.sq.mask + 1 {
            return None;
        }
        let sqe = unsafe {
            let s = &mut *self
                .sq
                .sqes
                .add((self.sq.sqe_tail & self.sq.mask) as usize);
            ptr::write_bytes(s, 0, 1);
            s
        };
        self.sq.sqe_tail = next;
        Some(sqe)
    }

    pub fn submit(&mut self) -> io::Result<u32> {
        self.submit_and_wait(0)
    }

    pub fn submit_and_wait(&mut self, wait_nr: u32) -> io::Result<u32> {
        let sq = &mut self.sq;
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

pub struct SubmissionQueue {
    head: *const AtomicU32,
    tail: *const AtomicU32,
    mask: u32,
    _flags: *const AtomicU32,
    dropped: *const AtomicU32,
    array: *mut u32,
    sqes: *mut io_uring_sqe,
    sqe_head: u32,
    sqe_tail: u32,
}

impl SubmissionQueue {
    unsafe fn new(
        ptr: *mut ffi::c_void,
        ptr_sqes: *mut ffi::c_void,
        p: &io_uring_params,
    ) -> Self {
        Self {
            head: ptr.add(p.sq_off.head as usize) as *const AtomicU32,
            tail: ptr.add(p.sq_off.tail as usize) as *const AtomicU32,
            mask: *(ptr.add(p.sq_off.ring_mask as usize) as *const u32),
            _flags: ptr.add(p.sq_off.flags as usize) as *const AtomicU32,
            dropped: ptr.add(p.sq_off.dropped as usize) as *const AtomicU32,
            array: ptr.add(p.sq_off.array as usize) as *mut u32,
            sqes: ptr_sqes as *mut io_uring_sqe,
            sqe_head: 0,
            sqe_tail: 0,
        }
    }
}

pub struct CompletionQueue {
    head: *const AtomicU32,
    tail: *const AtomicU32,
    mask: u32,
    overflow: *const AtomicU32,
    cqes: *mut io_uring_cqe,
    _flags: *const AtomicU32,
}

impl CompletionQueue {
    unsafe fn new(ptr: *mut ffi::c_void, p: &io_uring_params) -> Self {
        Self {
            head: ptr.add(p.cq_off.head as usize) as *const AtomicU32,
            tail: ptr.add(p.cq_off.tail as usize) as *const AtomicU32,
            mask: *(ptr.add(p.cq_off.ring_mask as usize) as *const u32),
            overflow: ptr.add(p.cq_off.overflow as usize) as *const AtomicU32,
            cqes: ptr.add(p.cq_off.cqes as usize) as *mut io_uring_cqe,
            _flags: ptr.add(p.cq_off.flags as usize) as *const AtomicU32,
        }
    }
}

struct RwMmap {
    ptr: *mut ffi::c_void,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        let _uring =
            IoUring::new(8, IoringSetupFlags::empty()).expect("setup failed");
    }

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
}
