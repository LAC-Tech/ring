use core::cmp;
use core::ffi::c_void;
use core::mem;
use core::ops::{Index, IndexMut};
use core::ptr;
use core::sync::atomic::{AtomicU32, Ordering};
use rustix::fd::BorrowedFd;
use rustix::io_uring::{
    io_cqring_offsets, io_sqring_offsets, io_uring_params, io_uring_sqe,
    io_uring_user_data, IoringCqeFlags, IoringSqFlags, IORING_OFF_CQ_RING,
    IORING_OFF_SQES, IORING_OFF_SQ_RING,
};
use rustix::{io, mm};
use std::os::fd::AsFd;

// Sanity check that we can cast from u32 to usize
const _: () = assert!(usize::BITS >= 32);

// I am constantly using these - makes it a little bit cleaner
const U32_SIZE: u32 = mem::size_of::<u32>() as u32;
const SQE_SIZE: u32 = mem::size_of::<io_uring_sqe>() as u32;
const CQE_SIZE: u32 = mem::size_of::<Cqe>() as u32;

const _: () = assert!(U32_SIZE == 4); // rofl why not

// Taken from the zig test "structs/offsets/entries"
// Because this is a 3rd party lib we can be more aggressive about preventing
// compilation.
const _: () = {
    assert!(mem::size_of::<io_uring_params>() == 120);
    assert!(SQE_SIZE == 64);
    assert!(CQE_SIZE == 16);

    assert!(IORING_OFF_SQ_RING == 0);
    assert!(IORING_OFF_CQ_RING == 0x8000000);
    assert!(IORING_OFF_SQES == 0x10000000);
};

/// An io_uring Completion Queue Entry.
///
/// While rustix provides one, it has some issues:
/// <https://github.com/bytecodealliance/rustix/issues/1568>
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct Cqe {
    pub user_data: io_uring_user_data,
    pub res: i32,
    pub flags: IoringCqeFlags,
}

// Convenience struct so mmap's are un-mapped on drop
#[derive(Debug)]
struct Mmap {
    ptr: *mut c_void,
    len: usize,
}

impl Mmap {
    fn new(
        size: usize,
        fd: BorrowedFd,
        flags: mm::MapFlags,
        offset: u64,
    ) -> rustix::io::Result<Self> {
        let prot_flags = mm::ProtFlags::READ | mm::ProtFlags::WRITE;
        // SAFETY: Fd cannot be dropped because it's borrowed
        let ptr = unsafe {
            mm::mmap(ptr::null_mut(), size, prot_flags, flags, fd, offset)?
        };

        Ok(Self { ptr, len: size })
    }

    fn check_bounds(&self, offset: u32, size: usize) {
        let end =
            (offset as usize).checked_add(size).expect("mmap offset overflow");
        assert!(
            end <= self.len,
            "mmap access out of bounds: offset {} size {} len {}",
            offset,
            size,
            self.len
        );
    }

    /// # Safety
    ///
    /// The caller must ensure that the memory at `byte_offset` is valid for
    /// mutation as type `T` and follows Rust's aliasing rules.
    unsafe fn mut_ptr_at<T>(&mut self, byte_offset: u32) -> *mut T {
        self.check_bounds(byte_offset, mem::size_of::<T>());
        self.ptr.cast::<u8>().add(byte_offset as usize) as *mut T
    }

    /// # Safety
    ///
    /// The caller must ensure that the memory at `byte_offset` is a valid
    /// representation of type `T`.
    unsafe fn ptr_at<T>(&self, byte_offset: u32) -> *const T {
        self.check_bounds(byte_offset, mem::size_of::<T>());
        (self.ptr as *const u8).add(byte_offset as usize) as *const T
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        // SAFETY: We own the pointer and the length is correct.
        unsafe {
            // If we fail while unmapping memory I don't know what to do.
            mm::munmap(self.ptr, self.len).expect("munmap failed");
        }
    }
}

#[derive(Debug)]
pub struct Sqes(Mmap);

impl Sqes {
    pub fn new(
        io_ring_fd: BorrowedFd,
        p: &io_uring_params,
    ) -> io::Result<Self> {
        let size_sqes = (p.sq_entries * SQE_SIZE) as usize;

        let mmap = Mmap::new(
            size_sqes,
            io_ring_fd.as_fd(),
            mm::MapFlags::SHARED | mm::MapFlags::POPULATE,
            IORING_OFF_SQES,
        )?;

        Ok(Self(mmap))
    }
}

impl Index<u32> for Sqes {
    type Output = io_uring_sqe;

    fn index(&self, idx: u32) -> &Self::Output {
        let byte_offset = idx * SQE_SIZE;
        // SAFETY: ptr_at checks bounds, T is Copy so valid to read
        unsafe { &*self.0.ptr_at::<Self::Output>(byte_offset) }
    }
}

impl IndexMut<u32> for Sqes {
    fn index_mut(&mut self, idx: u32) -> &mut Self::Output {
        let byte_offset = idx * SQE_SIZE;
        // SAFETY: mut_ptr_at checks bounds, T is Copy so valid to write
        unsafe { &mut *self.0.mut_ptr_at::<Self::Output>(byte_offset) }
    }
}

#[derive(Debug)]
pub struct Ioring {
    mmap: Mmap,
    sq_off: io_sqring_offsets,
    cq_off: io_cqring_offsets,
    sq_entries: u32,
    cq_entries: u32,
}

impl Ioring {
    pub fn new(
        ring_fd: BorrowedFd,
        p: &io_uring_params,
    ) -> rustix::io::Result<Self> {
        let io_uring_params {
            sq_entries,
            cq_entries,
            flags,
            sq_thread_cpu,
            sq_thread_idle,
            features,
            wq_fd,
            resv,
            sq_off,
            cq_off,
            ..
        } = *p;

        let size = cmp::max(
            sq_off.array + sq_entries * U32_SIZE,
            cq_off.cqes + cq_entries * CQE_SIZE,
        ) as usize;
        let mmap = Mmap::new(
            size,
            ring_fd,
            mm::MapFlags::SHARED | mm::MapFlags::POPULATE,
            IORING_OFF_SQ_RING,
        )?;

        Ok(Self { mmap, sq_off, cq_off, sq_entries, cq_entries })
    }
}

// All our pointer accesses are by u32s..
impl Ioring {
    // man 7 io_uring:
    // - "You add SQEs to the tail of the SQ. The kernel reads SQEs off the head
    //   of the queue."
    pub fn sq_head(&self) -> u32 {
        // SAFETY: The offset is provided by the kernel.
        unsafe { self.atomic_load_u32_at(self.sq_off.head, Ordering::Acquire) }
    }

    pub fn sq_flag_contains(&self, flag: IoringSqFlags) -> bool {
        // SAFETY: The offset is provided by the kernel.
        let bits = unsafe {
            self.atomic_load_u32_at(self.sq_off.flags, Ordering::Relaxed)
        };
        let flags = IoringSqFlags::from_bits_retain(bits);
        flags.contains(flag)
    }

    // This method helped me catch a bug. I do not care if it is shallow.
    pub fn sq_tail(&self) -> u32 {
        // SAFETY: The offset is provided by the kernel.
        unsafe { self.u32_at(self.sq_off.tail) }
    }

    pub fn sq_tail_write(&mut self, new_tail: u32) {
        // SAFETY: The offset is provided by the kernel.
        let tail = unsafe { self.atomic_u32_at(self.sq_off.tail) };
        tail.store(new_tail, Ordering::Release);
    }

    pub fn sqe_indices(&mut self) -> &mut [u32] {
        // SAFETY: Offset and length provided by the kernel
        unsafe { self.mut_slice_at::<u32>(self.sq_off.array, self.sq_entries) }
    }

    pub fn cqes(&self) -> &[Cqe] {
        // SAFETY: Offset and length provided by the kernel
        unsafe { self.slice_at::<Cqe>(self.cq_off.cqes, self.cq_entries) }
    }

    /// # Safety
    ///
    /// The caller must ensure that the memory at `byte_offset` contains a valid
    /// u32.
    unsafe fn u32_at(&self, byte_offset: u32) -> u32 {
        *self.mmap.ptr_at(byte_offset)
    }

    /// # Safety
    ///
    /// The caller must ensure that the memory at `byte_offset` is valid for
    /// atomic access as a u32.
    unsafe fn atomic_load_u32_at(
        &self,
        byte_offset: u32,
        ordering: Ordering,
    ) -> u32 {
        let ptr = self.mmap.ptr_at::<u32>(byte_offset) as *mut u32;
        AtomicU32::from_ptr(ptr).load(ordering)
    }

    /// Use this when you need to do atomic writes
    ///
    /// # Safety
    ///
    /// The caller must ensure that the memory at `byte_offset` is valid for
    /// atomic access as a u32.
    unsafe fn atomic_u32_at(&mut self, byte_offset: u32) -> &AtomicU32 {
        AtomicU32::from_ptr(self.0.mut_ptr_at(byte_offset))
    }

    unsafe fn raw_slice_at<T>(&self, byte_offset: u32, len: u32) -> *mut [T] {
        self.0.check_bounds(byte_offset, (len as usize) * mem::size_of::<T>());
        let ptr = (self.0.ptr as *mut u8).add(byte_offset as usize) as *mut T;
        ptr::slice_from_raw_parts_mut(ptr, len as usize)
    }

    /// # Safety
    ///
    /// The caller must ensure that the memory starting at `byte_offset` is a
    /// valid representation of `len` elements of type `T`.
    unsafe fn slice_at<T>(&self, byte_offset: u32, len: u32) -> &[T] {
        &*self.raw_slice_at(byte_offset, len)
    }

    /// # Safety
    ///
    /// The caller must ensure that the memory starting at `byte_offset` is a
    /// valid representation of `len` elements of type `T`.
    unsafe fn mut_slice_at<T>(
        &mut self,
        byte_offset: u32,
        len: u32,
    ) -> &mut [T] {
        &mut *self.raw_slice_at(byte_offset, len)
    }
}
