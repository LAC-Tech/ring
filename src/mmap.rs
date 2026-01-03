use core::ffi::c_void;
use core::mem;
use core::ptr;
use core::sync::atomic::{AtomicU32, Ordering};
use rustix::fd::BorrowedFd;
use rustix::mm;

#[derive(Debug)]
pub struct RwMmap {
    ptr: *mut c_void,
    len: usize,
}

impl RwMmap {
    pub fn new(
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
}

impl Drop for RwMmap {
    fn drop(&mut self) {
        // SAFETY: We own the pointer and the length is correct.
        unsafe {
            // If we fail while unmapping memory I don't know what to do.
            mm::munmap(self.ptr, self.len).expect("munmap failed");
        }
    }
}

// All our pointer accesses are by u32s..
impl RwMmap {
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

    /// # Safety
    ///
    /// The caller must ensure that the memory at `byte_offset` contains a valid
    /// u32.
    // just avoids a bit of line noise all the times I do this
    pub unsafe fn u32_at(&self, byte_offset: u32) -> u32 {
        *self.ptr_at(byte_offset)
    }

    /// # Safety
    ///
    /// The caller must ensure that the memory at `byte_offset` is valid for
    /// atomic access as a u32.
    // Atomic needs a mutable ptr but we never mutate the value
    // So this method avoids &mut self when we just want to read
    pub unsafe fn atomic_load_u32_at(
        &self,
        byte_offset: u32,
        ordering: Ordering,
    ) -> u32 {
        let ptr = self.ptr_at::<u32>(byte_offset) as *mut u32;
        AtomicU32::from_ptr(ptr).load(ordering)
    }

    /// Use this when you need to do atomic writes
    ///
    /// # Safety
    ///
    /// The caller must ensure that the memory at `byte_offset` is valid for
    /// atomic access as a u32.
    pub unsafe fn atomic_u32_at(&mut self, byte_offset: u32) -> &AtomicU32 {
        AtomicU32::from_ptr(self.mut_ptr_at(byte_offset))
    }

    unsafe fn raw_slice_at<T>(&self, byte_offset: u32, len: u32) -> *mut [T] {
        self.check_bounds(byte_offset, (len as usize) * mem::size_of::<T>());
        let ptr = (self.ptr as *mut u8).add(byte_offset as usize) as *mut T;
        ptr::slice_from_raw_parts_mut(ptr, len as usize)
    }

    /// # Safety
    ///
    /// The caller must ensure that the memory starting at `byte_offset` is a
    /// valid representation of `len` elements of type `T`.
    pub unsafe fn slice_at<T>(&self, byte_offset: u32, len: u32) -> &[T] {
        &*self.raw_slice_at(byte_offset, len)
    }

    /// # Safety
    ///
    /// The caller must ensure that the memory starting at `byte_offset` is a
    /// valid representation of `len` elements of type `T`.
    pub unsafe fn mut_slice_at<T>(
        &mut self,
        byte_offset: u32,
        len: u32,
    ) -> &mut [T] {
        &mut *self.raw_slice_at(byte_offset, len)
    }
}
