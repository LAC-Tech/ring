use core::ffi::*;
use core::ptr::*;
use core::mem;
//use rustix::fd::*;
use rustix::fs::*;
use rustix::io::ReadWriteFlags;
use rustix::io_uring::IoringOp::*;
use rustix::io_uring::*;

use crate::{Fd, ReadBuf, Sqe};

// This trait let us have a more declarative API.
// Note these are done rather differently than IoUring.zig - specifically they
// zero out more fields.
//
// Jens Axboe:
// "one reason for that is that fields that aren't used could be used in the
// future, and ensuring they are zero and  having the kernel verify they are
// zero makes it so you can actually use one  of those fields in the future for
// functionality, flags, etc"

pub trait Prep {
    fn prep(&self, sqe: &mut Sqe);
}

pub struct Nop {
    user_data: u64,
}

impl Prep for Nop {
    fn prep(&self, sqe: &mut Sqe) {
        sqe.opcode = Nop;
        sqe.fd = -1;
        sqe.clear_buf();
        sqe.user_data.u64_ = self.user_data;
    }
}

pub fn nop(user_data: u64) -> Nop {
    Nop { user_data }
}

pub struct Fsync<'a> {
    user_data: u64,
    fd: &'a Fd,
    flags: ReadWriteFlags,
}

impl<'a> Prep for Fsync<'a> {
    fn prep(&self, sqe: &mut Sqe) {
        sqe.opcode = Fsync;
        sqe.fd = self.fd.as_raw_fd();
        sqe.clear_buf();
        sqe.op_flags.rw_flags = self.flags;
        sqe.user_data.u64_ = self.user_data;
    }
}

pub fn fsync<'a>(
    user_data: u64,
    fd: &'a Fd,
    flags: ReadWriteFlags,
) -> Fsync<'a> {
    Fsync { user_data, fd: fd, flags }
}

pub struct Read<'a, const BUF_LEN: usize, RB: ReadBuf<BUF_LEN>> {
    user_data: u64,
    fd: &'a Fd,
    buf: &'a RB,
    offset: u64,
}

impl<'a, const BUF_LEN: usize, RB: ReadBuf<BUF_LEN>> Prep
    for Read<'a, BUF_LEN, RB>
{
    fn prep(&self, sqe: &mut Sqe) {
        sqe.opcode = Read;
        sqe.fd = self.fd.as_raw_fd();
        sqe.set_buf(self.buf.as_ptr(), BUF_LEN, self.offset);
        sqe.user_data.u64_ = self.user_data;
    }
}

pub fn read<'a, const BUF_LEN: usize, RB: ReadBuf<BUF_LEN>>(
    user_data: u64,
    fd: &'a Fd,
    buf: &'a RB,
    offset: u64,
) -> Read<'a, BUF_LEN, RB> {
    Read { user_data, fd, buf, offset }
}

/*
pub unsafe fn read<'a, const BUF_LEN: usize>(
    user_data: u64,
    fd: BorrowedFd<'a>,
    buf: &mut impl ReadBuf<BUF_LEN>,
    offset: u64,
) -> impl FnOnce(&mut Sqe) + use<'a, BUF_LEN> {
    move |sqe| {
        sqe.opcode = Read;
        sqe.fd = fd.as_raw_fd();
        sqe.set_buf(buf.as_mut_ptr(), BUF_LEN, offset);
        sqe.user_data.u64_ = user_data;
    }
}
*/

// IoUring.zig has top level functions like read, nop etc inside the main struct
// But I think it's a lot more ergonomic to keep the IoUring interface small,
// and also make it clear to people that what you are doing is mutating SQEs.
/// These all only set the relevant fields,
/// they do not zero out everything - they are intented be called on the return
/// value of [`crate::IoUring::get_sqe`].
///
/// Mostly these follow liburing's `io_uring_accept_prep_`, but with some extra
/// additions to avoid flag and union wrangling. syscalls when they are
/// submitted.
impl Sqe {
    pub fn addr(&self) -> io_uring_ptr {
        // SAFETY: All the fields have the same underlying representation.
        unsafe { self.addr_or_splice_off_in.addr }
    }

    pub fn off(&self) -> u64 {
        // SAFETY: All the fields have the same underlying representation.
        unsafe { self.off_or_addr2.off }
    }

    pub fn set_len(&mut self, len: usize) {
        self.len.len =
            len.try_into().expect("io_uring requires lengths to fit in a u32");
    }

    pub fn set_buf<T>(&mut self, ptr: *const T, len: usize, offset: u64) {
        self.addr_or_splice_off_in.addr = io_uring_ptr::new(ptr as *mut c_void);
        self.set_len(len);
        self.off_or_addr2.off = offset;
    }

    pub fn clear_buf(&mut self) {
        self.set_buf(null::<c_long>(), 0, 0); // NULL is a long in C
    }

    /*
    pub fn prep_read(
        &mut self,
        user_data: u64,
        fd: BorrowedFd,
        buf: &mut [u8],
        offset: u64,
    ) {
        self.opcode = Read;
        self.fd = fd.as_raw_fd();
        self.set_buf(buf.as_ptr(), buf.len(), offset);
        self.user_data.u64_ = user_data;
    }
    */

    pub fn prep_readv(
        &mut self,
        user_data: u64,
        fd: &Fd,
        iovecs: &[iovec],
        offset: u64,
    ) {
        self.opcode = Readv;
        self.fd = fd.as_raw_fd();
        self.set_buf(iovecs.as_ptr(), iovecs.len(), offset);
        self.user_data.u64_ = user_data;
    }

    pub fn prep_readv_fixed(
        &mut self,
        user_data: u64,
        file_index: usize,
        iovecs: &[iovec],
        offset: u64,
    ) {
        self.opcode = Readv;
        self.fd = file_index
            .try_into()
            .expect("fixed file index must fit into a u32");
        self.set_buf(iovecs.as_ptr(), iovecs.len(), offset);
        self.flags.set(IoringSqeFlags::FIXED_FILE, true);
        self.user_data.u64_ = user_data;
    }

    pub fn prep_write(
        &mut self,
        user_data: u64,
        fd: &Fd,
        buf: &[u8],
        offset: u64,
    ) {
        self.opcode = Write;
        self.fd = fd.as_raw_fd();
        self.set_buf(buf.as_ptr(), buf.len(), offset);
        self.user_data.u64_ = user_data;
    }

    pub fn prep_writev(
        &mut self,
        user_data: u64,
        fd: &Fd,
        iovecs: &[iovec],
        offset: u64,
    ) {
        self.opcode = Writev;
        self.fd = fd.as_raw_fd();
        self.set_buf(iovecs.as_ptr(), iovecs.len(), offset);
        self.user_data.u64_ = user_data;
    }

    pub fn prep_splice(
        &mut self,
        user_data: u64,
        fd_in: &Fd,
        off_in: u64,
        fd_out: &Fd,
        off_out: u64,
        len: usize,
    ) {
        self.opcode = Splice;
        self.fd = fd_out.as_raw_fd();
        self.set_len(len);
        self.off_or_addr2.off = off_out;
        self.addr_or_splice_off_in.splice_off_in = off_in;
        self.splice_fd_in_or_file_index_or_addr_len.splice_fd_in =
            fd_in.as_raw_fd();
        self.user_data.u64_ = user_data;
    }

    pub fn prep_write_fixed(
        &mut self,
        user_data: u64,
        fd: &Fd,
        buffer: &iovec,
        offset: u64,
        buffer_index: u16,
    ) {
        self.opcode = WriteFixed;
        self.fd = fd.as_raw_fd();
        self.set_buf(buffer.iov_base, buffer.iov_len, offset);
        self.buf.buf_index = buffer_index;
        self.user_data.u64_ = user_data;
    }

    pub fn prep_read_fixed(
        &mut self,
        user_data: u64,
        fd: &Fd,
        buffer: &mut iovec,
        offset: u64,
        buffer_index: u16,
    ) {
        self.opcode = ReadFixed;
        self.fd = fd.as_raw_fd();
        self.set_buf(buffer.iov_base, buffer.iov_len, offset);
        self.buf.buf_index = buffer_index;
        self.user_data.u64_ = user_data;
    }

    pub fn prep_openat(
        &mut self,
        user_data: u64,
        fd: &Fd,
        path: &CStr,
        flags: OFlags,
        mode: Mode,
    ) {
        self.opcode = Openat;
        self.fd = fd.as_raw_fd();
        self.set_buf(path.as_ptr(), mode.as_raw_mode() as usize, 0);
        self.op_flags.open_flags = flags;
        self.user_data.u64_ = user_data;
    }

    /// Unlike other methods we take an OwnedFd here - should not be used after.
    pub fn prep_close(&mut self, user_data: u64, fd: Fd) {
        self.opcode = Close;
        self.fd = fd.as_raw_fd();
        // We want to prevent close being called on the fd when it goes out of
        // the scope...
        // Of course one could create this struct and never submit it to
        // io_uring...
        // TODO: revisit this when we make it a struct
        mem::forget(fd);
        self.user_data.u64_ = user_data;
    }

    /*
    pub fn prep_accept(
        &mut self,
        user_data: u64,
        fd: BorrowedFd,
        addr: &mut SocketAddrAny,
        flags: ReadWriteFlags,
    ) {
        self.opcode = Accept;
        self.fd = fd.as_raw_fd();
        // TODO I am not sure of this
        self.set_buf(addr.as_mut_ptr(), 0, addr.addr_len().into());
        self.op_flags.rw_flags = flags;
        self.user_data.u64_ = user_data;
    }

    pub fn prep_connect(
        &mut self,
        user_data: u64,
        fd: BorrowedFd,
        addr: &SocketAddrAny,
    ) {
        self.opcode = Connect;
        self.fd = fd.as_raw_fd();
        self.set_buf(addr.as_ptr(), 0, addr.addr_len().into());
        self.user_data.u64_ = user_data;
    }
    */
}
