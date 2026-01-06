use core::ffi::c_void;
use rustix::fd::{AsRawFd, BorrowedFd, RawFd};
use rustix::io::ReadWriteFlags;
use rustix::io_uring::{
    addr3_or_cmd_union, addr_or_splice_off_in_union, buf_union, io_uring_ptr,
    io_uring_user_data, ioprio_union, iovec, len_union, off_or_addr2_union,
    op_flags_union, splice_fd_in_or_file_index_or_addr_len_union,
    IoringCqeFlags, IoringOp, IoringSqeFlags,
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

/// An io_uring Submission Queue Entry.
///
/// While rustix provides one, it has some issues:
/// <https://github.com/bytecodealliance/rustix/issues/1568>
///
/// This also lets use define prep_methods.
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct Sqe {
    pub opcode: IoringOp,
    pub flags: IoringSqeFlags,
    pub ioprio: ioprio_union,
    pub fd: RawFd,
    pub off_or_addr2: off_or_addr2_union,
    pub addr_or_splice_off_in: addr_or_splice_off_in_union,
    pub len: len_union,
    pub op_flags: op_flags_union,
    pub user_data: io_uring_user_data,
    pub buf: buf_union,
    pub personality: u16,
    pub splice_fd_in_or_file_index_or_addr_len:
        splice_fd_in_or_file_index_or_addr_len_union,
    pub addr3_or_cmd: addr3_or_cmd_union,
}

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

    pub fn prep_nop(&mut self, user_data: u64) {
        self.opcode = IoringOp::Nop;
        self.user_data = user_data.into();
    }

    pub fn prep_fsync(
        &mut self,
        user_data: u64,
        fd: BorrowedFd,
        flags: ReadWriteFlags,
    ) {
        self.opcode = IoringOp::Fsync;
        self.fd = fd.as_raw_fd();
        self.op_flags.rw_flags = flags;
        self.user_data.u64_ = user_data;
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

    pub fn prep_read(
        &mut self,
        user_data: u64,
        fd: BorrowedFd,
        buf: &mut [u8],
        offset: u64,
    ) {
        self.opcode = IoringOp::Read;
        self.fd = fd.as_raw_fd();
        self.set_buf(buf.as_ptr(), buf.len(), offset);
        self.user_data.u64_ = user_data;
    }

    pub fn prep_readv(
        &mut self,
        user_data: u64,
        fd: BorrowedFd,
        iovecs: &[iovec],
        offset: u64,
    ) {
        self.opcode = IoringOp::Readv;
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
        self.opcode = IoringOp::Readv;
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
        fd: BorrowedFd,
        buf: &[u8],
        offset: u64,
    ) {
        self.opcode = IoringOp::Write;
        self.fd = fd.as_raw_fd();
        self.set_buf(buf.as_ptr(), buf.len(), offset);
        self.user_data.u64_ = user_data;
    }

    pub fn prep_writev(
        &mut self,
        user_data: u64,
        fd: BorrowedFd,
        iovecs: &[iovec],
        offset: u64,
    ) {
        self.opcode = IoringOp::Writev;
        self.fd = fd.as_raw_fd();
        self.set_buf(iovecs.as_ptr(), iovecs.len(), offset);
        self.user_data.u64_ = user_data;
    }

    pub fn prep_splice(
        &mut self,
        user_data: u64,
        fd_in: BorrowedFd,
        off_in: u64,
        fd_out: BorrowedFd,
        off_out: u64,
        len: usize,
    ) {
        self.opcode = IoringOp::Splice;
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
        fd: BorrowedFd,
        buffer: &iovec,
        offset: u64,
        buffer_index: u16,
    ) {
        self.opcode = IoringOp::WriteFixed;
        self.fd = fd.as_raw_fd();
        self.set_buf(buffer.iov_base, buffer.iov_len, offset);
        self.buf.buf_index = buffer_index;
        self.user_data.u64_ = user_data;
    }

    pub fn prep_read_fixed(
        &mut self,
        user_data: u64,
        fd: BorrowedFd,
        buffer: &mut iovec,
        offset: u64,
        buffer_index: u16,
    ) {
        self.opcode = IoringOp::ReadFixed;
        self.fd = fd.as_raw_fd();
        self.set_buf(buffer.iov_base, buffer.iov_len, offset);
        self.buf.buf_index = buffer_index;
        self.user_data.u64_ = user_data;
    }
}
