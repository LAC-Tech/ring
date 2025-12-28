use core::ffi::c_void;

use rustix::io_uring::{io_uring_ptr, io_uring_sqe, IoringOp};

pub unsafe fn nop(sqe: *mut io_uring_sqe) {
    *sqe = io_uring_sqe { opcode: IoringOp::Nop, ..Default::default() }
}

pub unsafe fn rw(
    sqe: *mut io_uring_sqe,
    opcode: IoringOp,
    fd: i32,
    addr: io_uring_ptr,
    len: u32,
    offset: u64,
) {
    *sqe = io_uring_sqe { opcode, fd, ..Default::default() };
    (*sqe).addr_or_splice_off_in.addr = addr;
    (*sqe).len.len = len;
    (*sqe).off_or_addr2.off = offset;
}

pub unsafe fn read(
    sqe: *mut io_uring_sqe,
    fd: i32,
    buffer: &mut [u8],
    offset: u64,
) {
    rw(
        sqe,
        IoringOp::Read,
        fd,
        io_uring_ptr::new(buffer.as_mut_ptr() as *mut c_void),
        buffer.len() as u32,
        offset,
    );
}
