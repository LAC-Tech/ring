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
