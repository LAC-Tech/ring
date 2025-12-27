use rustix::io_uring::{io_uring_sqe, IoringOp};

pub unsafe fn nop(sqe: *mut io_uring_sqe) {
    *sqe = io_uring_sqe { opcode: IoringOp::Nop, ..Default::default() }
}
