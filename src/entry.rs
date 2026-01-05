use rustix::io_uring::{io_uring_user_data, IoringCqeFlags};

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
