#![no_std]
use hringas::rustix::fd::AsFd;
use hringas::rustix::fs::{openat, Mode, OFlags, CWD};
use hringas::{Cqe, IoUring};

fn main() {
    let mut ring = IoUring::new(8).unwrap();

    let fd = openat(CWD, "README.md", OFlags::RDONLY, Mode::empty()).unwrap();
    let mut buf = [0; 1024];

    let sqe = ring.get_sqe().expect("submission queue is full");
    sqe.prep_read(0x42, fd.as_fd(), &mut buf, 0);

    // Note that the developer needs to ensure
    // that the entry pushed into submission queue is valid (e.g. fd, buffer).
    let Cqe { user_data, res, .. } = unsafe {
        ring.submit_and_wait(1).unwrap();
        ring.copy_cqe().expect("completion queue is empty")
    };

    assert_eq!(user_data.u64_(), 0x42);
    assert!(res >= 0, "read error: {}", res);
}
