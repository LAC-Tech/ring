#![no_std]
use ring::{rustix, IoUring};
use rustix::fd::AsRawFd;
use rustix::fs::{openat, Mode, OFlags, CWD};
use rustix::io_uring::io_uring_cqe;

fn main() {
    let mut ring = IoUring::new(8).unwrap();

    let fd = openat(CWD, "README.md", OFlags::RDONLY, Mode::empty()).unwrap();
    let mut buf = [0; 1024];

    let io_uring_cqe { user_data, res, .. } = unsafe {
        ring.read(0x42, fd.as_raw_fd(), &mut buf, 0).unwrap();
        ring.submit_and_wait(1).unwrap();
        ring.copy_cqe().unwrap()
    };

    assert_eq!(user_data.u64_(), 0x42);
    assert!(res >= 0, "read error: {}", res);
}
