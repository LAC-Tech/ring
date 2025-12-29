#![no_std]
use ring::rustix::fd::AsRawFd;
use ring::rustix::fs::{openat, Mode, OFlags, CWD};
use ring::rustix::io_uring::io_uring_cqe;
use ring::{prep, IoUring};

fn main() {
    let mut ring = IoUring::new(8).unwrap();

    let fd = openat(CWD, "README.md", OFlags::RDONLY, Mode::empty()).unwrap();
    let mut buf = [0; 1024];

    // Note that the developer needs to ensure
    // that the entry pushed into submission queue is valid (e.g. fd, buffer).
    let io_uring_cqe { user_data, res, .. } = unsafe {
        let sqe = ring.get_sqe().unwrap();
        prep::read(sqe, 0x42, fd.as_raw_fd(), &mut buf, 0);
        ring.submit_and_wait(1).expect("completion queue is empty");
        ring.copy_cqe().unwrap()
    };

    assert_eq!(user_data.u64_(), 0x42);
    assert!(res >= 0, "read error: {}", res);
}
