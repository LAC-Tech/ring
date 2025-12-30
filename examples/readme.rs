#![no_std]
use hringas::rustix::fs::{openat, Mode, OFlags, CWD};
use hringas::rustix::io::Errno;
use hringas::rustix::io_uring::io_uring_cqe;
use hringas::{IoUring, PrepSqe};

fn main() {
    let mut ring = IoUring::new(8).unwrap();

    let fd = openat(CWD, "README.md", OFlags::RDONLY, Mode::empty()).unwrap();
    let mut buf = [0; 1024];

    let mut sqe = ring.get_sqe().unwrap();
    sqe.prep_read(0x42, fd, &mut buf, 0);

    // Note that the developer needs to ensure
    // that the entry pushed into submission queue is valid (e.g. fd, buffer).
    let io_uring_cqe { user_data, res, .. } = unsafe {
        ring.submit_and_wait(1).expect("completion queue is empty");
        ring.copy_cqe().unwrap()
    };

    assert_eq!(user_data.u64_(), 0x42);
    assert!(res >= 0, "read error: {}", Errno::from_raw_os_error(-res));
}
