use hringas::{Cqe, IoUring};
use std::fs;
use std::os::fd::AsFd;

fn main() {
    let mut ring = IoUring::new(8).unwrap();

    let fd = fs::File::open("README.md").unwrap();
    let mut buf = vec![0; 1024];

    let sqe = ring.get_sqe().unwrap();
    sqe.prep_read(0x42, fd.as_fd(), &mut buf, 0);

    // Note that the developer needs to ensure
    // that the entry pushed into submission queue is valid (e.g. fd, buffer).
    let Cqe { user_data, res, .. } = unsafe {
        ring.submit_and_wait(1).expect("completion queue is empty");
        ring.copy_cqe().unwrap()
    };

    assert_eq!(user_data.u64_(), 0x42);
    assert!(res >= 0, "read error: {}", res);
}
