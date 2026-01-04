use hringas::{IoUring, IoringOp, SqeExt};
use rustix::fd::{AsFd, OwnedFd};
use rustix::fs::{openat, Mode, OFlags, CWD};
use rustix::io_uring::{io_uring_ptr, IoringCqeFlags, IoringSqeFlags};
use tempfile::{tempdir, TempDir};

use std::time::Instant;

fn temp_file(dir: &TempDir, path: &'static str) -> OwnedFd {
    openat(
        CWD,
        dir.path().join(path),
        OFlags::CREATE | OFlags::RDWR | OFlags::TRUNC,
        Mode::RUSR | Mode::WUSR,
    )
    .unwrap()
}

fn test() {
    let mut ring = IoUring::new(4).unwrap();

    let tmp = tempdir().unwrap();
    let fd_src = temp_file(&tmp, "test_io_uring_splice_src");
    let fd_dst = temp_file(&tmp, "test_io_uring_splice_dst");

    let buffer_write = [97u8; 20];
    let mut buffer_read = [98u8; 20];
    assert_eq!(
        rustix::io::write(fd_src.as_fd(), &buffer_write),
        Ok(buffer_write.len())
    );

    let (reading_fd_pipe, writing_fd_pipe) = rustix::pipe::pipe().unwrap();
    let pipe_offset = u64::MAX;

    {
        let mut sqe = ring.get_sqe().unwrap();
        sqe.prep_splice(
            0x11111111,
            fd_src.as_fd(),
            0,
            writing_fd_pipe.as_fd(),
            pipe_offset,
            buffer_write.len(),
        );
        assert_eq!(sqe.opcode, IoringOp::Splice);
        assert_eq!(sqe.addr(), io_uring_ptr::null());
        assert_eq!(sqe.off(), pipe_offset);
        sqe.flags.set(IoringSqeFlags::IO_LINK, true);
    }
    {
        let mut sqe = ring.get_sqe().unwrap();
        sqe.prep_splice(
            0x22222222,
            reading_fd_pipe.as_fd(),
            pipe_offset,
            fd_dst.as_fd(),
            10,
            buffer_write.len(),
        );
        assert_eq!(sqe.opcode, IoringOp::Splice);
        assert_eq!(
            unsafe { sqe.addr_or_splice_off_in.splice_off_in },
            pipe_offset
        );
        assert_eq!(sqe.off(), 10);
        sqe.flags.set(IoringSqeFlags::IO_LINK, true);
    }

    {
        let mut sqe = ring.get_sqe().unwrap();
        sqe.prep_read(0x33333333, fd_dst.as_fd(), &mut buffer_read, 10);
        assert_eq!(sqe.opcode, IoringOp::Read);
        assert_eq!(sqe.off(), 10);
    }

    // TODO: this slows everything down!!! why is it so slow?
    assert_eq!(unsafe { ring.submit() }, Ok(3));

    let cqe_splice_to_pipe = unsafe { ring.copy_cqe() }.unwrap();
    let cqe_splice_from_pipe = unsafe { ring.copy_cqe() }.unwrap();
    let cqe_read = unsafe { ring.copy_cqe() }.unwrap();

    assert_eq!(cqe_splice_to_pipe.user_data.u64_(), 0x11111111);
    assert_eq!(cqe_splice_to_pipe.res, buffer_write.len() as i32);
    assert_eq!(cqe_splice_to_pipe.flags, IoringCqeFlags::empty());

    assert_eq!(cqe_splice_from_pipe.user_data.u64_(), 0x22222222);
    assert_eq!(cqe_splice_from_pipe.res, buffer_write.len() as i32);
    assert_eq!(cqe_splice_from_pipe.flags, IoringCqeFlags::empty());

    assert_eq!(cqe_read.user_data.u64_(), 0x33333333);
    assert_eq!(cqe_read.res, buffer_read.len() as i32);
    assert_eq!(cqe_read.flags, IoringCqeFlags::empty());

    assert_eq!(&buffer_write, &buffer_read);
}

fn main() {
    println!("\n=== Running in main thread (should be fast) ===");
    let start = Instant::now();
    test();
    println!("Main thread test took: {:?}\n", start.elapsed());

    println!("=== Running in spawned thread (will likely be slow) ===");
    let start = Instant::now();
    let handle = std::thread::spawn(|| {
        test();
    });
    handle.join().unwrap();
    println!("Spawned thread test took: {:?}\n", start.elapsed());
}
