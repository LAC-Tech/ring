use rustix::fd::OwnedFd;
use rustix::io::Errno;
use rustix::io_uring::{io_uring_params, io_uring_setup, IoringSetupFlags};
use rustix::net::{ipproto, socket, AddressFamily, Ipv4Addr, SocketAddrV4, SocketType};

enum State {
    Accept,
    Recv,
    Send,
}

struct Socket {
    handle: OwnedFd,
    buffer: [u8; 1024],
    state: State,
}

fn main() -> Result<(), Errno> {
    let entries = 32;
    let flags = IoringSetupFlags::empty();

    // This is what Zig does with std.os.linux.IoUring.init
    let mut params = io_uring_params::default();
    params.sq_thread_idle = 1000;
    params.flags = flags;
    let mut ring = io_uring_setup(entries, &mut params);

    let handle = socket(AddressFamily::INET, SocketType::STREAM, Some(ipproto::TCP))?;

    let mut server = Socket {
        handle,
        buffer: [0u8; 1024],
        state: State::Accept,
    };

    let port = 12345;
    let socket = SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), port);

    Ok(())
}
