use rustix::fd::OwnedFd;
use rustix::net::{
    bind_v4, ipproto, listen, socket, sockopt::set_socket_reuseaddr, AddressFamily, Ipv4Addr,
    SocketAddrV4, SocketType,
};
use rustix_uring::{opcode, types, IoUring};
use std::os::unix::io::{AsRawFd, FromRawFd};
use std::{io, ptr};

#[derive(Debug)]
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

fn main() -> io::Result<()> {
    // Initialize io_uring
    let mut ring = IoUring::new(32)?;

    let handle = socket(AddressFamily::INET, SocketType::STREAM, Some(ipproto::TCP))?;
    let server = Socket {
        handle,
        buffer: [0u8; 1024],
        state: State::Accept,
    };

    let addr = SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 12345);
    set_socket_reuseaddr(&server.handle, true)?;
    bind_v4(&server.handle, &addr)?;
    listen(&server.handle, 128)?;

    let accept_e = opcode::Accept::new(
        types::Fd(server.handle.as_raw_fd()),
        ptr::null_mut(),
        ptr::null_mut(),
    )
    .build()
    .user_data(ptr::addr_of!(server) as u64);

    unsafe {
        ring.submission()
            .push(&accept_e)
            .expect("submission queue is full");
    }

    loop {
        ring.submit_and_wait(1)?;

        let mut done = false;
        while !done {
            // We have to use submission queue because of rust's borrow checker.
            // Or else we have two mutable borrows of `ring` more than once.
            // This way we collect all ops and
            // submit them all at once at the end to the kernel.
            let mut queue = Vec::new();

            if let Some(cqe) = ring.completion().next() {
                let client_ptr = cqe.user_data() as *mut Socket;
                let client = unsafe { &mut *client_ptr };

                if cqe.result() < 0 {
                    eprintln!(
                        "Error in state {:?}: {}",
                        client.state,
                        io::Error::from_raw_os_error(-cqe.result())
                    );
                    continue;
                }

                // Docs for opcodes: https://docs.rs/io-uring/latest/io_uring/opcode/index.html
                match client.state {
                    State::Accept => {
                        // Create socket for clint connection
                        let client_handle =
                            unsafe { rustix::fd::OwnedFd::from_raw_fd(cqe.result()) };
                        let client = Box::new(Socket {
                            handle: client_handle,
                            buffer: [0u8; 1024],
                            state: State::Recv,
                        });

                        let client_ptr = Box::into_raw(client);

                        // Prepare recv op
                        let recv_e = opcode::Recv::new(
                            types::Fd(unsafe { (*client_ptr).handle.as_raw_fd() }),
                            unsafe { (*client_ptr).buffer.as_mut_ptr() },
                            1024,
                        )
                        .build()
                        .user_data(client_ptr as u64);

                        // Prepare next accept
                        let accept_e = opcode::Accept::new(
                            types::Fd(server.handle.as_raw_fd()),
                            ptr::null_mut(),
                            ptr::null_mut(),
                        )
                        .build()
                        .user_data(ptr::addr_of!(server) as u64);

                        queue.push(recv_e);
                        queue.push(accept_e);
                    }
                    State::Recv => {
                        let read = cqe.result() as usize;
                        if read == 0 {
                            // Connection closed
                            unsafe {
                                let _ = Box::from_raw(client_ptr);
                            }
                            continue;
                        }

                        let received_message = String::from_utf8_lossy(&client.buffer[..read]);
                        println!("Client sent: {}", received_message);

                        client.state = State::Send;

                        // Prepare send op
                        let send_e = opcode::Send::new(
                            types::Fd(client.handle.as_raw_fd()),
                            client.buffer.as_ptr(),
                            read as _,
                        )
                        .build()
                        .user_data(client_ptr as u64);

                        queue.push(send_e);
                    }
                    State::Send => {
                        client.state = State::Recv;

                        // Prepare next receive op
                        let recv_e = opcode::Recv::new(
                            types::Fd(client.handle.as_raw_fd()),
                            client.buffer.as_mut_ptr(),
                            1024,
                        )
                        .build()
                        .user_data(client_ptr as u64);

                        queue.push(recv_e);
                    }
                }
            } else {
                done = true;
            }

            // Submit all ops to the io_uring's queue which in turn submits them to the kernel.
            for sqe in queue {
                unsafe {
                    ring.submission()
                        .push(&sqe)
                        .expect("submission queue is full");
                }
            }
        }

        ring.submit()?;
    }
}
