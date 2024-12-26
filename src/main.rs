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
    handle: i32,
    buffer: [u8; 1024],
    state: State,
}

// TCP Echo server
// nc localhost 12345
fn main() {
    // Initialize io_uring
    let mut ring = IoUring::new(32).unwrap();
    let listener = std::net::TcpListener::bind(("127.0.0.1", 12345)).unwrap();

    let server = Socket {
        handle: listener.as_raw_fd(),
        buffer: [0u8; 1024],
        state: State::Accept,
    };

    let accept_e = opcode::Accept::new(types::Fd(server.handle), ptr::null_mut(), ptr::null_mut())
        .build()
        .user_data(ptr::addr_of!(server) as u64);

    unsafe {
        ring.submission()
            .push(&accept_e)
            .expect("submission queue is full");
    }

    loop {
        ring.submit_and_wait(1).unwrap();

        let mut sq = unsafe { ring.submission_shared() };
        let cq = unsafe { ring.completion_shared() };

        for cqe in cq {
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
            match client.state {
                State::Accept => {
                    // Create socket for clint connection
                    let client_handle = cqe.result();
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

                    unsafe {
                        sq.push_multiple(&[recv_e, accept_e]).unwrap();
                    }
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

                    unsafe {
                        sq.push(&send_e).unwrap();
                    }
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

                    unsafe {
                        sq.push(&recv_e).unwrap();
                    }
                }
            }
        }
    }
}
