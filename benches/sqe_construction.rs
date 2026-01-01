use core::hint::black_box;
use core::mem;
use criterion::{criterion_group, criterion_main, Criterion};
use hringas::{io_uring_sqe, sqe, IoringOp};

struct Ring {
    sqe: io_uring_sqe,
}

impl Ring {
    fn new() -> Self {
        Self { sqe: Default::default() }
    }

    fn get_sqe(&mut self) -> &mut io_uring_sqe {
        &mut self.sqe
    }

    fn get_sqe_zeroed(&mut self) -> &mut io_uring_sqe {
        self.sqe = unsafe { mem::zeroed() };
        &mut self.sqe
    }
}

fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("io_uring_sqe_nop_construction");

    group.bench_function("set individual fields on pre-zeroed out sqe", |b| {
        let mut ring = Ring::new();
        b.iter(|| {
            let sqe = ring.get_sqe_zeroed();
            sqe.opcode = IoringOp::Nop;
            sqe.user_data = 42.into();
            black_box(sqe);
        })
    });

    group.bench_function("overwrite raw sqe", |b| {
        let mut ring = Ring::new();
        b.iter(|| {
            let sqe = ring.get_sqe();
            *sqe = io_uring_sqe {
                opcode: IoringOp::Nop,
                user_data: 42.into(),
                ..unsafe { mem::zeroed() }
            };
            black_box(sqe);
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
