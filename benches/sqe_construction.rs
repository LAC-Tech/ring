use core::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use hringas::{io_uring_sqe, sqe, IoringOp};

fn nop_with_default(user_data: u64) -> io_uring_sqe {
    let mut sqe = io_uring_sqe { opcode: IoringOp::Nop, ..Default::default() };
    sqe.user_data.u64_ = user_data;
    sqe
}

fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("io_uring_sqe_nop_construction");

    group.bench_function("with_Default::default()", |b| {
        b.iter(|| {
            let sqe = nop_with_default(black_box(42));
            black_box(sqe); // prevent full elision
        })
    });

    group.bench_function("with_mem::zeroed()", |b| {
        b.iter(|| {
            let sqe = sqe::nop(black_box(42));
            black_box(sqe); // prevent full elision
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
