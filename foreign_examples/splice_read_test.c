// Compile: clang -Wall -Wextra splice_read_test.c -luring -lpthread

#include <fcntl.h>
#include <liburing.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

const size_t buf_len = 20;

void *test() {
    struct io_uring ring;
    int ret = io_uring_queue_init(4, &ring, 0);
    if (ret < 0) {
        fprintf(stderr, "io_uring_queue_init failed: %d\n", ret);
        return NULL;
    }

    // Create temp files
    int fd_src =
        open("/tmp/test_io_uring_splice_src", O_RDWR | O_CREAT | O_TRUNC, 0644);
    int fd_dst = open("/tmp/test_io_uring_splice_dest",
                      O_RDWR | O_CREAT | O_TRUNC, 0644);

    uint8_t buffer_write[buf_len];
    memset(buffer_write, 97, buf_len);
    uint8_t buffer_read[buf_len];
    memset(buffer_read, 98, buf_len);
    write(fd_src, buffer_write, buf_len);
    lseek(fd_src, 0, SEEK_SET);

    int pipe_fds[2];
    if (pipe(pipe_fds) < 0) {
        perror("pipe");
        goto cleanup;
    }

    struct io_uring_sqe *sqe = NULL;

    sqe = io_uring_get_sqe(&ring);
    io_uring_prep_splice(sqe, fd_src, 0, pipe_fds[1], -1, 20, 0);
    sqe->user_data = 1;

    sqe = io_uring_get_sqe(&ring);
    io_uring_prep_splice(sqe, pipe_fds[0], -1, fd_dst, 10, 20, 0);
    sqe->user_data = 2;

    io_uring_submit(&ring);

    // Get completions
    struct io_uring_cqe *cqe;
    io_uring_wait_cqe(&ring, &cqe);
    io_uring_cqe_seen(&ring, cqe);
    io_uring_wait_cqe(&ring, &cqe);
    io_uring_cqe_seen(&ring, cqe);

    printf("Test passed!\n");

cleanup:
    close(pipe_fds[0]);
    close(pipe_fds[1]);
    close(fd_src);
    close(fd_dst);
    io_uring_queue_exit(&ring);
    unlink("/tmp/test_src");
    unlink("/tmp/test_dst");

    return NULL;
}

long get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

int main() {
    pthread_t thread;
    long start, end;

    printf("\n=== Main Thread ===\n");
    start = get_time_ns();
    test();
    end = get_time_ns();
    printf("Main thread test took: %.3f ms\n\n", (end - start) / 1000000.0);

    printf("=== Spawned Thread ===\n");
    start = get_time_ns();
    pthread_create(&thread, NULL, test, NULL);
    pthread_join(thread, NULL);
    end = get_time_ns();
    printf("Spawned thread test took: %.3f ms\n\n", (end - start) / 1000000.0);

    return 0;
}
