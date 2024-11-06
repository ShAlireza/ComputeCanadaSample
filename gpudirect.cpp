#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>

// Error handling macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    // Initialize CUDA
    CHECK_CUDA(cudaSetDevice(0));

    // Allocate GPU memory
    size_t size = 1024 * 1024; // 1 MB
    void *d_ptr;
    CHECK_CUDA(cudaMalloc(&d_ptr, size));

    // Initialize InfiniBand resources
    struct ibv_device **dev_list = ibv_get_device_list(NULL);
    if (!dev_list) {
        perror("Failed to get IB devices list");
        return EXIT_FAILURE;
    }

    struct ibv_context *context = ibv_open_device(dev_list[0]);
    if (!context) {
        perror("Failed to open device");
        return EXIT_FAILURE;
    }

    struct ibv_pd *pd = ibv_alloc_pd(context);
    if (!pd) {
        perror("Failed to allocate protection domain");
        return EXIT_FAILURE;
    }

    // Register GPU memory for RDMA
    struct ibv_mr *mr = ibv_reg_mr(pd, d_ptr, size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
    if (!mr) {
        perror("Failed to register memory region");
        return EXIT_FAILURE;
    }

    // At this point, you can set up QPs and perform RDMA operations using the registered memory region

    // Clean up
    if (ibv_dereg_mr(mr)) {
        perror("Failed to deregister memory region");
        return EXIT_FAILURE;
    }

    if (ibv_dealloc_pd(pd)) {
        perror("Failed to deallocate protection domain");
        return EXIT_FAILURE;
    }

    if (ibv_close_device(context)) {
        perror("Failed to close device context");
        return EXIT_FAILURE;
    }

    ibv_free_device_list(dev_list);

    CHECK_CUDA(cudaFree(d_ptr));

    return EXIT_SUCCESS;
}

