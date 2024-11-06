#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int device1 = 0;
    int device2 = 1;
    size_t dataSize = 1024 * sizeof(float);

    // Check P2P capability
    int canAccessPeer = 0;
    cudaDeviceCanAccessPeer(&canAccessPeer, device1, device2);
    if (!canAccessPeer) {
        printf("P2P access not supported between device %d and device %d\n", device1, device2);
        return -1;
    }

    // Enable P2P access
    cudaSetDevice(device1);
    cudaDeviceEnablePeerAccess(device2, 0);
    cudaSetDevice(device2);
    cudaDeviceEnablePeerAccess(device1, 0);

    // Allocate memory on both devices
    float *d_A, *d_B;
    cudaSetDevice(device1);
    cudaMalloc(&d_A, dataSize);
    cudaSetDevice(device2);
    cudaMalloc(&d_B, dataSize);

    // Initialize data on device2
    // ... (initialize d_B with data)

    // Transfer data from device2 to device1
    cudaMemcpyPeer(d_A, device1, d_B, device2, dataSize);

    // Use d_A on device1
    // ... (process data on device1)

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaSetDevice(device1);
    cudaDeviceDisablePeerAccess(device2);
    cudaSetDevice(device2);
    cudaDeviceDisablePeerAccess(device1);

    return 0;
}

