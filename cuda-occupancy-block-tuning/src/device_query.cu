
#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;

    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA-compatible GPU detected.\n";
        return 1;
    }

    std::cout << "========================================\n";
    std::cout << "       CUDA DEVICE INFORMATION\n";
    std::cout << "========================================\n";

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;

        error = cudaGetDeviceProperties(&prop, i);

        if (error != cudaSuccess) {
            std::cerr << "Device query failed: "
                      << cudaGetErrorString(error) << "\n";
            return 1;
        }

        std::cout << "\nGPU Device: " << i << "\n";
        std::cout << "Name: " << prop.name << "\n";
        std::cout << "Compute Capability: "
                  << prop.major << "." << prop.minor << "\n";
        std::cout << "Streaming Multiprocessors: "
                  << prop.multiProcessorCount << "\n";
        std::cout << "Warp Size: " << prop.warpSize << "\n";
        std::cout << "Maximum Threads per Block: "
                  << prop.maxThreadsPerBlock << "\n";
        std::cout << "Global Memory: "
                  << prop.totalGlobalMem / (1024 * 1024)
                  << " MB\n";
        std::cout << "Shared Memory per Block: "
                  << prop.sharedMemPerBlock / 1024
                  << " KB\n";
    }

    std::cout << "\nDevice query completed successfully.\n";

    return 0;
}

