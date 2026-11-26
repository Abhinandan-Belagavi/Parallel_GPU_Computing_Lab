
#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA Error: "                       \
                      << cudaGetErrorString(err) << "\n";     \
            return 1;                                         \
        }                                                     \
    } while (0)

__global__ void vectorAdd(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {

    const int N = 16777216;
    const size_t bytes = N * sizeof(float);

    int blockSizes[] = {128, 256, 512, 1024};
    size_t sharedSizes[] = {0, 1024, 2048, 4096};

    std::vector<float> A(N, 1.0f);
    std::vector<float> B(N, 2.0f);
    std::vector<float> C(N);

    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes,
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes,
                          cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    int device;

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "\n";
    std::cout << "===============================================\n";
    std::cout << "       CUDA LAUNCH CONFIGURATION ANALYZER\n";
    std::cout << "===============================================\n";

    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Vector Size: " << N << "\n";
    std::cout << "Streaming Multiprocessors: "
              << prop.multiProcessorCount << "\n\n";

    std::cout << std::left
              << std::setw(12) << "Block Size"
              << std::setw(16) << "Shared (Bytes)"
              << std::setw(16) << "Occupancy (%)"
              << std::setw(16) << "Time (ms)"
              << "\n";

    std::cout << "------------------------------------------------------------\n";

    float bestTime = 1e30f;
    int bestBlock = 0;
    size_t bestShared = 0;

    for (int block : blockSizes) {

        int activeBlocks = 0;

        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &activeBlocks,
            vectorAdd,
            block,
            0
        ));

        float occupancy =
            (float)(activeBlocks * block) /
            prop.maxThreadsPerMultiProcessor * 100.0f;

        for (size_t shared : sharedSizes) {

            int grid = (N + block - 1) / block;

            CUDA_CHECK(cudaEventRecord(start));

            vectorAdd<<<grid, block, shared>>>(
                d_A, d_B, d_C, N
            );

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float elapsed = 0.0f;

            CUDA_CHECK(cudaEventElapsedTime(
                &elapsed, start, stop
            ));

            CUDA_CHECK(cudaMemcpy(
                C.data(), d_C, bytes,
                cudaMemcpyDeviceToHost
            ));

            bool valid = true;

            for (int i = 0; i < N; i++) {
                if (C[i] != 3.0f) {
                    valid = false;
                    break;
                }
            }

            std::cout << std::left
                      << std::setw(12) << block
                      << std::setw(16) << shared
                      << std::setw(16) << std::fixed
                      << std::setprecision(2) << occupancy
                      << std::setw(16) << std::setprecision(4)
                      << elapsed;

            std::cout << (valid ? " PASS" : " FAIL") << "\n";

            if (valid && elapsed < bestTime) {
                bestTime = elapsed;
                bestBlock = block;
                bestShared = shared;
            }
        }
    }

    std::cout << "\n===============================================\n";
    std::cout << "             FINAL BEST CONFIGURATION\n";
    std::cout << "===============================================\n";

    std::cout << "Best Block Size: "
              << bestBlock << "\n";

    std::cout << "Shared Memory Reservation: "
              << bestShared << " bytes\n";

    std::cout << "Best Execution Time: "
              << std::fixed << std::setprecision(4)
              << bestTime << " ms\n";

    std::cout << "===============================================\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

