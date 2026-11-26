
#include <iostream>
#include <vector>
#include <cmath>
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

    const int N = 1048576;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    size_t bytes = N * sizeof(float);

    std::vector<float> A(N), B(N), C_cpu(N), C_gpu(N);

    for (int i = 0; i < N; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i * 2);
    }

    // CPU vector addition
    for (int i = 0; i < N; i++) {
        C_cpu[i] = A[i] + B[i];
    }

    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes,
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes,
                          cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpuTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

    CUDA_CHECK(cudaMemcpy(C_gpu.data(), d_C, bytes,
                          cudaMemcpyDeviceToHost));

    // Correctness validation
    float maxError = 0.0f;
    int mismatches = 0;

    for (int i = 0; i < N; i++) {
        float error = std::fabs(C_cpu[i] - C_gpu[i]);

        if (error > maxError)
            maxError = error;

        if (error > 1e-5f)
            mismatches++;
    }

    std::cout << "\n========================================\n";
    std::cout << "       CUDA VECTOR ADDITION TEST\n";
    std::cout << "========================================\n";

    std::cout << "Vector Size: " << N << "\n";
    std::cout << "Block Size: " << threads << "\n";
    std::cout << "Grid Size: " << blocks << "\n";
    std::cout << "GPU Kernel Time: " << gpuTime << " ms\n";
    std::cout << "Maximum Absolute Error: " << maxError << "\n";
    std::cout << "Mismatched Elements: " << mismatches << "\n";

    if (mismatches == 0)
        std::cout << "RESULT: PASS\n";
    else
        std::cout << "RESULT: FAIL\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return mismatches == 0 ? 0 : 1;
}

