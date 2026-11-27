
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TRANSPOSE_N 4096
#define MATMUL_N 1024
#define TILE 32
#define REPEATS 5

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA Error: %s\n",                   \
                cudaGetErrorString(err));                     \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)


// PART A: Uncoalesced Matrix Transpose

__global__ void transpose_uncoalesced(
    const float *input,
    float *output,
    int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        output[col * N + row] = input[row * N + col];
    }
}


// PART A: Coalesced Matrix Transpose using Shared Memory

__global__ void transpose_coalesced(
    const float *input,
    float *output,
    int N)
{
    __shared__ float tile[TILE][TILE + 1];

    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;

    if (row < N && col < N) {
        tile[threadIdx.y][threadIdx.x] =
            input[row * N + col];
    }

    __syncthreads();

    int transposed_col = blockIdx.y * TILE + threadIdx.x;
    int transposed_row = blockIdx.x * TILE + threadIdx.y;

    if (transposed_row < N && transposed_col < N) {
        output[transposed_row * N + transposed_col] =
            tile[threadIdx.x][threadIdx.y];
    }
}


// PART B: Naive Matrix Multiplication

__global__ void naive_matmul(
    const float *A,
    const float *B,
    float *C,
    int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}


// Calculate average CUDA execution time

template <typename KernelLaunch>
float measure_time(KernelLaunch launch)
{
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEATS; i++) {
        launch();
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms, start, stop
    ));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / REPEATS;
}


// Validate transpose output

bool validate_transpose(
    const float *input,
    const float *output,
    int N)
{
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {

            float expected = input[row * N + col];
            float actual = output[col * N + row];

            if (fabsf(expected - actual) > 0.0001f) {
                return false;
            }
        }
    }

    return true;
}


// Validate selected matrix multiplication elements

bool validate_matmul(
    const float *A,
    const float *B,
    const float *C,
    int N)
{
    const int tests[][2] = {
        {0, 0},
        {1, 1},
        {10, 20},
        {100, 200},
        {500, 500},
        {N - 1, N - 1}
    };

    for (int t = 0; t < 6; t++) {

        int row = tests[t][0];
        int col = tests[t][1];

        float expected = 0.0f;

        for (int k = 0; k < N; k++) {
            expected +=
                A[row * N + k] * B[k * N + col];
        }

        float actual = C[row * N + col];

        if (fabsf(expected - actual) > 0.01f) {
            return false;
        }
    }

    return true;
}


int main()
{
    int device = 0;
    cudaDeviceProp prop;

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("GPU: %s (compute capability %d.%d)\n",
           prop.name,
           prop.major,
           prop.minor);


    // ==========================================
    // PART A: MATRIX TRANSPOSE
    // ==========================================

    const int N = TRANSPOSE_N;
    size_t matrix_bytes =
        (size_t)N * N * sizeof(float);

    float *h_input = (float *)malloc(matrix_bytes);
    float *h_uncoalesced = (float *)malloc(matrix_bytes);
    float *h_coalesced = (float *)malloc(matrix_bytes);

    if (!h_input || !h_uncoalesced || !h_coalesced) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N * N; i++) {
        h_input[i] = (float)(i % 1000) / 100.0f;
    }

    float *d_input, *d_uncoalesced, *d_coalesced;

    CUDA_CHECK(cudaMalloc((void **)&d_input, matrix_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_uncoalesced, matrix_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_coalesced, matrix_bytes));

    CUDA_CHECK(cudaMemcpy(
        d_input, h_input, matrix_bytes,
        cudaMemcpyHostToDevice
    ));

    dim3 transpose_block(TILE, TILE);
    dim3 transpose_grid(
        (N + TILE - 1) / TILE,
        (N + TILE - 1) / TILE
    );

    printf("\n=== PART A: Matrix transpose (N = %d) ===\n", N);

    float uncoalesced_ms = measure_time([&]() {
        transpose_uncoalesced<<<
            transpose_grid, transpose_block
        >>>(d_input, d_uncoalesced, N);

        CUDA_CHECK(cudaGetLastError());
    });

    float coalesced_ms = measure_time([&]() {
        transpose_coalesced<<<
            transpose_grid, transpose_block
        >>>(d_input, d_coalesced, N);

        CUDA_CHECK(cudaGetLastError());
    });

    CUDA_CHECK(cudaMemcpy(
        h_uncoalesced, d_uncoalesced, matrix_bytes,
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        h_coalesced, d_coalesced, matrix_bytes,
        cudaMemcpyDeviceToHost
    ));

    bool uncoalesced_pass =
        validate_transpose(h_input, h_uncoalesced, N);

    bool coalesced_pass =
        validate_transpose(h_input, h_coalesced, N);

    double total_bytes = 2.0 * matrix_bytes;

    double uncoalesced_bw =
        total_bytes / (uncoalesced_ms * 1e6);

    double coalesced_bw =
        total_bytes / (coalesced_ms * 1e6);

    printf("Uncoalesced : %.3f ms | %.1f GB/s | %s\n",
           uncoalesced_ms,
           uncoalesced_bw,
           uncoalesced_pass ? "PASS" : "FAIL");

    printf("Coalesced   : %.3f ms | %.1f GB/s | %s\n",
           coalesced_ms,
           coalesced_bw,
           coalesced_pass ? "PASS" : "FAIL");

    printf("Speedup (uncoalesced / coalesced) = %.2fx\n",
           uncoalesced_ms / coalesced_ms);


    // ==========================================
    // PART B: NAIVE MATRIX MULTIPLICATION
    // ==========================================

    const int M = MATMUL_N;
    size_t matmul_bytes =
        (size_t)M * M * sizeof(float);

    float *h_A = (float *)malloc(matmul_bytes);
    float *h_B = (float *)malloc(matmul_bytes);
    float *h_C = (float *)malloc(matmul_bytes);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < M * M; i++) {
        h_A[i] = (float)((i % 100) + 1) / 100.0f;
        h_B[i] = (float)((i % 50) + 1) / 50.0f;
    }

    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc((void **)&d_A, matmul_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, matmul_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, matmul_bytes));

    CUDA_CHECK(cudaMemcpy(
        d_A, h_A, matmul_bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B, h_B, matmul_bytes,
        cudaMemcpyHostToDevice
    ));

    dim3 matmul_block(16, 16);
    dim3 matmul_grid(
        (M + 15) / 16,
        (M + 15) / 16
    );

    printf("\n=== PART B: Naive matrix multiplication (N = %d) ===\n", M);

    float matmul_ms = measure_time([&]() {
        naive_matmul<<<matmul_grid, matmul_block>>>(
            d_A, d_B, d_C, M
        );

        CUDA_CHECK(cudaGetLastError());
    });

    CUDA_CHECK(cudaMemcpy(
        h_C, d_C, matmul_bytes,
        cudaMemcpyDeviceToHost
    ));

    bool matmul_pass =
        validate_matmul(h_A, h_B, h_C, M);

    double operations = 2.0 * M * M * M;

    double gflops =
        operations / (matmul_ms * 1e6);

    printf("Naive matmul: %.3f ms | %.1f GFLOPS | %s\n",
           matmul_ms,
           gflops,
           matmul_pass ? "PASS" : "FAIL");


    // ==========================================
    // SUMMARY
    // ==========================================

    printf("\n=== SUMMARY ===\n");

    printf("Transpose uncoalesced : %.3f ms\n",
           uncoalesced_ms);

    printf("Transpose coalesced   : %.3f ms\n",
           coalesced_ms);

    printf("Naive matmul          : %.3f ms\n",
           matmul_ms);


    // Cleanup

    cudaFree(d_input);
    cudaFree(d_uncoalesced);
    cudaFree(d_coalesced);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_input);
    free(h_uncoalesced);
    free(h_coalesced);

    free(h_A);
    free(h_B);
    free(h_C);

    return (uncoalesced_pass && coalesced_pass && matmul_pass)
        ? EXIT_SUCCESS : EXIT_FAILURE;
}

