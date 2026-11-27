
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BATCH 256
#define INPUT_SIZE 128
#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 10
#define REPEATS 100

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error at %s:%d - %s\n",          \
                __FILE__, __LINE__, cudaGetErrorString(err));  \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)


// Matrix multiplication + bias
// C = A * B + bias

__global__ void matmul_bias(
    const float *A,
    const float *B,
    const float *bias,
    float *C,
    int M,
    int K,
    int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = bias[col];

        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}


// ReLU activation

__global__ void relu_kernel(float *data, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        data[i] = fmaxf(0.0f, data[i]);
    }
}


// Sigmoid activation

__global__ void sigmoid_kernel(float *data, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        data[i] = 1.0f / (1.0f + expf(-data[i]));
    }
}


// CPU matrix multiplication + bias

void cpu_matmul_bias(
    const float *A,
    const float *B,
    const float *bias,
    float *C,
    int M,
    int K,
    int N)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {

            float sum = bias[j];

            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}


// CPU ReLU

void cpu_relu(float *data, int size)
{
    for (int i = 0; i < size; i++) {
        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }
}


// CPU Sigmoid

void cpu_sigmoid(float *data, int size)
{
    for (int i = 0; i < size; i++) {
        data[i] = 1.0f / (1.0f + expf(-data[i]));
    }
}


// Compare CPU and GPU outputs

float max_difference(
    const float *a,
    const float *b,
    int size)
{
    float max_diff = 0.0f;

    for (int i = 0; i < size; i++) {
        float diff = fabsf(a[i] - b[i]);

        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    return max_diff;
}


// Initialize data

void initialize(float *data, int size, unsigned int seed)
{
    srand(seed);

    for (int i = 0; i < size; i++) {
        data[i] = ((float)(rand() % 2001) / 1000.0f) - 1.0f;
    }
}


int main()
{
    printf("EXPERIMENT 5: NEURAL NETWORK LAYERS IN CUDA\n");

    // Detect GPU

    int device = 0;
    cudaDeviceProp prop;

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("GPU detected: %s\n", prop.name);

    // Matrix dimensions

    const int input_count = BATCH * INPUT_SIZE;
    const int weight1_count = INPUT_SIZE * HIDDEN_SIZE;
    const int hidden_count = BATCH * HIDDEN_SIZE;
    const int weight2_count = HIDDEN_SIZE * OUTPUT_SIZE;
    const int output_count = BATCH * OUTPUT_SIZE;

    // Allocate CPU memory

    float *input = (float *)malloc(input_count * sizeof(float));
    float *weight1 = (float *)malloc(weight1_count * sizeof(float));
    float *bias1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));

    float *hidden_cpu = (float *)malloc(hidden_count * sizeof(float));
    float *weight2 = (float *)malloc(weight2_count * sizeof(float));
    float *bias2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    float *output_cpu = (float *)malloc(output_count * sizeof(float));

    float *hidden_gpu = (float *)malloc(hidden_count * sizeof(float));
    float *output_gpu = (float *)malloc(output_count * sizeof(float));

    if (!input || !weight1 || !bias1 || !hidden_cpu ||
        !weight2 || !bias2 || !output_cpu ||
        !hidden_gpu || !output_gpu) {
        fprintf(stderr, "CPU memory allocation failed\n");
        return EXIT_FAILURE;
    }

    initialize(input, input_count, 10);
    initialize(weight1, weight1_count, 20);
    initialize(bias1, HIDDEN_SIZE, 30);
    initialize(weight2, weight2_count, 40);
    initialize(bias2, OUTPUT_SIZE, 50);

    // GPU memory allocation

    float *d_input, *d_weight1, *d_bias1, *d_hidden;
    float *d_weight2, *d_bias2, *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, input_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight1, weight1_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden, hidden_count * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_weight2, weight2_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, output_count * sizeof(float)));

    // Copy input data to GPU

    CUDA_CHECK(cudaMemcpy(
        d_input, input, input_count * sizeof(float),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_weight1, weight1, weight1_count * sizeof(float),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_bias1, bias1, HIDDEN_SIZE * sizeof(float),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_weight2, weight2, weight2_count * sizeof(float),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_bias2, bias2, OUTPUT_SIZE * sizeof(float),
        cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid1(
        (HIDDEN_SIZE + 15) / 16,
        (BATCH + 15) / 16
    );

    dim3 grid2(
        (OUTPUT_SIZE + 15) / 16,
        (BATCH + 15) / 16
    );

    int threads = 256;

    printf("\nTesting individual CUDA kernels...\n");

    // Test matrix multiplication + bias

    matmul_bias<<<grid1, block>>>(
        d_input, d_weight1, d_bias1,
        d_hidden, BATCH, INPUT_SIZE, HIDDEN_SIZE
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cpu_matmul_bias(
        input, weight1, bias1,
        hidden_cpu, BATCH, INPUT_SIZE, HIDDEN_SIZE
    );

    CUDA_CHECK(cudaMemcpy(
        hidden_gpu, d_hidden,
        hidden_count * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    float diff1 = max_difference(
        hidden_cpu, hidden_gpu, hidden_count
    );

    printf("Matrix multiplication + bias: %s\n",
           diff1 < 1e-4f ? "PASSED" : "FAILED");

    // Test ReLU

    float *d_relu_test;
    float *relu_cpu = (float *)malloc(hidden_count * sizeof(float));
    float *relu_gpu = (float *)malloc(hidden_count * sizeof(float));

    CUDA_CHECK(cudaMalloc(
        &d_relu_test, hidden_count * sizeof(float)
    ));

    for (int i = 0; i < hidden_count; i++) {
        relu_cpu[i] = hidden_cpu[i];
    }

    CUDA_CHECK(cudaMemcpy(
        d_relu_test, hidden_cpu,
        hidden_count * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    relu_kernel<<<
        (hidden_count + threads - 1) / threads, threads
    >>>(d_relu_test, hidden_count);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cpu_relu(relu_cpu, hidden_count);

    CUDA_CHECK(cudaMemcpy(
        relu_gpu, d_relu_test,
        hidden_count * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    float diff2 = max_difference(
        relu_cpu, relu_gpu, hidden_count
    );

    printf("ReLU: %s\n",
           diff2 < 1e-4f ? "PASSED" : "FAILED");

    // Test Sigmoid

    float *d_sigmoid_test;
    float *sigmoid_cpu = (float *)malloc(output_count * sizeof(float));
    float *sigmoid_gpu = (float *)malloc(output_count * sizeof(float));

    CUDA_CHECK(cudaMalloc(
        &d_sigmoid_test, output_count * sizeof(float)
    ));

    initialize(sigmoid_cpu, output_count, 60);

    for (int i = 0; i < output_count; i++) {
        sigmoid_gpu[i] = sigmoid_cpu[i];
    }

    CUDA_CHECK(cudaMemcpy(
        d_sigmoid_test, sigmoid_cpu,
        output_count * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    sigmoid_kernel<<<
        (output_count + threads - 1) / threads, threads
    >>>(d_sigmoid_test, output_count);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cpu_sigmoid(sigmoid_cpu, output_count);

    CUDA_CHECK(cudaMemcpy(
        sigmoid_gpu, d_sigmoid_test,
        output_count * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    float diff3 = max_difference(
        sigmoid_cpu, sigmoid_gpu, output_count
    );

    printf("Sigmoid: %s\n",
           diff3 < 1e-4f ? "PASSED" : "FAILED");

    // CPU forward pass timing

    const int cpu_repeats = 10;

    double cpu_start = 0.0;
    double cpu_end = 0.0;

    // Use CUDA-independent host clock

    cpu_start = (double)clock() / CLOCKS_PER_SEC;

    for (int r = 0; r < cpu_repeats; r++) {

        cpu_matmul_bias(
            input, weight1, bias1,
            hidden_cpu, BATCH, INPUT_SIZE, HIDDEN_SIZE
        );

        cpu_relu(hidden_cpu, hidden_count);

        cpu_matmul_bias(
            hidden_cpu, weight2, bias2,
            output_cpu, BATCH, HIDDEN_SIZE, OUTPUT_SIZE
        );

        cpu_sigmoid(output_cpu, output_count);
    }

    cpu_end = (double)clock() / CLOCKS_PER_SEC;

    double cpu_ms =
        ((cpu_end - cpu_start) * 1000.0) / cpu_repeats;

    // GPU kernel-only timing

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int r = 0; r < REPEATS; r++) {

        matmul_bias<<<grid1, block>>>(
            d_input, d_weight1, d_bias1,
            d_hidden, BATCH, INPUT_SIZE, HIDDEN_SIZE
        );

        relu_kernel<<<
            (hidden_count + threads - 1) / threads, threads
        >>>(d_hidden, hidden_count);

        matmul_bias<<<grid2, block>>>(
            d_hidden, d_weight2, d_bias2,
            d_output, BATCH, HIDDEN_SIZE, OUTPUT_SIZE
        );

        sigmoid_kernel<<<
            (output_count + threads - 1) / threads, threads
        >>>(d_output, output_count);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &kernel_ms, start, stop
    ));

    kernel_ms /= REPEATS;

    // GPU end-to-end timing including data transfers

    CUDA_CHECK(cudaEventRecord(start));

    for (int r = 0; r < REPEATS; r++) {

        CUDA_CHECK(cudaMemcpy(
            d_input, input,
            input_count * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        matmul_bias<<<grid1, block>>>(
            d_input, d_weight1, d_bias1,
            d_hidden, BATCH, INPUT_SIZE, HIDDEN_SIZE
        );

        relu_kernel<<<
            (hidden_count + threads - 1) / threads, threads
        >>>(d_hidden, hidden_count);

        matmul_bias<<<grid2, block>>>(
            d_hidden, d_weight2, d_bias2,
            d_output, BATCH, HIDDEN_SIZE, OUTPUT_SIZE
        );

        sigmoid_kernel<<<
            (output_count + threads - 1) / threads, threads
        >>>(d_output, output_count);

        CUDA_CHECK(cudaMemcpy(
            output_gpu, d_output,
            output_count * sizeof(float),
            cudaMemcpyDeviceToHost
        ));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float end_to_end_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &end_to_end_ms, start, stop
    ));

    end_to_end_ms /= REPEATS;

    // Final output comparison

    cpu_matmul_bias(
        input, weight1, bias1,
        hidden_cpu, BATCH, INPUT_SIZE, HIDDEN_SIZE
    );

    cpu_relu(hidden_cpu, hidden_count);

    cpu_matmul_bias(
        hidden_cpu, weight2, bias2,
        output_cpu, BATCH, HIDDEN_SIZE, OUTPUT_SIZE
    );

    cpu_sigmoid(output_cpu, output_count);

    CUDA_CHECK(cudaMemcpy(
        output_gpu, d_output,
        output_count * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    float final_diff = max_difference(
        output_cpu, output_gpu, output_count
    );

    printf("\nForward-pass dimensions:\n");
    printf("Input: %d x %d\n", BATCH, INPUT_SIZE);
    printf("Hidden layer: %d x %d\n", BATCH, HIDDEN_SIZE);
    printf("Output: %d x %d\n", BATCH, OUTPUT_SIZE);

    printf("\nAverage CPU compute time: %.6f ms\n", cpu_ms);
    printf("Average GPU kernel-only time: %.6f ms\n", kernel_ms);
    printf("Average GPU end-to-end time: %.6f ms\n", end_to_end_ms);

    printf("Kernel-only speedup: %.6fx\n",
           kernel_ms > 0 ? cpu_ms / kernel_ms : 0.0);

    printf("End-to-end speedup: %.6fx\n",
           end_to_end_ms > 0 ? cpu_ms / end_to_end_ms : 0.0);

    printf("\nMaximum absolute output difference: %.6f\n",
           final_diff);

    printf("CPU and GPU outputs: %s\n",
           final_diff < 1e-4f ? "MATCH (within tolerance)" : "DO NOT MATCH");

    printf("\nFirst 10 output values (GPU):\n");

    for (int i = 0; i < 10; i++) {
        printf("%.6f ", output_gpu[i]);
    }

    printf("\n");

    // Cleanup

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_input);
    cudaFree(d_weight1);
    cudaFree(d_bias1);
    cudaFree(d_hidden);
    cudaFree(d_weight2);
    cudaFree(d_bias2);
    cudaFree(d_output);
    cudaFree(d_relu_test);
    cudaFree(d_sigmoid_test);

    free(input);
    free(weight1);
    free(bias1);
    free(hidden_cpu);
    free(weight2);
    free(bias2);
    free(output_cpu);
    free(hidden_gpu);
    free(output_gpu);
    free(relu_cpu);
    free(relu_gpu);
    free(sigmoid_cpu);
    free(sigmoid_gpu);

    return 0;
}

