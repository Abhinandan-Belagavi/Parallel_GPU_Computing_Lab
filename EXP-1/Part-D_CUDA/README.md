# Part D – CUDA Matrix Multiplication

## Overview

This experiment implements **Matrix Multiplication using CUDA** on an NVIDIA GPU to perform parallel computation using GPU cores.

## Theory

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform that executes computations on the GPU. The CPU transfers matrices to GPU memory, launches a CUDA kernel, and retrieves the computed output from the GPU. Thousands of GPU threads work simultaneously, making CUDA the fastest implementation in this experiment.

## Working Principle

1. Allocate memory on the CPU and GPU.
2. Copy matrices from CPU memory to GPU memory.
3. Launch the CUDA kernel using grid and block configuration.
4. Perform matrix multiplication in parallel on GPU threads.
5. Copy the output matrix back to CPU memory.
6. Verify the result and record execution time.

## Tools Used

* CUDA C/C++
* CUDA Toolkit
* NVCC Compiler
* NVIDIA GPU
* Ubuntu Terminal

## Output

* Matrix Size: **4000 × 4000**
* Verification: **C[0][0] = 4000.00**
* Grid Size: **250 × 250 Blocks**
* Block Size: **16 × 16 Threads**
* Fastest execution among all implementations.
