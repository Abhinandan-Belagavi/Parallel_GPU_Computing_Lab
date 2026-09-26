# Part B – OpenMP Matrix Multiplication

## Overview

This experiment implements **Matrix Multiplication using OpenMP**, a shared-memory parallel programming model that utilizes multiple CPU threads to improve execution performance.

## Theory

OpenMP allows a program to execute multiple threads simultaneously on a multi-core processor. In this experiment, the outer loop of the matrix multiplication algorithm is parallelized using the `#pragma omp parallel for` directive. Each thread computes different rows of the output matrix while sharing the same memory space.

## Working Principle

1. Initialize matrices A and B.
2. Set the number of OpenMP threads using `OMP_NUM_THREADS`.
3. Parallelize the outer loop using OpenMP directives.
4. Execute matrix multiplication using multiple CPU threads.
5. Verify the output and measure execution time.

## Tools Used

* C Programming
* GCC Compiler with OpenMP (`-fopenmp`)
* WSL2 Ubuntu
* OpenMP Library

## Output

* Matrix Size: **4000 × 4000**
* Threads Used: **8**
* Verification: **C[0][0] = 4000.00**
* Faster execution than Sequential implementation.
