# EXP-1: Matrix Multiplication using Sequential, OpenMP, MPI and CUDA

## Overview

This experiment implements matrix multiplication using four different computing models: **Sequential Programming, OpenMP, MPI, and CUDA**. The objective is to understand how different parallel computing techniques execute the same computational problem and compare their performance using execution time and speedup.

The experiment uses two **4000 × 4000** matrices where all elements are initialized to **1.0**. All implementations generate the same verification result **C[0][0] = 4000.00** while using different execution models.

---

## Theory

Matrix multiplication is a fundamental operation in scientific computing, artificial intelligence, machine learning, and image processing. As the matrix size increases, the computation becomes time-consuming. Parallel computing techniques divide the workload among multiple processing units to reduce execution time and improve performance.

This experiment compares four execution models:

* **Sequential Programming:** Executes the computation on a single CPU core using a normal C program.
* **OpenMP:** Uses multiple CPU threads in a shared-memory environment to execute matrix multiplication in parallel.
* **MPI:** Distributes the computation across multiple processes running on different Ubuntu virtual machines using message passing.
* **CUDA:** Executes matrix multiplication on an NVIDIA GPU using thousands of parallel CUDA threads for maximum performance.

---

## Technologies Used

* C Programming
* GCC Compiler
* OpenMP
* Open MPI
* CUDA Toolkit
* Windows PowerShell
* WSL2 Ubuntu
* VMware Workstation

---

## Experiment Structure

```text
EXP-1
│── README.md
│
├── Part-A_Sequential
├── Part-B_OpenMP
├── Part-C_MPI
└── Part-D_CUDA
```

Each folder contains:

* Source code (`.c` or `.cu`)
* Individual `README.md`
* Lab report (`.docx`)
* Screenshots of commands and outputs

---

## Performance Comparison

| Implementation | Execution Model                | Resources Used  | Execution Time |
| -------------- | ------------------------------ | --------------- | -------------: |
| Sequential     | Single CPU Execution           | 1 CPU Core      |  **244.120 s** |
| OpenMP         | Shared Memory Parallelism      | 8 CPU Threads   |   **30.830 s** |
| MPI            | Distributed Memory Parallelism | 4 MPI Processes |   **92.980 s** |
| CUDA           | GPU Parallel Computing         | NVIDIA GPU      |    **0.165 s** |

---

## Execution Time Graph

```text
Execution Time (Seconds)

Sequential  ██████████████████████████████ 244.12

OpenMP      ████                           30.83

MPI         ███████████                    92.98

CUDA         ▏                             0.165
```

**Observation:** CUDA provides the lowest execution time, while Sequential execution takes the highest time for the same matrix multiplication.

---

## Speedup Comparison

Speedup is calculated using:

**Speedup = Sequential Execution Time / Parallel Execution Time**

| Implementation |      Speedup |
| -------------- | -----------: |
| Sequential     |    **1.00×** |
| OpenMP         |    **7.92×** |
| MPI            |    **2.63×** |
| CUDA           | **1479.48×** |

### Speedup Graph

```text
Speedup over Sequential

Sequential  █ 1×

MPI         ███ 2.63×

OpenMP      ████████ 7.92×

CUDA        ██████████████████████████████████████ 1479.48×
```

---

## Result Summary

* Sequential implementation provides the baseline execution time.
* OpenMP improves performance using multi-threaded CPU execution.
* MPI distributes computation across multiple virtual machines using message passing.
* CUDA achieves the highest performance by executing matrix multiplication on the GPU.
* All implementations produce the correct verification value **C[0][0] = 4000.00**.

---

## Conclusion

This experiment demonstrates the practical difference between **Sequential**, **OpenMP**, **MPI**, and **CUDA** implementations of matrix multiplication. Although all four implementations solve the same problem correctly, their execution models and performance differ significantly. Parallel computing techniques reduce execution time by utilizing CPU threads, distributed processes, and GPU cores, making them suitable for high-performance computing applications.
