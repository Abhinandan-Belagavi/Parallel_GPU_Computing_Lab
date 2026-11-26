# CUDA Launch Configuration Analyzer

### PGC CUDA Lab Evaluation – Theme 5: Occupancy and Block Size Tuning

---

## 1. Project Overview

The **CUDA Launch Configuration Analyzer** is a GPU performance analysis project developed using NVIDIA CUDA.

The project investigates how different CUDA kernel launch configurations affect GPU occupancy and execution time.

Using a CUDA vector addition program, we evaluate different thread block sizes and dynamic shared-memory reservations to understand their impact on GPU resource utilization and kernel performance.

The main goal is to analyze different configurations and identify the configuration that provides the fastest measured execution time.

## 2. Project Objectives

- Understand CUDA grid and thread block configurations.
- Analyze the effect of block size on GPU occupancy.
- Study GPU resource utilization.
- Investigate dynamic shared-memory reservation.
- Measure CUDA kernel execution time.
- Compare different GPU launch configurations.
- Identify the best-performing configuration among the tested configurations.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| CUDA C/C++ | GPU programming |
| NVIDIA CUDA Toolkit | CUDA compilation and execution |
| NVIDIA GPU | Parallel processing |
| Visual Studio Build Tools | CUDA compilation on Windows |
| C++ | CPU and GPU vector addition implementation |

## 4. Hardware and Software Configuration

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti |
| Compute Capability | 12.0 |
| Streaming Multiprocessors | 36 |
| Warp Size | 32 |
| Maximum Threads per Block | 1024 |
| CUDA Runtime Version | 13.3 |
| Operating System | Windows |

## 5. Project Architecture and Workflow

The project follows a sequence of GPU device detection, CUDA kernel execution, occupancy analysis, and performance measurement.

### Workflow

1. **GPU Detection:** Identify the available NVIDIA GPU and collect its hardware specifications.
2. **CUDA Vector Addition:** Execute vector addition on the GPU and compare its output with CPU computation.
3. **Launch Configuration:** Test different CUDA thread block sizes.
4. **Resource Analysis:** Evaluate theoretical occupancy and dynamic shared-memory reservations.
5. **Performance Measurement:** Record kernel execution time for each configuration.
6. **Configuration Selection:** Identify the fastest measured configuration.

## 6. Compilation and Execution

The CUDA programs were compiled and executed using NVIDIA CUDA Toolkit on Windows.

Open CMD or PowerShell in the project directory.

### 6.1 Open the Project Directory

```bash
cd cuda-occupancy-block-tuning
```

### 6.2 Check CUDA Compiler

```bash
nvcc --version
```

This command displays the installed NVIDIA CUDA compiler version.

### 6.3 GPU Device Information

**Compile:**

```bash
nvcc src/device_query.cu -o device_query.exe
```

**Run:**

```bash
.\device_query.exe
```

This program displays NVIDIA GPU hardware information, including GPU name, compute capability, streaming multiprocessors, and memory information.

### 6.4 CUDA Vector Addition

**Compile:**

```bash
nvcc src/vector_add_basic.cu -o vector_add_basic.exe
```

**Run:**

```bash
.\vector_add_basic.exe
```

This program performs vector addition on the CPU and GPU and checks whether their results match.

### 6.5 CUDA Launch Configuration Analyzer

**Compile:**

```bash
nvcc src/analyzer.cu -o analyzer.exe
```

**Run:**

```bash
.\analyzer.exe
```

This program evaluates different CUDA block sizes and dynamic shared-memory reservations, measures kernel execution time, and validates the output.

The analyzer generates a benchmark results file:

```text
benchmark_results.csv
```

### 6.6 Compilation Requirements

- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit
- NVIDIA GPU driver
- Visual Studio Build Tools with C++ compiler
- Windows CMD or PowerShell

**Note:** Run commands from the project root directory. The executable files and generated CSV are created locally and do not need to be uploaded to GitHub.

## 7. Experimental Methodology

The experiment uses CUDA vector addition to study the impact of different launch configurations on GPU performance.

### 7.1 Block Size Configurations

The following thread block sizes were tested:

- 128 threads
- 256 threads
- 512 threads
- 1024 threads

### 7.2 Dynamic Shared-Memory Configurations

The following shared-memory reservations were tested:

- 0 bytes
- 1024 bytes
- 2048 bytes
- 4096 bytes

### 7.3 Benchmark Configuration

| Parameter | Value |
|---|---:|
| Benchmark Vector Size | 16,777,216 elements |
| Number of Block Sizes | 4 |
| Number of Shared-Memory Configurations | 4 |
| Total Configurations Tested | 16 |

Each configuration was evaluated using theoretical occupancy and measured CUDA kernel execution time.

## 8. Experimental Results and Analysis

The experiment evaluated 16 combinations of block size and dynamic shared-memory reservation.

### 8.1 Occupancy Results

According to the recorded experiment:

- Block sizes of 128, 256, and 512 threads achieved 100% theoretical occupancy across the tested reservations.
- The 1024-thread block configurations achieved 66.67% theoretical occupancy.
- Occupancy was evaluated using CUDA resource information and occupancy calculations.

### 8.2 Best Measured Configuration

The fastest measured configuration in the recorded experiment was:

| Parameter | Result |
|---|---:|
| Block Size | 512 threads |
| Dynamic Shared-Memory Reservation | 2048 bytes |
| Theoretical Occupancy | 100% |
| Registers per Thread | 12 |
| Kernel Execution Time | 0.5094 ms |
| Validation Errors | 0 |
| Result | PASS |

### 8.3 Key Observations

- Block size influences theoretical GPU occupancy.
- Different launch configurations can produce different measured execution times.
- The 512-thread configuration with a 2048-byte reservation achieved the lowest measured kernel execution time in the recorded experiment.
- The 1024-thread configurations showed lower theoretical occupancy in this experiment.

**Important Note:** Dynamic shared memory was reserved for resource analysis but was not used in the actual vector addition calculation. Therefore, this experiment studies the effect of resource reservation rather than shared-memory computation.

The reported benchmark values represent the recorded experiment and may differ when the programs are compiled and executed again.

## 9. Project Execution Screenshots

The following screenshots demonstrate GPU detection, CUDA program execution, correctness verification, and performance analysis.

### 9.1 GPU Device Information

Displays the detected NVIDIA GeForce RTX 5060 Ti and its hardware specifications, including compute capability, streaming multiprocessors, warp size, and memory information.

![GPU Device Information](screenshots/gpu_device_information.png)

### 9.2 CUDA Vector Addition – Correctness Test 1

Vector addition execution using 1,048,576 elements and a block size of 256 threads.

- CPU and GPU outputs match.
- Maximum absolute error: 0.000000
- Mismatched elements: 0
- Result: PASS

![Vector Addition Test 1](screenshots/vector_add_test_1.png)

### 9.3 CUDA Vector Addition – Correctness Test 2

Large vector addition execution using 16,777,216 elements and a block size of 1024 threads.

- CPU and GPU outputs match.
- Maximum absolute error: 0.000000
- Mismatched elements: 0
- Result: PASS

![Vector Addition Test 2](screenshots/vector_add_test_2.png)

### 9.4 CUDA Launch Configuration Analyzer

Displays the performance analysis across different block sizes and dynamic shared-memory reservations.

The analyzer reports:

- Block size
- Shared-memory reservation
- Theoretical occupancy
- Kernel execution time
- Validation errors

![CUDA Analyzer Output](screenshots/cuda_analyzer_output.png)

### 9.5 Final Benchmark Summary

Displays the recorded benchmark comparison and identifies the fastest measured configuration among the 16 tested configurations.

**Final recorded result:**

- Fastest block size: 512 threads
- Best shared-memory reservation: 2048 bytes
- Kernel execution time: 0.5094 ms
- Theoretical occupancy: 100%
- Result: PASS

![Final Benchmark Summary](screenshots/final_benchmark_summary.png)

## 10. Project Directory Structure

```text
cuda-occupancy-block-tuning/
│
├── README.md
├── .gitignore
│
├── src/
│   ├── device_query.cu
│   ├── vector_add_basic.cu
│   └── analyzer.cu
│
└── screenshots/
    ├── gpu_device_information.png
    ├── vector_add_test_1.png
    ├── vector_add_test_2.png
    ├── cuda_analyzer_output.png
    └── final_benchmark_summary.png
```

## 11. Team Members and Contributions

This project was developed collaboratively by a team of four members.

| Team Member | Contribution |
|---|---|
| Member 1 – Abhinandan | CUDA device information and vector addition implementation |
| Member 2 – Pradeep | CUDA occupancy analyzer and launch configuration experiments |
| Member 3 – Bhavana | Experimental analysis and performance evaluation |
| Member 4 – Zakiya | Testing, documentation, and GitHub repository management |

## 12. Conclusion

The CUDA Launch Configuration Analyzer demonstrates how different CUDA kernel launch configurations influence GPU resource utilization and execution performance.

By evaluating different thread block sizes and dynamic shared-memory reservations, we identified the fastest measured configuration among the tested experiments.

The recorded experiment achieved 100% theoretical occupancy and a measured kernel execution time of 0.5094 ms with a block size of 512 threads and a 2048-byte dynamic shared-memory reservation.

This project provided practical experience in CUDA programming, GPU occupancy analysis, kernel benchmarking, and performance evaluation.

---

**PGC CUDA Lab Evaluation – Theme 5: Occupancy and Block Size Tuning**
