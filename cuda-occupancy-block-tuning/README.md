# CUDA Launch Configuration Analyzer
### PGC CUDA Lab Evaluation – Theme 5: Occupancy and Block Size Tuning

<p align="center">
  <img src="screenshots/banner.png" width="100%" alt="CUDA Launch Configuration Analyzer">
</p>

## 1. Project Overview

The **CUDA Launch Configuration Analyzer** is a GPU performance analysis project developed using NVIDIA CUDA.

The project studies how different CUDA kernel launch configurations affect GPU occupancy and execution time.

Using a vector addition program, we evaluate different thread block sizes and dynamic shared-memory reservations to understand their influence on GPU resource utilization and performance.

## 2. Project Objectives

- Understand CUDA thread blocks and grid configurations.
- Analyze the effect of block size on GPU occupancy.
- Study register usage per thread.
- Investigate dynamic shared-memory reservation.
- Measure CUDA kernel execution time.
- Compare different kernel launch configurations.
- Identify the best-performing configuration among the tested configurations.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| CUDA C/C++ | GPU programming |
| NVIDIA CUDA Toolkit | Compilation and execution |
| Python | Result analysis |
| Pandas | Benchmark data processing |
| Matplotlib | Performance visualization |
| Visual Studio Build Tools | CUDA compilation on Windows |

## 4. Hardware and Software Configuration

| Component | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti |
| Compute Capability | 12.0 |
| Streaming Multiprocessors | 36 |
| Operating System | Windows |
| CUDA Toolkit | 13.4 (verify installed compiler version) |
| Python | 3.12 |

## 5. Project Architecture

The project follows a workflow involving CUDA program execution, GPU resource analysis, performance measurement, and Python-based result visualization.

**Workflow:**

1. Device information is collected from the NVIDIA GPU.
2. A CUDA vector addition program is executed.
3. Different thread block sizes are tested.
4. Dynamic shared-memory reservations are varied.
5. GPU occupancy and execution time are measured.
6. Experimental results are processed using Python.
7. Performance comparisons are generated.

<p align="center">
  <img src="screenshots/architecture.png" width="100%" alt="Project Architecture">
</p>

## 6. Experimental Methodology

The experiment uses CUDA vector addition to investigate the impact of launch configurations on GPU performance.

Two main parameters are varied:

**Thread Block Size**
- 128 threads
- 256 threads
- 512 threads
- 1024 threads

**Dynamic Shared-Memory Reservation**
- 0 bytes
- 1024 bytes
- 2048 bytes
- 4096 bytes

The benchmark uses a vector size of 16,777,216 elements.

For each configuration, theoretical occupancy and kernel execution time are recorded.

## 7. Experimental Results

The fastest measured configuration in our experiment was:

| Parameter | Result |
|---|---:|
| Block Size | 512 threads |
| Shared-Memory Reservation | 2048 bytes |
| Theoretical Occupancy | 100% |
| Registers per Thread | 12 |
| Kernel Execution Time | 0.5094 ms |
| Validation Errors | 0 |

The correctness test was separately performed using 1,048,576 elements and reported zero mismatches.

**Observation:** The 512-thread configuration with a 2048-byte shared-memory reservation achieved the lowest measured execution time among the tested configurations.

*Note: The vector addition kernel reserves dynamic shared memory for resource analysis but does not use that memory in its actual calculation. The measured result should therefore be interpreted as a resource-reservation experiment.*

## 8. Execution Screenshots

The following screenshots demonstrate the project execution and experimental output.

### GPU Device Information

![GPU Device Information](screenshots/device_info.png)

### Vector Addition Correctness Test

![Correctness Test](screenshots/correctness_test.png)

### Benchmark Execution

![Benchmark Output](screenshots/benchmark_output.png)

### Final Execution Result

![Execution Result](screenshots/execution_result.png)

## 9. Project Directory Structure

```text
cuda-occupancy-block-tuning/
├── README.md
├── src/
│   ├── device_query.cu
│   ├── vector_add_basic.cu
│   └── analyzer.cu
├── scripts/
│   ├── plot_results.py
│   └── final_summary.py
├── screenshots/
├── requirements.txt
└── .gitignore
```

## 10. Team Members and Contributions

This project was developed collaboratively by a team of four members.

| Team Member | Contribution |
|---|---|
| Member 1 | CUDA device information and vector addition |
| Member 2 | Occupancy analyzer and launch configuration experiments |
| Member 3 | Python-based analysis and visualization |
| Member 4 | Testing, documentation, and GitHub integration |

## 11. Conclusion

This project demonstrates how CUDA kernel launch configurations influence GPU resource utilization and execution performance.

By comparing different block sizes and dynamic shared-memory reservations, we identified the fastest measured configuration among our experiments.

The project provides practical experience in CUDA programming, GPU occupancy analysis, kernel benchmarking, and performance visualization.

---

**PGC CUDA Lab Evaluation – Theme 5: Occupancy and Block Size Tuning**
