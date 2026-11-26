# CUDA Launch Configuration Analyzer
### PGC CUDA Lab Evaluation – Theme 5: Occupancy and Block Size Tuning

## 📌 Project Overview

The **CUDA Launch Configuration Analyzer** is a GPU performance analysis project developed using NVIDIA CUDA.

The project investigates how CUDA kernel launch configurations affect GPU occupancy and execution time. It uses a vector addition program to evaluate different thread block sizes and dynamic shared-memory reservations.

The experiment analyzes GPU resource utilization, theoretical occupancy, and kernel execution performance to identify a suitable launch configuration.

## 🎯 Objectives

- Understand CUDA thread block and grid configurations.
- Analyze the effect of block size on GPU occupancy.
- Study register usage per thread.
- Investigate dynamic shared-memory reservation.
- Measure CUDA kernel execution time.
- Compare different launch configurations.
- Identify a configuration with better measured performance.

## 🛠️ Technologies Used

- NVIDIA CUDA C/C++
- Python
- Pandas
- Matplotlib
- NVIDIA CUDA Toolkit
- Visual Studio Build Tools

## 🖥️ Hardware Environment

- GPU: NVIDIA GeForce RTX 5060 Ti
- Compute Capability: 12.0
- Number of Streaming Multiprocessors: 36
- Operating System: Windows

## 🧪 Experimental Setup

The project evaluates the following configurations:

**Block sizes:** 128, 256, 512, and 1024 threads.

**Dynamic shared-memory reservations:** 0, 1024, 2048, and 4096 bytes.

The benchmark uses a vector size of 16,777,216 elements.

The analyzer records theoretical occupancy, register usage, and kernel execution time for each configuration.

## 📊 Key Result

The fastest measured configuration in the experiment was:

| Parameter | Result |
|---|---:|
| Block size | 512 threads |
| Dynamic shared memory | 2048 bytes |
| Theoretical occupancy | 100% |
| Registers per thread | 12 |
| Kernel execution time | 0.5094 ms |
| Validation errors | 0 |

*Note: The dynamic shared memory is reserved for the experiment; the vector addition calculation itself does not use that memory.*

## 📁 Project Structure

```text
cuda-occupancy-block-tuning/
│
├── src/
│   ├── device_query.cu
│   ├── vector_add_basic.cu
│   └── analyzer.cu
│
├── scripts/
│   ├── plot_results.py
│   └── final_summary.py
│
├── results/
│   ├── benchmark_results.csv
│   └── graphs/
│
├── screenshots/
│
├── README.md
├── requirements.txt
└── .gitignore
```

## 📈 Results and Visualizations

The project includes graphs comparing:

- Occupancy versus block size.
- Execution time versus block size.
- Shared-memory reservation versus execution time.

## 👥 Team Members

This project was developed collaboratively by a team of four members.

1. Member 1 – CUDA device information and vector addition.
2. Member 2 – CUDA occupancy analyzer and launch configuration experiments.
3. Member 3 – Result processing, Python scripts, and visualization.
4. Member 4 – Documentation, testing, and GitHub repository management.

## 🚀 Conclusion

The experiment demonstrates that CUDA launch configuration can influence GPU resource utilization and kernel execution performance.

Among the tested configurations, a block size of 512 threads with a 2048-byte dynamic shared-memory reservation achieved the fastest measured execution time.

The project provides practical experience with CUDA programming, occupancy analysis, GPU benchmarking, and performance visualization.
