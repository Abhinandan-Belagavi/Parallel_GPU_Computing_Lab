# Experiment 1: Matrix Multiplication using Sequential, OpenMP, MPI, and CUDA

## Theory

Matrix multiplication is a fundamental operation used in scientific computing, artificial intelligence, machine learning, and image processing. In this experiment, two **4000 × 4000** matrices are multiplied using four different computing models: **Sequential Programming, OpenMP, MPI, and CUDA**. The purpose is to understand different parallel computing techniques and compare their execution performance.

* **Sequential Matrix Multiplication:** Executes the program on a single CPU core using three nested loops. It is the baseline implementation and provides the reference execution time for comparison.

* **OpenMP Matrix Multiplication:** Uses shared-memory parallel programming to divide the computation among multiple CPU threads. Multiple cores work simultaneously, reducing the overall execution time.

* **MPI Matrix Multiplication:** Uses distributed-memory parallel programming where the computation is divided among multiple processes running on different Ubuntu virtual machines. The processes communicate using message passing to complete the multiplication.

* **CUDA Matrix Multiplication:** Uses an NVIDIA GPU to perform matrix multiplication with thousands of parallel GPU threads. The CPU transfers data to GPU memory, the CUDA kernel performs the computation in parallel, and the result is copied back to the CPU, providing the fastest execution among all implementations.

All four implementations produce the same verification result **C[0][0] = 4000.00**, demonstrating the correctness of the matrix multiplication while highlighting the performance benefits of different parallel computing models.
