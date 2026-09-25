# Part B – OpenMP Matrix Multiplication

## Overview

This part implements **Matrix Multiplication using OpenMP** for shared-memory parallel programming. The computation is divided among multiple CPU threads to improve performance over the sequential implementation.

## Theory

OpenMP is a shared-memory parallel programming model that allows multiple threads to execute different parts of a program simultaneously. The outer loop of the matrix multiplication algorithm is parallelized using `#pragma omp parallel for`, enabling multiple CPU cores to compute matrix rows concurrently.

## Tools and Technologies

* C Programming
* GCC Compiler with OpenMP (`-fopenmp`)
* WSL2 Ubuntu
* OpenMP Library

## Files Included

* `matrix_openmp.c` – OpenMP source code.
* `Lab1_PartB_Report.docx` – Detailed lab report.
* `screenshots/` – Thread configuration, compilation, CPU usage, and output screenshots.

## Expected Output

* Matrix Size: **4000 × 4000**
* Threads Used: **8**
* Verification: **C[0][0] = 4000.00**
* Faster execution than Sequential implementation.
