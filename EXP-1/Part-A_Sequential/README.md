# Part A – Sequential Matrix Multiplication

## Overview

This part implements **Sequential Matrix Multiplication** using the C programming language in the Ubuntu WSL environment. The program multiplies two **4000 × 4000** matrices using a single CPU core and records the execution time. This implementation serves as the baseline for comparing parallel computing techniques.

## Theory

Sequential matrix multiplication executes the computation one element at a time using three nested loops. Since only one CPU core performs the entire computation, the execution time is higher than parallel implementations. The output matrix is verified by checking the value of **C[0][0] = 4000.00**.

## Tools and Technologies

* C Programming
* GCC Compiler
* Windows PowerShell
* WSL2 Ubuntu

## Files Included

* `matrix_sequential.c` – Sequential matrix multiplication source code.
* `Lab1_PartA_Report.docx` – Detailed lab report.
* `screenshots/` – Commands, compilation, and output screenshots.

## Expected Output

* Matrix Size: **4000 × 4000**
* Verification: **C[0][0] = 4000.00**
* Execution Time: Sequential CPU execution.
