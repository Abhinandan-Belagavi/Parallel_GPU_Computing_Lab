# Part A – Sequential Matrix Multiplication

## Overview

This experiment implements **Sequential Matrix Multiplication** using the C programming language in the Ubuntu WSL environment. It serves as the baseline implementation for comparing the performance of parallel computing techniques.

## Theory

Sequential Matrix Multiplication executes the computation using a **single CPU core**. The program uses three nested loops to multiply two **4000 × 4000** matrices. Each element of Matrix A and Matrix B is initialized with **1.0**, and the output matrix C is calculated one element at a time. Since all operations are executed sequentially, this implementation has the highest execution time among all four approaches.

## Working Principle

1. Initialize matrices A and B with the value **1.0**.
2. Initialize output matrix C with **0.0**.
3. Multiply matrices using three nested loops.
4. Record execution time using the `clock()` function.
5. Verify the result by checking **C[0][0] = 4000.00**.

## Tools Used

* C Programming
* GCC Compiler
* Windows PowerShell
* WSL2 Ubuntu

## Output

* Matrix Size: **4000 × 4000**
* Verification: **C[0][0] = 4000.00**
* Execution Model: Single CPU Core
