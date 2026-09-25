# Part C – MPI Distributed Matrix Multiplication

## Overview

This part implements **Distributed Matrix Multiplication using MPI** across four Ubuntu virtual machines. The workload is divided among multiple processes running on Master and Worker nodes.

## Theory

MPI (Message Passing Interface) is a distributed-memory parallel programming model where multiple processes communicate through message passing. Matrix rows are distributed using `MPI_Scatter`, Matrix B is shared using `MPI_Bcast`, and partial results are collected using `MPI_Gather`.

## Tools and Technologies

* C Programming
* Open MPI
* OpenSSH
* Ubuntu Virtual Machines
* VMware Workstation

## Files Included

* `matrix_mpi.c` – MPI source code.
* `Lab1_PartC_Report.docx` – Detailed lab report.
* `screenshots/` – VM setup, SSH configuration, compilation, and execution screenshots.

## Expected Output

* Matrix Size: **4000 × 4000**
* MPI Processes: **4**
* Verification: **C[0][0] = 4000.00**
* Distributed execution across four virtual machines.
