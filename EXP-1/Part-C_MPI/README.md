# Part C – MPI Distributed Matrix Multiplication

## Overview

This experiment implements **Distributed Matrix Multiplication using MPI** across four Ubuntu virtual machines connected through a network.

## Theory

MPI (Message Passing Interface) is a distributed-memory parallel programming model where multiple processes execute simultaneously on different systems. In this experiment, one VM acts as the **Master** and three VMs act as **Workers**. Matrix rows are distributed using `MPI_Scatter`, Matrix B is shared using `MPI_Bcast`, and the final results are collected using `MPI_Gather`.

## Working Principle

1. Configure one Master node and three Worker nodes.
2. Initialize matrices on the Master node.
3. Distribute rows of Matrix A to all processes.
4. Broadcast Matrix B to every process.
5. Each process computes its assigned rows.
6. Gather all results into the final output matrix.

## Tools Used

* C Programming
* Open MPI
* OpenSSH
* Ubuntu Virtual Machines
* VMware Workstation

## Output

* Matrix Size: **4000 × 4000**
* MPI Processes: **4**
* Verification: **C[0][0] = 4000.00**
* Distributed execution across multiple virtual machines.
