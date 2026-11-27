# Experiment 3 – Cache Coherence and NUMA Memory Access Performance

### PGC Lab Experiments

---

## 1. Experiment Overview

This experiment studies cache coherence protocols and NUMA (Non-Uniform Memory Access) performance in multi-core processors.

The experiment focuses on three cache coherence protocols:

- MSI (Modified, Shared, Invalid)
- MESI (Modified, Exclusive, Shared, Invalid)
- MOESI (Modified, Owned, Exclusive, Shared, Invalid)

It also investigates false sharing and the performance difference between local and remote NUMA memory access.

## 2. Objectives

- Understand cache coherence in multi-core processors.
- Study MSI, MESI and MOESI protocols.
- Understand cache-line states and invalidation.
- Analyze false sharing between multiple threads.
- Compare adjacent and padded memory layouts.
- Understand NUMA architecture.
- Measure local and remote memory access performance.
- Analyze execution time and scalability.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| C | Benchmark implementation |
| Python 3 | Cache coherence simulation |
| GCC | C program compilation |
| OpenMP | Multi-threaded programming |
| numactl | NUMA memory and CPU binding |
| hwloc | Hardware topology inspection |
| Ubuntu 22.04 LTS | Execution environment |
| VMware Workstation | Virtual machine environment |

## 4. Experimental Environment

| Component | Configuration |
|---|---|
| Virtualization Software | VMware Workstation Pro |
| Guest Operating System | Ubuntu 22.04 LTS |
| Compiler | GCC |
| Parallel Programming | OpenMP |
| Cache Protocols | MSI, MESI, MOESI |
| NUMA Tool | numactl |
| Python Version | Python 3 |

## 5. Experiment Description

### 5.1 Cache Coherence Protocol Simulation

The first program simulates three cache coherence protocols:

- MSI
- MESI
- MOESI

It models two processor cores accessing shared cache lines.

The simulator compares bus transactions, invalidations, write-backs and cache-to-cache transfers.

### 5.2 False Sharing

False sharing occurs when multiple threads modify different variables that belong to the same cache line.

This experiment compares:

- Adjacent counters: counters share the same cache line.
- Padded counters: counters are placed on separate cache lines.

The purpose is to understand how false sharing affects execution time and scalability.

### 5.3 NUMA Memory Access

NUMA stands for Non-Uniform Memory Access.

In NUMA systems, memory access time can depend on the location of the memory relative to the processor.

This experiment compares:

- Local memory access
- Remote memory access

The `numactl` utility is used to bind CPU execution and memory allocation to NUMA nodes.

### 5.4 Performance Comparison

The comparison program evaluates adjacent and padded counters using different thread counts.

It also measures memory sweep execution time using different thread counts.

## 6. Installation and Setup

Open the Ubuntu terminal inside VMware.

Update the package list:

```bash
sudo apt update
```

Install the required packages:

```bash
sudo apt install build-essential numactl hwloc python3 -y
```

Verify GCC:

```bash
gcc --version
```

Verify OpenMP:

```bash
echo '#include <omp.h>' | gcc -E -fopenmp -xc - > /dev/null && echo OpenMP OK
```

Check NUMA hardware:

```bash
numactl --hardware
```

Check CPU topology:

```bash
lscpu
```

Check CPU-to-NUMA-node mapping:

```bash
lscpu -e
```

## 7. Compilation and Execution

Enter the experiment directory:

```bash
cd 03-Cache-Coherence-NUMA
```

### 7.1 Program 1 – MSI, MESI and MOESI Simulator

Run the Python simulator:

```bash
python3 src/coherence_sim.py 1000
```

This executes 1000 iterations of the cache coherence simulation.

### 7.2 Program 2 – False Sharing

Compile:

```bash
gcc -O2 -fopenmp src/false_sharing.c -o false_sharing
```

Run with two threads using adjacent counters:

```bash
OMP_NUM_THREADS=2 ./false_sharing adjacent
```

Run with two threads using padded counters:

```bash
OMP_NUM_THREADS=2 ./false_sharing padded
```

### 7.3 Program 3 – NUMA Memory Sweep

Compile:

```bash
gcc -O2 -fopenmp src/numa_sweep.c -o numa_sweep
```

Run with local memory placement:

```bash
OMP_NUM_THREADS=2 numactl --cpunodebind=0 --membind=0 ./numa_sweep
```

Run with remote memory placement:

```bash
OMP_NUM_THREADS=2 numactl --cpunodebind=0 --membind=1 ./numa_sweep
```

### 7.4 Program 4 – Performance Comparison

Compile:

```bash
gcc -O2 -fopenmp src/compare_numa.c -o compare_numa
```

Run:

```bash
./compare_numa
```

Run with local NUMA memory:

```bash
numactl --cpunodebind=0 --membind=0 ./compare_numa
```

Run with remote NUMA memory:

```bash
numactl --cpunodebind=0 --membind=1 ./compare_numa
```

**Important:** Replace NUMA node numbers with the nodes available on your Ubuntu system. Remote-memory testing requires at least two available NUMA nodes.

## 8. Experimental Parameters

| Parameter | Value |
|---|---:|
| Cache Coherence Protocols | 3 |
| Simulated Iterations | 1000 |
| False Sharing Increments | 100,000,000 |
| Thread Configurations | 1, 2 and 4 |
| Padded Counter Stride | 8 long elements |
| Cache Line Size Assumption | 64 bytes |
| NUMA Memory Buffer | 512 MB |
| Memory Sweep Passes | 4 |

## 9. Results and Observations

### 9.1 Cache Coherence

The simulation demonstrates that:

- MSI uses Modified, Shared and Invalid states.
- MESI introduces the Exclusive state to reduce unnecessary bus transactions.
- MOESI introduces the Owned state to support cache-to-cache data transfers without an immediate memory write-back.

### 9.2 False Sharing

The experiment compares adjacent and padded counter layouts.

Expected observations:

- Adjacent counters can generate more cache coherence traffic.
- Padded counters reduce false sharing.
- Padded layouts can improve performance and scalability.

### 9.3 NUMA Performance

The memory sweep compares local and remote memory placement.

Expected observations:

- Local memory access is generally faster on multi-socket NUMA systems.
- Remote memory access can introduce additional latency.
- On a single-socket computer or virtual machine, the difference may be small or unavailable.

### 9.4 Performance Comparison

Execution time and correctness are compared across different thread counts.

Actual timing results depend on CPU hardware, VM configuration and memory placement. Record the measured values from your own execution.

## 10. Execution Screenshots

The following screenshots document the experiment execution.

### 10.1 Ubuntu Environment

![Ubuntu Environment](screenshots/01-ubuntu-environment.png)

### 10.2 NUMA Hardware Information

![NUMA Topology](screenshots/02-numa-topology.png)

### 10.3 Cache Coherence Simulation

![Cache Coherence Output](screenshots/03-cache-coherence-output.png)

### 10.4 False Sharing Execution

![False Sharing Output](screenshots/04-false-sharing-output.png)

### 10.5 NUMA Memory Sweep

![NUMA Memory Output](screenshots/05-numa-memory-output.png)

### 10.6 Performance Comparison

![Comparison Output](screenshots/06-comparison-output.png)

## 11. Project Directory Structure

```text
03-Cache-Coherence-NUMA/
│
├── README.md
│
├── src/
│   ├── coherence_sim.py
│   ├── false_sharing.c
│   ├── numa_sweep.c
│   └── compare_numa.c
│
└── screenshots/
    ├── 01-ubuntu-environment.png
    ├── 02-numa-topology.png
    ├── 03-cache-coherence-output.png
    ├── 04-false-sharing-output.png
    ├── 05-numa-memory-output.png
    └── 06-comparison-output.png
```

## 12. Conclusion

This experiment demonstrates how cache coherence protocols maintain data consistency in multi-core processors and how false sharing affects parallel performance.

The MSI, MESI and MOESI simulations illustrate differences in cache state transitions and coherence traffic.

The false-sharing experiment demonstrates the importance of cache-line alignment and padding, while the NUMA experiment explains the performance impact of local and remote memory placement.

Overall, the experiment provides practical experience with cache coherence, OpenMP parallel programming, NUMA architecture and multi-core performance analysis.

---

**PGC Lab – Experiment 3: Cache Coherence and NUMA Memory Access Performance**
