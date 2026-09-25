Part B – OpenMP Matrix Multiplication
Student Name: Abhinandan Shantinath Belagavi

USN: 01FE24BCI052

1. Aim

To implement matrix multiplication using OpenMP and execute the program using multiple CPU threads in the Ubuntu WSL environment.

2. Objective

To understand shared-memory parallel programming using OpenMP.

To configure the OpenMP runtime.

To execute matrix multiplication using multiple CPU threads.

To compare execution time with sequential implementation.

3. Theory

OpenMP is a shared-memory parallel programming model that allows multiple CPU threads to execute different parts of the program simultaneously. In this experiment, the outer loop of the matrix multiplication algorithm is parallelized using the #pragma omp parallel for directive. Each thread processes different rows of the output matrix while sharing the same memory.


4. Requirements

Ubuntu WSL2

GCC Compiler

OpenMP Support (-fopenmp)

Multi-core Processor

5. Procedure
Step 1 – Launch Ubuntu

Open Ubuntu using WSL.

Screenshot: Ubuntu Terminal.

Step 2 – Check Available CPU Cores

Command

nproc

Explanation

Displays the number of logical CPU cores available for OpenMP execution.

Screenshot: nproc Output.

Step 3 – Configure OpenMP Threads

Command

export OMP_NUM_THREADS=8

Explanation

Sets the number of OpenMP threads to 8.

Screenshot: Thread Configuration.

Step 4 – Verify Thread Count

Command

echo $OMP_NUM_THREADS

Explanation

Confirms that the OpenMP environment variable is correctly configured.

Screenshot: Output showing 8.

Step 5 – Create Project Directory

Commands

mkdir -p ~/parallel_lab/openmp
cd ~/parallel_lab/openmp

Explanation

Creates a dedicated directory for OpenMP implementation.

Screenshot: Directory Creation.

Step 6 – Create OpenMP Source Code

Command

nano matrix_openmp.c

Explanation

Creates the OpenMP source code containing omp.h and parallel loop directives.

Screenshot: Source Code.

Step 7 – Compile OpenMP Program

Command

gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp

Explanation

Compiles the program with OpenMP support enabled.

Screenshot: Compilation Output.

Step 8 – Execute OpenMP Program

Command

./matrix_openmp

Explanation

Runs matrix multiplication using multiple CPU threads simultaneously.

Screenshot: Program Output.

Step 9 – Monitor CPU Usage

Command

htop

Explanation

Displays CPU utilization and shows multiple CPU cores actively executing the program.

Screenshot: htop Output.

6. Result

The OpenMP Matrix Multiplication program executed successfully.

Observed Output

Matrix Size: 4000 × 4000

Number of Threads: 8

Verification Value: C[0][0] = 4000.00

Execution Time: __________ seconds

The execution time was significantly reduced compared to the sequential implementation.

7. Conclusion

The OpenMP implementation successfully utilized multiple CPU threads to execute matrix multiplication in parallel. The output remained correct while the execution time decreased due to shared-memory parallel processing.
