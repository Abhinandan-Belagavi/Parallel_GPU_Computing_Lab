Part A – Sequential Matrix Multiplication Using

Student Name: Abhinandan Shantinath Belagavi

USN: 01FE24BCI052

1. Aim

To implement Sequential Matrix Multiplication using the C programming language in the WSL Ubuntu environment and measure the execution time for multiplying two 4000 × 4000 matrices.

2. Objective

To understand sequential execution in C programming.

To compile and execute a C program using GCC in Ubuntu WSL.

To perform matrix multiplication using a single CPU execution flow.

To verify the correctness of the output matrix.

To record the execution time for performance comparison with parallel implementations.

3. Theory

Matrix multiplication is a fundamental operation used in scientific computing, machine learning, and image processing. In this experiment, two matrices of size 4000 × 4000 are multiplied using a sequential algorithm written in C. Every element of Matrix A and Matrix B is initialized with the value 1.0. Therefore, each element of the output matrix becomes 4000.00 after multiplication.

The sequential implementation executes using a single CPU core, meaning each matrix element is computed one after another. The execution time obtained from this implementation serves as the baseline for comparing OpenMP, MPI, and CUDA implementations.

4. Procedure
Step 1: Open Windows PowerShell

Open Windows PowerShell from the Start Menu. This terminal is used to verify the WSL installation and launch the Ubuntu environment.

Screenshot: Insert Screenshot 1 here.

Figure 1.1: Opening Windows PowerShell.

Step 2: Verify WSL Installation

Command

wsl --status

Explanation

This command checks whether Windows Subsystem for Linux (WSL) is installed correctly. It displays the WSL version and default Linux distribution available on the system.

Screenshot: Insert Screenshot 2 here.

Figure 1.2: WSL Status Verification.

Step 3: Check Ubuntu Distribution

Command

wsl -l -v

Explanation

This command lists all installed Linux distributions and verifies that Ubuntu is running under WSL2.

Screenshot: Insert Screenshot 3 here.

Figure 1.3: Ubuntu Distribution Verification.

Step 4: Launch Ubuntu Terminal

Command

wsl

Explanation

This command opens the Ubuntu terminal inside WSL. All Linux commands for compilation and execution are performed in this environment.

Screenshot: Insert Screenshot 4 here.

Figure 1.4: Ubuntu Terminal.

Step 5: Update Ubuntu Packages

Command

sudo apt update

Explanation

This command updates the Ubuntu package repository so that the latest packages and compiler tools are available.

Screenshot: Insert Screenshot 5 here.

Figure 1.5: Updating Ubuntu Packages.

Step 6: Install GCC Compiler

Command

sudo apt install build-essential -y

Explanation

The build-essential package installs the GCC compiler, GNU Make, libraries, and other tools required to compile C programs.

Screenshot: Insert Screenshot 6 here.

Figure 1.6: Installing GCC Compiler.

Step 7: Verify GCC Installation

Command

gcc --version

Explanation

This command confirms that GCC has been installed successfully by displaying its version information.

Screenshot: Insert Screenshot 7 here.

Figure 1.7: GCC Version Verification.

Step 8: Create Project Directory

Commands

mkdir -p ~/parallel_lab/sequential
cd ~/parallel_lab/sequential

Explanation

A separate working directory is created for the Sequential Matrix Multiplication experiment. This keeps the source code and executable organized.

Screenshot: Insert Screenshot 8 here.

Figure 1.8: Creating Working Directory.

Step 9: Create Source Code File

Command

nano matrix_sequential.c

Explanation

A C source file is created using the Nano editor. The program initializes matrices, performs sequential matrix multiplication, measures execution time, and verifies the output.

Screenshot: Insert Screenshot 9 here.

Figure 1.9: Sequential Matrix Multiplication Source Code.

Step 10: Compile the Program

Command

gcc -O2 matrix_sequential.c -o matrix_sequential

Explanation

This command compiles the C source code into an executable file named matrix_sequential. The -O2 optimization flag improves execution performance.

Screenshot: Insert Screenshot 10 here.

Figure 1.10: Compilation of Sequential Program.

Step 11: Verify Executable File

Command

ls -l

Explanation

The ls -l command lists files in the current directory and confirms that the executable has been created successfully.

Screenshot: Insert Screenshot 11 here.

Figure 1.11: Executable Verification.

Step 12: Execute the Program

Command

./matrix_sequential

Explanation

The executable performs matrix multiplication using a single CPU core. The output displays matrix size, execution time, and verification value.

Screenshot: Insert Screenshot 12 here.

Figure 1.12: Sequential Program Output.

5. Result

The Sequential Matrix Multiplication program executed successfully in the Ubuntu WSL environment.

Observed Output

Matrix Size: 4000 × 4000

Execution Model: Sequential (Single CPU Core)

Verification Value: C[0][0] = 4000.00

Execution Time: __________ seconds

The program produced the expected verification value and completed successfully.

7. Conclusion

The Sequential Matrix Multiplication experiment was successfully implemented using the GCC compiler inside the Ubuntu WSL environment. The program executed on a single CPU core and produced the correct output. The execution time obtained from this implementation serves as the baseline for comparing parallel implementations.
