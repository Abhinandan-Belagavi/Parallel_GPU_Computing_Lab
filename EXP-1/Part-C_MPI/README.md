Part C – MPI Distributed Matrix Multiplication
1. Aim

To implement distributed matrix multiplication using MPI across four Ubuntu virtual machines.

2. Objective

To understand distributed-memory parallel programming.

To configure MPI cluster using Master and Worker nodes.

To distribute computation across multiple processes.

To verify distributed execution using MPI communication.

3. Theory

Message Passing Interface (MPI) is a distributed-memory programming model used for parallel execution across multiple systems. In this experiment, four Ubuntu virtual machines communicate over a network. The Master node distributes matrix rows using MPI_Scatter, shares Matrix B using MPI_Bcast, and gathers results using MPI_Gather.

eeeba76e-8a2d-4d59-b2d1-7ae5159180ea.docx
4. Requirements

VMware Workstation

Four Ubuntu Virtual Machines

OpenSSH Server

Open MPI

Network Connectivity

5. Procedure
Step 1 – Create Ubuntu Virtual Machines

Create one Master VM and three Worker VMs connected to the same network.

Screenshot: VMware showing all four VMs.

Step 2 – Configure Hostnames

Command

sudo hostnamectl set-hostname master

Explanation

Assigns unique hostname to each virtual machine.

Screenshot: Hostname Configuration.

Step 3 – Check IP Address

Command

hostname -I

Explanation

Displays IP address assigned to each VM for network communication.

Screenshot: IP Address Output.

Step 4 – Verify Network Connectivity

Command

ping -c 4 worker1

Explanation

Tests communication between Master and Worker nodes.

Screenshot: Ping Output.

Step 5 – Install OpenSSH Server

Commands

sudo apt install openssh-server -y
sudo systemctl enable --now ssh

Explanation

Installs SSH service required for passwordless communication.

Screenshot: SSH Installation.

Step 6 – Install Open MPI

Command

sudo apt install openmpi-bin libopenmpi-dev -y

Explanation

Installs MPI compiler and runtime environment.

Screenshot: MPI Installation.

Step 7 – Verify MPI Installation

Commands

mpicc --version
mpirun --version

Explanation

Verifies successful installation of MPI tools.

Screenshot: MPI Version Output.

Step 8 – Generate SSH Key

Command

ssh-keygen -t rsa

Explanation

Creates SSH key pair for passwordless authentication.

Screenshot: SSH Key Generation.

Step 9 – Copy SSH Key

Command

ssh-copy-id worker1

Explanation

Copies public key to Worker node for secure passwordless login.

Screenshot: SSH Copy Output.

Step 10 – Test SSH Connection

Command

ssh worker1 hostname

Explanation

Confirms passwordless SSH connection between Master and Worker.

Screenshot: SSH Verification.

Step 11 – Create MPI Project Directory

Commands

mkdir -p ~/parallel_lab/mpi
cd ~/parallel_lab/mpi

Explanation

Creates separate directory for MPI implementation.

Screenshot: Directory Creation.

Step 12 – Create Hostfile

Command

nano hosts

Explanation

Creates hostfile listing all participating MPI nodes.

Screenshot: Hostfile Contents.

Step 13 – Create MPI Source Code

Command

nano matrix_mpi.c

Explanation

Creates MPI program using Scatter, Broadcast, Gather, and Finalize functions.

Screenshot: MPI Source Code.

Step 14 – Compile MPI Program

Command

mpicc -O2 matrix_mpi.c -o matrix_mpi

Explanation

Compiles MPI source code using MPI compiler wrapper.

Screenshot: Compilation Output.

Step 15 – Copy Executable to Worker Nodes

Command

scp matrix_mpi worker1:~/matrix_mpi

Explanation

Copies executable to Worker nodes before distributed execution.

Screenshot: SCP Output.

Step 16 – Execute MPI Program

Command

mpirun -np 4 --hostfile hosts sh -c '$HOME/matrix_mpi'

Explanation

Launches four MPI processes across Master and Worker nodes. Each process computes its assigned rows and sends results back to the Master node.

Screenshot: MPI Execution Output.

6. Result

The MPI Matrix Multiplication program executed successfully across four Ubuntu virtual machines.

Observed Output

Matrix Size: 4000 × 4000

Number of MPI Processes: 4

Verification Value: C[0][0] = 4000.00

Execution Time: __________ seconds

The matrix multiplication task was successfully distributed among four MPI processes.

7. Conclusion

The MPI implementation successfully distributed the matrix multiplication workload across four virtual machines using message passing. Each process computed a portion of the matrix independently, and the Master node combined the results into the final output matrix. The program produced the correct verification value and demonstrated distributed-memory parallel computing.
