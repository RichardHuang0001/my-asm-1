#!/bin/bash

# --- Check for correct number of arguments ---
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <num_nodes> <tasks_per_node> <problem_size>"
    exit 1
fi

# --- Assign command-line arguments to variables ---
NUM_NODES=$1
TASKS_PER_NODE=$2
PROBLEM_SIZE=$3
TOTAL_TASKS=$(( NUM_NODES * TASKS_PER_NODE ))

# --- Define job-specific variables ---
JOB_NAME="SOBEL_MPI_${NUM_NODES}n_${TASKS_PER_NODE}t_${PROBLEM_SIZE}"
OUTPUT_FILE="SOBEL_MPI_${NUM_NODES}n_${TASKS_PER_NODE}t_${PROBLEM_SIZE}_%j.out"
EXECUTABLE="./sobel_mpi"

# --- Create the Slurm job script using a heredoc ---
cat <<EOF > "sobel_mpi.job"
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=/uac/lec/ktlam/cmsc5702/demos/${OUTPUT_FILE}
#SBATCH --mail-user=ktlam@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --time=00:10:00           # Wall-clock time limit (e.g., 10 minutes)

#SBATCH --nodes=${NUM_NODES}
#SBATCH --ntasks=${TOTAL_TASKS}
#SBATCH --ntasks-per-node=${TASKS_PER_NODE}

# Create a hostfile based on SLURM allocated nodes
scontrol show hostnames "\$SLURM_NODELIST" | awk '{print \$0" slots=${TASKS_PER_NODE}"}' > hostfile.txt

# Run the MPI program
mpiexec.openmpi --hostfile hostfile.txt -n ${TOTAL_TASKS} ${EXECUTABLE} ${PROBLEM_SIZE}

# Clean up the hostfile
rm hostfile.txt
EOF

# --- Submit the generated job script ---
sbatch -p cmsc5702_hpc -q cmsc5702 sobel_mpi.job
