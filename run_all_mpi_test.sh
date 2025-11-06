#!/bin/bash
# --- Configuration ---
SIZES=(256 1024 4000 16000)
# For 1 node: processes = 1, 2, 4 (== tasks_per_node)
# For 4 nodes: processes = 8, 16, 32 (tasks_per_node = processes/4)
declare -a COMBOS=("1 1" "1 2" "1 4" "4 2" "4 4" "4 8")

# Output CSV
CSV="report/mpi_times.csv"
echo "image_size,nodes,processes,run1_time,run2_time,run3_time" > "$CSV"

# --- Helper function: run and extract time ---
run_one_case() {
    local nodes=$1
    local tpn=$2
    local psize=$3
    local runs=3
    local total=$((nodes * tpn))
    local times=()

    echo ">>> Running image=${psize}, nodes=${nodes}, total_tasks=${total}"

    for r in $(seq 1 $runs); do
        # Submit job using existing sbatch script
        job_output=$(./sbatch_sobel_mpi.sh ${nodes} ${tpn} ${psize})
        jobid=$(echo "$job_output" | grep -oE '[0-9]+$')
        if [ -z "$jobid" ]; then
            echo "  [Error] Failed to get jobid for ${nodes}n ${tpn}t ${psize}"
            continue
        fi
        echo "  Run ${r}: submitted job ${jobid}"

        # Wait for job to finish
        while squeue -j ${jobid} 2>/dev/null | grep -q ${jobid}; do
            sleep 2
        done

        outfile="/uac/msc/whuang25/cmsc5702/SOBEL_MPI_${nodes}n_${tpn}t_${psize}_${jobid}.out"

        if [ ! -f "$outfile" ]; then
            echo "  [Warn] Output file not found: $outfile"
            times+=("NaN")
            continue
        fi

        # Extract execution time
        t=$(grep -oP 'MPI版本执行时间:\s*\K[0-9]+\.[0-9]+' "$outfile" | tail -n1)
        if [ -z "$t" ]; then
            t=$(grep -oE '[0-9]+\.[0-9]+' "$outfile" | tail -n1)
        fi
        times+=("$t")
        echo "  Run ${r}: time = ${t}s"
    done

    # Append to CSV
    echo "${psize},${nodes},${total},${times[0]},${times[1]},${times[2]}" >> "$CSV"
    echo ">>> Done image=${psize}, nodes=${nodes}, total=${total}"
    echo
}

# --- Main loop ---
for size in "${SIZES[@]}"; do
  for combo in "${COMBOS[@]}"; do
    n=$(echo $combo | awk '{print $1}')
    tpn=$(echo $combo | awk '{print $2}')
    run_one_case $n $tpn $size
  done
done

echo "All done. Results saved to $CSV"
