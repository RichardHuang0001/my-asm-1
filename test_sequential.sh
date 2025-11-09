#!/bin/bash
# test sequential sobel implementation and collect timing data

echo "compiling sobel.c ..."
gcc sobel.c -o sobel -lm -fopenmp -O3
if [ $? -ne 0 ]; then
    echo "compilation failed"
    exit 1
fi
echo "compilation done"

CSV_FILE="sequential_times.csv"
echo "image_size,run1_time,run2_time,run3_time" > $CSV_FILE

SIZES=(256 1024 4000 16000)

echo "starting tests..."

for SIZE in "${SIZES[@]}"; do
    echo "testing ${SIZE}x${SIZE} image"
    
    times=()
    
    for RUN in 1 2 3; do
        echo "  run $RUN"
        
        output=$(./sobel -n $SIZE 2>&1)
        time=$(echo "$output" | grep -Eo "[0-9]+\.[0-9]+")
        
        if [ -z "$time" ]; then
            echo "    error: could not extract time"
            time="0.0"
        else
            echo "    time: $time seconds"
        fi
        
        times+=($time)
        sleep 0.5
    done
    
    echo "$SIZE,${times[0]},${times[1]},${times[2]}" >> $CSV_FILE
done

echo "tests completed, results saved to $CSV_FILE"
echo ""
cat $CSV_FILE
echo ""

echo "summary:"
awk -F',' 'NR>1 {
    size=$1
    avg=($2+$3+$4)/3
    printf "%s: avg time = %.6f seconds\n", size, avg
}' $CSV_FILE