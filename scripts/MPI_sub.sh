#!/bin/bash

# Define processes counts
PROCESSES=(2 4 8 16 32 64)
FILE_BASE=$HOME/Documents/University/PSEM-kmeans
FORM_DATE=$(date +"%Y_%m_%d_%H_%M_%S")
OVERALL_OUT_FILE=$FILE_BASE/benchmark_mpi.txt

echo "Benchmark output file" >> $OVERALL_OUT_FILE

# Loop through each thread count
for t in "${PROCESSES[@]}"; do
    # Submit 100 runs for each thread count
    echo "Running with ${t}/64 processes..."
    for i in {1..5}; do
        echo "    Running iteration ${i}/5..."
        CONTENT="\n\n==========================================\n  Processes: ${t}/64 - Iteration: ${i}/5\n==========================================\n\n"

        echo -e $CONTENT >> $OVERALL_OUT_FILE

        condor_submit job.sub \
            -append "executable = $FILE_BASE/KMEANS_mpi.out" \
            -append "arguments = \"$FILE_BASE/test_files/input100D2.inp 70 100000 0.1 0.1 $FILE_BASE/out_mpi_t${t}_run${i}.txt\"" \
            -append "output = $FILE_BASE/out/out_mpi_${t}_run${i}.out" \
            -append "error = $FILE_BASE/logs/err_mpi_${t}_run${i}.err" \
            -append "log = $FILE_BASE/logs/log_mpi_${t}_run${i}.log" \
            -append "request_cpus = ${t}"

        echo $FILE_BASE/out/out_mpi_${t}_run${i}.out >> $OVERALL_OUT_FILE

        # mpirun -n $t --oversubscribe ./KMEANS_mpi.out $FILE_BASE/test_files/input100D.inp 70 100000 0.1 0.1 $FILE_BASE/out_mpi.txt >> $OVERALL_OUT_FILE || true
    done
done
