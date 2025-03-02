#!/bin/bash

# Define thread counts
THREADS=(1 2 3 4 5 6 7 8 9 10)

# Loop through each thread count
for t in "${THREADS[@]}"; do
    # Submit 100 runs for each thread count
    for i in {1..10}; do
        condor_submit job.sub \
            -append "executable = /home/venditti_2031589/k-means/builds/KMEANS_pt" \
            -append "arguments = \"/home/venditti_2031589/k-means/test_files/input100D2.inp 70 100000 0.1 0.1 /home/venditti_2031589/k-means/results/PTHREADS_t${t}_run${i} ${t}\"" \
            -append "output = /home/venditti_2031589/k-means/logs/out/Pt${t}_run${i}.out" \
            -append "error = /home/venditti_2031589/k-means/logs/err/Pt${t}_run${i}.err" \
            -append "log = /home/venditti_2031589/k-means/logs/log/Pt${t}_run${i}.log" \
            -append "request_cpus = ${t}"
    done
done
