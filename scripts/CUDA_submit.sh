#!/bin/bash

# Define test files
TEST_FILES=(
    "/home/biason_2045751/k-means/test_files/input100D.inp"
)

# Loop through each test file
for test_file in "${TEST_FILES[@]}"; do
    # Extract filename without path and extension for unique naming
    test_name=$(basename "$test_file" .inp)

    # Submit 10 runs per test file
    for i in {1..10}; do
        condor_submit job.sub \
            -append "executable = /home/biason_2045751/k-means/builds/KMEANS_cuda" \
            -append "arguments = \"$test_file 70 100000 0.001 0.1 /home/biason_2045751/k-means/results/CUDA_${test_name}_run${i}\"" \
            -append "output = /home/biason_2045751/k-means/logs/out/CUDA_${test_name}_run${i}.out" \
            -append "error = /home/biason_2045751/k-means/logs/err/CUDA_${test_name}_run${i}.err" \
            -append "log = /home/biason_2045751/k-means/logs/log/CUDA_${test_name}_run${i}.log" \
            -append "request_cpus = 1" \
            -append "request_gpus = 1"
    done
done
