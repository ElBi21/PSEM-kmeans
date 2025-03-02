#!/bin/bash

TEST_FILES=(
    "/home/venditti_2031589/k-means/test_files/input100D2.inp"
    "/home/venditti_2031589/k-means/test_files/input100D.inp"
    "/home/venditti_2031589/k-means/test_files/input20D.inp"
    "/home/venditti_2031589/k-means/test_files/input10D.inp"
    "/home/venditti_2031589/k-means/test_files/input2D.inp"
)

for test_file in "${TEST_FILES[@]}"; do
    test_name=$(basename "$test_file" .inp)

    for i in {1..10}; do
        condor_submit job.sub \
            -append "executable = /home/venditti_2031589/k-means/builds/KMEANS_seqnof" \
            -append "arguments = \"$test_file 70 100000 0.001 0.1 /home/venditti_2031589/k-means/results/SEQ_${test_name}_run${i}\"" \
            -append "output = /home/venditti_2031589/k-means/logs/outnof/SEQ_${test_name}_run${i}.out" \
            -append "error = /home/venditti_2031589/k-means/logs/err/SEQ_${test_name}_run${i}.err" \
            -append "log = /home/venditti_2031589/k-means/logs/log/SEQ_${test_name}_run${i}.log" \
            -append "request_cpus = 1"
    done
done

