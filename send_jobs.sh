#!/bin/bash

make KMEANS_omp_mpi
make KMEANS_mpi
make KMEANS_seq

file_test="input20D.inp"
#test_to_do="test_files/$file_test 30 500 0.1 0.1 outputs/out_omp_mpi_32.txt"
#thread_array=(1 2 4 8 16 32)
#process_array=(1 2 4)

test_to_do="test_files/$file_test 30 500 0.1 0.1 outputs/out_mpi.txt"
thread_array=(1)
process_array=(1 2 4 8 16 32 64)

#test_to_do="test_files/$file_test 30 500 0.1 0.1 outputs/out_seq.txt"
#thread_array=(1)
#process_array=(1)

for proc in ${process_array[@]}; do
    for thread in ${thread_array[@]}; do
        for run in {1..30}; do
            echo "[Pr$proc] [Thr$thread] Starting run $run/30"

            cat slurm_batch_base.sh > slurm_batch.sh

            # echo -e "#SBATCH --output=\"logs/slurm/mpi_omp_${file_test}_p${proc}_t${thread}_run_${run}.txt\"\n" >> slurm_batch.sh
            echo -e "#SBATCH --output=\"logs/slurm/mpi_${file_test}_p${proc}_t1_run_${run}.txt\"\n" >> slurm_batch.sh
            # echo -e "#SBATCH --output=\"logs/slurm/seq_${file_test}_run_${run}.txt\"\n" >> slurm_batch.sh

            # echo "srun mpirun -np $proc --oversubscribe KMEANS_omp_mpi.out $test_to_do $thread > \"logs/slurm/mpi_omp_${file_test}_p${proc}_t${thread}_run_${run}.txt\"" >> slurm_batch.sh
            echo "srun mpirun -np $proc --oversubscribe KMEANS_mpi.out $test_to_do > \"logs/slurm/mpi_${file_test}_p${proc}_t1_run_${run}.txt\"" >> slurm_batch.sh
            # echo "srun KMEANS_seq.out $test_to_do > \"logs/slurm/seq_${file_test}_run_${run}.txt\"" >> slurm_batch.sh

            sbatch slurm_batch.sh
        done
    done
done