# Loop over machine counts
for mc in 5; do
    # Loop over 10 runs for each machine count
    for run in {1..5}; do
        echo "Submitting job: machine_count=$mc, run_id=$run"
        condor_submit job.job -append "machine_count=$mc" -append "run_id=$run"
    done
done

