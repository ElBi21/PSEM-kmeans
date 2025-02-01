import subprocess
import re
import numpy as np
import pandas as pd

# Define input files and process counts
input_files = [
    "input2D.inp",
    "input2D2.inp",
    "input10D.inp",
    "input20D.inp",
    "input100D.inp",
    "input100D2.inp"
]
process_counts = [5, 7, 8, 9, 10, 11, 12]

# Regular expression to extract computation time for Rank 0
time_pattern = r"Computation: ([0-9]+\.[0-9]+) seconds"

# Function to execute the command and collect computation times
def run_experiment(input_file, num_processes):
    command = f"mpirun --oversubscribe -n {num_processes} ./KMEANS_mpi.out test_files/{input_file} 40 5000 1 0.0001 output_files/out_{input_file}_mpi.txt"
    computation_times = []

    for process in range(50):
        try:
            # Execute the command and capture the output
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            # Extract the computation time from the output
            match = re.search(time_pattern, result.stdout)
            if match:
                computation_times.append(float(match.group(1)))
            print(f"Finished process {process} with {num_processes} processes and input file {input_file}")
        except Exception as e:
            print(f"Error for {input_file} with {num_processes} processes: {e}")
    
    return computation_times

# Collect data and calculate both mean and standard deviation
data_stats = {}
for input_file in input_files:
    for num_processes in process_counts:
        times = run_experiment(input_file, num_processes)
        if times:
            mean_time = np.mean(times)
            std_dev_time = np.std(times)
            data_stats[(num_processes, input_file)] = (mean_time, std_dev_time)

# Build the results table with both mean and standard deviation
table_data_stats = []
for num_processes in process_counts:
    row = [num_processes] + [
        f"{data_stats.get((num_processes, file), ('N/A', 'N/A'))[0]:.6f} ± {data_stats.get((num_processes, file), ('N/A', 'N/A'))[1]:.6f}"
        if data_stats.get((num_processes, file)) else "N/A"
        for file in input_files
    ]
    table_data_stats.append(row)

# Create and display the DataFrame
columns = ["Processes"] + input_files
results_stats_df = pd.DataFrame(table_data_stats, columns=columns)

# Display the table (requires a compatible environment like Jupyter Notebook)
print(results_stats_df)

