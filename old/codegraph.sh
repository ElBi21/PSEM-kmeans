#!/usr/bin/zsh

local execution_time_date="$(date '+%F_%H-%M-%S')"

###################################

# Edit these parameters

local in_file="test_files/input100D2.inp"
local clusters_amount=100
local iterations_amount=1000
local changes_max=5000
local threshold=1
local output_data_file="out/KMEANS-$execution_time_date.txt"

###################################


# Main function
generate_graph() {
    local perf_time_exec="$2"
    local to_run="$1"
    local to_run_path=""
    local to_make=""

    # Run the code
    case $to_run in
        "seq")
            echo "[OK] Selected the sequential version. Search for the file..."
            to_run_path="KMEANS_seq.out"
            to_make="KMEANS_seq"
            file_search
            ;;

        "mpi")
            echo "[OK] Selected the MPI version. Search for the file..."
            to_run_path="KMEANS_mpi.out"
            to_make="KMEANS_mpi"
            file_search
            ;;

        "omp")
            echo "[OK] Selected the OpenMP version. Search for the file..."
            to_run_path="KMEANS_omp.out"
            to_make="KMEANS_omp"
            file_search
            ;;

        "cuda")
            echo "[OK] Selected the CUDA version. Search for the file..."
            to_run_path="KMEANS_cuda.out"
            to_make="KMEANS_cuda"
            file_search
            ;;

        "pth")
            echo "Tocca ancora farlo lol"
            exit
            ;;

        *)
            echo "Only one value between 'seq', 'mpi', 'omp', 'pth' or 'cuda' is allowed"
            ;;
    esac


    # Start recording on the system
    echo "[OK] Starting to record for $perf_time_exec seconds..."
    echo $to_run_path
    perf record -F 99 -a -g -o "perf.data" -- sleep $perf_time_exec &
    $to_run_path $in_file $clusters_amount $iterations_amount $changes_max $threshold $output_data_file

    wait
    perf script | ./FlameGraph/stackcollapse-perf.pl > out.perf-folded
    ./FlameGraph/flamegraph.pl out.perf-folded > "$graph_title.svg"
    #rm out/out.perf-folded

    echo "[OK] Completed the execution. SVG file created in the out folder"
    echo "[OK] Moving the files into the out folder and granting write permissions to the user..."

    chmod u+rw perf.data
    chmod u+rw out.perf-folded

    mv "$graph_title.svg" "out/$graph_title.svg"
    mv "perf.data" "out/KMEANS-$execution_time_date-perf.data"
    mv "out.perf-folded" "out/KMEANS-$execution_time_date-out.perf-folded"

    echo "[OK] Finished setting everything up! Enjoy!"
}


file_search() {
    if [ -f "$to_run_path" ]; then
        echo "[OK] The file exists. Proceed to measure the timings..."
    else
        echo "[ER] File doesn't exist. Proceeding to make it..."
        make $to_make
        if [ $? -eq 0 ]; then
            echo "[OK] File built successfully. Proceeding to measure the timings..."
        else
            echo "[ER] Fatal error: couldn't build the file. Stopping..."
            exit
        fi
    fi
    to_run_path="./$to_run_path"
}


help_output() {
    echo "No more than 3 parameters are needed."
    echo "The correct syntax is the following:\n"
    echo "  ./codegraph <file_version> <time_to_run> <output_file_name>\n\n"
    echo "where:\n      file_version: a value between 'seq', 'mpi', 'omp', 'pth' or 'cuda'\n"
    echo "       time_to_run: the time in seconds to capture data from the system\n"
    echo "  output_file_name: the name of the output SVG file. Optional"

    exit
}



# Will be executed

if [[ "$1" == "help" ]] || {[ "$#" -gt 3 ] || [ "$#" -lt 2 ]}; then
    help_output
fi

if [ ! -d "out" ]; then
    echo "[OK] Folder 'out' got created"
    mkdir out
fi

# If there is no title parameter, put the date
local graph_title="KMEANS-$execution_time_date"

if [ "$#" -eq 3 ]; then
    graph_title="$3"
fi


generate_graph "$1" "$2" "$3"