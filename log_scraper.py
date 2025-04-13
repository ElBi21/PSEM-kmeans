import os
import re
import json

def main_htcondor(out_file: str) -> None:
    print(f"Starting to scrape...\nWill save in {out_file}")
    root = "logs/htcondor/out"
    pattern = re.compile(" Computation: .* seconds")
    files = os.listdir(root)
    results = dict()
    found = 0

    """prev_json = check_json(out_file)
    if prev_json != None:
        for key, value in prev_json:
            results[key] = value"""

    for file_path in files:
        path = os.path.join(root, file_path)
        run_params = file_path.split("_")[1:]
        run_params.pop(2)

        run_str = f"{run_params[0]}_{run_params[1]}"

        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                if pattern.search(line):
                    line = line[14:-9]

                    if run_str not in results.keys():
                        results[run_str] = []
                    
                    results[run_str].append(float(line))
                    found += 1

    with open(out_file, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Saved in total {found} results")
    return None


def main_slurm(out_file: str, check_for: str) -> None:
    print(f"Starting to scrape...\nWill save in {out_file}")
    root = "logs/slurm"
    pattern = re.compile(" Computation: .* seconds")
    files = os.listdir(root)
    results = dict()
    found = 0

    for file_path in files:
        if file_path.startswith(check_for) and not os.path.isdir(file_path): #and file_path.endswith("_2.txt"):#and file_path.__contains__("_t1_run"):
            path = os.path.join(root, file_path)
            run_params = file_path.split("_")[2:-2]
            print(run_params)
            run_str = "seq"

            if check_for != "seq":
                run_str = f"{run_params[0]}_{run_params[1]}"

            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    if pattern.search(line):
                        line = line[14:-9]

                        if run_str not in results.keys():
                            results[run_str] = []
                        
                        results[run_str].append(float(line))
                        found += 1

    with open(out_file, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Saved in total {found} results")
    return None

def seq_slurm(out_file: str, check_for: str) -> None:
    print(f"Starting to scrape...\nWill save in {out_file}")
    root = "logs/slurm"
    pattern = re.compile("Computation: .* seconds")
    files = os.listdir(root)
    results = dict()
    found = 0

    for file_path in files:
        if file_path.startswith(check_for): #and file_path.endswith("_2.txt"):#and file_path.__contains__("_t1_run"):
            path = os.path.join(root, file_path)
            run_str = "seq"

            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    if pattern.search(line):
                        line = line[13:-9]

                        if run_str not in results.keys():
                            results[run_str] = []
                        
                        results[run_str].append(float(line))
                        found += 1

    with open(out_file, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Saved in total {found} results")
    return None


def check_json(out_file: str):
    if os.path.isfile(out_file):
        with open(out_file, "r") as file:
            json_cont = json.load(file)
            print(json_cont)
            return json_cont
    return None


if __name__ == "__main__":
    #main_slurm("mpi_omp_scrape_20d_slurm.json", "mpi_omp_input20D.inp")
    #seq_slurm("seq_20d_slurm.json", "seq_input20D.inp")
    main_slurm("mpi_omp_scrape_100d_slurm.json", "mpi_omp_input100d")
    #print(check_json("mpi_scrape_100d4.json"))