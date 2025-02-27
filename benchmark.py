import sys

def main(file: str, mode: str, outfile: str) -> None:
    match mode:
        case "mpi":
            mpi_test(file, outfile)
        case "pthreads":
            pass
        case "cuda":
            pass
        case "combined":
            pass
    
    return None

def mpi_test(file: str) -> None:
    processes = [2, 4, 8, 16, 32, 64]
    return None



if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])