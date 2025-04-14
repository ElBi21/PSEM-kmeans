#
# K-means 
#
# Parallel computing (Degree in Computer Engineering)
# 2022/2023
#
# (c) 2023 Diego Garcia-Alvarez and Arturo Gonzalez-Escribano
# Grupo Trasgo, Universidad de Valladolid (Spain)
#

# Compilers
CC=gcc
OMPFLAG=-fopenmp# -ffp-contract=off -fno-associative-math -mfma -fno-fast-math -ffloat-store
CUDAFLAGS=--generate-line-info -arch=sm_75
MPICC=mpicc
CUDACC=nvcc
PTHREADS=-lpthread

# Flags for optimization and libs
FLAGS=-O3 -Wall#-fno-omit-frame-pointer -mfma
LIBS=-lm

# Targets to build
OBJS=KMEANS_seq.out KMEANS_omp.out KMEANS_mpi.out KMEANS_cuda.out KMEANS_cuda_f1.out KMEANS_pt.out KMEANS_omp_mpi.out KMEANS_pt_old.out

# Rules. By default show help
help:
	@echo
	@echo "K-means clustering method"
	@echo
	@echo "Group Trasgo, Universidad de Valladolid (Spain)"
	@echo
	@echo "make KMEANS_seq	Build only the sequential version"
	@echo "make KMEANS_omp	Build only the OpenMP version"
	@echo "make KMEANS_mpi	Build only the MPI version"
	@echo "make KMEANS_cuda	Build only the CUDA version"
	@echo
	@echo "make all	Build all versions (Sequential, OpenMP)"
	@echo "make debug	Build all version with demo output for small surfaces"
	@echo "make clean	Remove targets"
	@echo

all: $(OBJS)

KMEANS_seq: KMEANS.c
	$(CC) $(FLAGS) $(DEBUG) $< $(LIBS) -o $@.out

KMEANS_seq_correctness: KMEANS.c
	$(CC) $(FLAGS) -mavx2 -mfma $(DEBUG) $< $(LIBS) -o $@.out

KMEANS_pt: KMEANS_pt.c
	$(CC) $(FLAGS) $(DEBUG) $(PTHREADS) $(LIBS) $< -o $@.out

KMEANS_pt_old: KMEANS_pt_old.c
	$(CC) $(FLAGS) $(DEBUG) $(PTHREADS) $(LIBS) $< -o $@.out

KMEANS_omp: KMEANS_omp.c
	$(CC) $(FLAGS) $(DEBUG) $(OMPFLAG) $< $(LIBS) -o $@.out

KMEANS_mpi: KMEANS_mpi.c
	$(MPICC) $(FLAGS) $(DEBUG) $< $(LIBS) -o $@.out

KMEANS_cuda: KMEANS_cuda.cu
	$(CUDACC) $(CUDAFLAGS) $(DEBUG) $< $(LIBS) -o $@.out

KMEANS_cuda_t: KMEANS_cuda_t.cu
	$(CUDACC) $(CUDAFLAGS) $(DEBUG) $< $(LIBS) -o $@.out

KMEANS_omp_mpi: KMEANS_omp_mpi.c
	$(MPICC) $(FLAGS) $(DEBUG) $(OMPFLAG) $< $(LIBS) -o $@.out

KMEANS_cuda_mpi: KMEANS_cuda_mpi.cu
	$(CUDACC) $(CUDAFLAGS) -lmpi $(DEBUG) $< $(LIBS) -o $@.out

KMEANS_mpi_pthreads: KMEANS_mpi_pthreads.c
	$(MPICC) $(FLAGS) $(DEBUG) $(PTHREADS) $< $(LIBS) -o $@.out

# Remove the target files
clean:
	rm -rf $(OBJS)

# Compile in debug mode
debug:
	make DEBUG="-DDEBUG -g" FLAGS= all
