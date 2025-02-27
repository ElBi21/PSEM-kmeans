/*
 * k-Means clustering algorithm
 *
 * CUDA version
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2022/2023
 *
 * Version: 1.0
 *
 * (c) 2022 Diego García-Álvarez, Arturo Gonzalez-Escribano
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <cuda.h>
#include <mpi.h>


#define MAXLINE 2000
#define MAXCAD 200

/*	Important assumption: each process will run on a different GPU. That is, no two (or more) processes will share
 *	the same GPU. In order to make the sharing possible, change the value here below accordingly:
 *	
 *		- 0: all processes will run on the same GPU
 *		- 1: all processes will run on different GPUs. Mind that the following must hold:
 *	
 * 				|processes| = |GPUs|
 *	
 **/

#define SINGLE_GPU_PER_PROCESS 0

//Macros
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/*
 * Macros to show errors when calling a CUDA library function,
 * or after launching a kernel
 */

#define CHECK_CUDA_CALL(a) { \
	cudaError_t ok = a; \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA call in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}

#define CHECK_CUDA_LAST()	{ \
	cudaError_t ok = cudaGetLastError(); \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA last in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}

#define CHECK_MPI_CALL(a) { \
	int result = a; \
    if (result != MPI_SUCCESS) { \
        fprintf(stderr, "[ERROR] Fatal error with MPI in line %d. Aborting\n", __LINE__); \
        MPI_Abort(MPI_COMM_WORLD, a); \
    } \
}

/* 
Function showFileError: It displays the corresponding error during file reading.
*/
void showFileError(int error, char* filename)
{
	printf("Error\n");
	switch (error)
	{
		case -1:
			fprintf(stderr,"\tFile %s has too many columns.\n", filename);
			fprintf(stderr,"\tThe maximum number of columns has been exceeded. MAXLINE: %d.\n", MAXLINE);
			break;
		case -2:
			fprintf(stderr,"Error reading file: %s.\n", filename);
			break;
		case -3:
			fprintf(stderr,"Error writing file: %s.\n", filename);
			break;
	}
	fflush(stderr);	
}

/* 
Function readInput: It reads the file to determine the number of rows and columns.
*/
int readInput(char* filename, int *lines, int *D)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int contlines, contsamples = 0;
    
    contlines = 0;

    if ((fp=fopen(filename,"r"))!=NULL)
    {
        while(fgets(line, MAXLINE, fp)!= NULL) 
		{
			if (strchr(line, '\n') == NULL)
			{
				return -1;
			}
            contlines++;       
            ptr = strtok(line, delim);
            contsamples = 0;
            while(ptr != NULL)
            {
            	contsamples++;
				ptr = strtok(NULL, delim);
	    	}	    
        }
        fclose(fp);
        *lines = contlines;
        *D = contsamples;  
        return 0;
    }
    else
	{
    	return -2;
	}
}

/* 
Function readInput2: It loads data from file.
*/
int readInput2(char* filename, float* data)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int i = 0;
    
    if ((fp=fopen(filename,"rt"))!=NULL)
    {
        while(fgets(line, MAXLINE, fp)!= NULL)
        {         
            ptr = strtok(line, delim);
            while(ptr != NULL)
            {
            	data[i] = atof(ptr);
            	i++;
				ptr = strtok(NULL, delim);
	   		}
	    }
        fclose(fp);
        return 0;
    }
    else
	{
    	return -2; //No file found
	}
}

/* 
Function writeResult: It writes in the output file the cluster of each sample (point).
*/
int writeResult(int *classMap, int lines, const char* filename)
{	
    FILE *fp;
    
    if ((fp=fopen(filename,"wt"))!=NULL)
    {
        for(int i=0; i<lines; i++)
        {
        	fprintf(fp,"%d\n",classMap[i]);
        }
        fclose(fp);  
   
        return 0;
    }
    else
	{
    	return -3; //No file found
	}
}

/*
Function initCentroids: This function copies the values of the initial centroids, using their 
position in the input data structure as a reference map.
*/
void initCentroids(const float *data, float* centroids, int* centroidPos, int D, int K)
{
	int i;
	int idx;
	for(i = 0; i < K; i++) {
		idx = centroidPos[i];
		memcpy(&centroids[i * D], &data[idx * D], (D * sizeof(float)));
	}
}

/*
Function euclideanDistance: Euclidean distance
This function could be modified
*/
__device__ void euclideanDistance(float *point, float *center, int D, float* return_addr) {
	float dist = 0.0;
	for (int i = 0; i < D; i++) {
		dist += (point[i] - center[i]) * (point[i] - center[i]);
	}
	*return_addr = sqrt(dist);
}

/*
Function zeroFloatMatriz: Set matrix elements to 0
This function could be modified
*/
void zeroFloatMatriz(float *matrix, int rows, int columns) {
	int i, j;
	for (i = 0; i < rows; i++) {
		for (j = 0; j < columns; j++) {
			matrix[i * columns + j] = 0.0;
		}
	}
}

/*
Function zeroIntArray: Set array elements to 0
This function could be modified
*/
void zeroIntArray(int *array, int size) {
	int i;
	for (i = 0; i < size; i++) {
		array[i] = 0;
	}
}


/* 
 *		CUDA Kernels and Variables
 */

__constant__ int gpu_K;
__constant__ int gpu_n;
__constant__ int gpu_d;
__constant__ int gpu_size;
__constant__ int gpu_rank;

/*  To each thread, a point with D dimensions gets assigned. The thread must compute the
 *  l_2 norm and take the minimum. Then, for each such point, get the associated cluster,
 * 	and count the number of points for each cluster. Then, for each point, sum its
 * 	coordinates into a matrix which is used for doing the average of the coordinates.
 * 	After that, average all the coordinates and check the maximum distance that changed.
 *
 *  Parameters:
 * 		- `data`: array of points, on the GPU;
 * 		- `centroids`: array of centroids, on the GPU;
 * 		- `class_map`: array with the classes, on the GPU.
 * 		- `changes_return`: address to which the total changes should be written on;
 * 		- `centroids_table`: pointer to the table for storing the centroids dimensions, on the GPU;
 * 		- `points_per_class`: pointer to the table storing the amount of points for each class, on the GPU;
 * 
 *  Returns:
 * 		- `NULL`
 */
__global__ void step_1_kernel(float* data, float* centroids, int* points_per_class, float* centroids_table, int* class_map, int* changes_return) {
	// Compute thread index
	int thread_index = (blockIdx.y * gridDim.x * blockDim.x * blockDim.y) + (blockIdx.x * blockDim.x * blockDim.y) +
							(threadIdx.y * blockDim.x) +
							threadIdx.x;

	/*extern __shared__ float shared_centroids[];	// K x D x sizeof(float)
	//__shared__ int ppc[gpu_K];

	// Define block size and local thread index (index within block)
	int block_size = blockDim.x * blockDim.y;
	int local_thread_index = threadIdx.x + threadIdx.y * blockDim.x;

	// Copy centroids data into shared memory
	for (int portion = 0; portion < (gpu_K * gpu_d) / block_size; portion++) {
		int copy_index = local_thread_index + portion * block_size;
		shared_centroids[copy_index] = centroids[copy_index];
	}*/

	if (thread_index < (gpu_n / gpu_size)) {
		if (thread_index < gpu_K)
			printf("PPC: %d\n", points_per_class[thread_index]);

		int data_index = thread_index * gpu_d;
		int class_int = class_map[thread_index];
		float min_dist = FLT_MAX;
		
		// For each centroid...
		for (int centroid = 0; centroid < gpu_K; centroid++) {
			float distance = 0.0f;

			// Compute the euclidean distance
			euclideanDistance(&data[data_index], &centroids[centroid * gpu_d], gpu_d, &distance);

			// If distance is smaller, replace the distance and assign new class
			if (distance < min_dist) {
				min_dist = distance;
				class_int = centroid + 1;
			}
		}

		// If the class is different, add one change and write new class
		if (class_map[thread_index] != class_int) {
			atomicAdd(changes_return, 1);
		}
		
		// Map the value to the class map
		class_map[thread_index] = class_int;

		int class_assignment = class_map[thread_index];
		int point_index = class_assignment - 1;

		if (thread_index < gpu_K)
			printf("PPC POST: %d, class: %d\n", *(points_per_class + point_index), class_assignment);

		// Atomically increase the number of points for the given class
		atomicAdd(&(points_per_class[point_index]), 1);
		//atomicAdd(ppc + point_index, 1);

		if (thread_index < gpu_K)
			printf("PPC ATOMIC: %d, class: %d\n", points_per_class[point_index], class_assignment);

		for (int dim = 0; dim < gpu_d; dim++) {
			int index = point_index * gpu_d + dim;
			atomicAdd(&centroids_table[index], data[data_index + dim]);
		}
	}
}


/*  To each thread, assign a centroid. The coordinates of each centroid get averaged, and then the
 *  l_2 norm gets computed in order to check the max distance
 *  
 *  
 *  Parameters:
 * 		- `centroids_table`: a table with all the temporary new coordinates of the centroids, on the GPU;
 * 		- `centroids`: array with the centroids, on the GPU;
 * 		- `points_per_class`: a table enumerating how many points have been assigned for each class, on the GPU;
 * 		- `dimensions`: the number of dimensions of each point;
 *  
 *  Returns:
 * 		- `NULL`
 */
__global__ void step_2_kernel(float* aux_centroids, float* centroids, int* points_per_class, float* max_distance) {
	// Index of the thread
	int thread_index = (blockIdx.y * gridDim.x * blockDim.x * blockDim.y) + (blockIdx.x * blockDim.x * blockDim.y) +
							(threadIdx.y * blockDim.x) +
							threadIdx.x;
	
	if (thread_index < (gpu_K / gpu_size)) {
		float distance = 0.0f;
		for (int d = 0; d < gpu_d; d++) {
			aux_centroids[thread_index * gpu_d + d] /= (float) points_per_class[thread_index];
			// Compute Euclidean distance (l_2 norm) to check for maximum distance
			distance += pow((centroids[thread_index * gpu_d + d] - aux_centroids[thread_index * gpu_d + d]), 2);
		}

		// Perform sqrt of distance
		distance = sqrt(distance);

		if (distance > *max_distance) {
			// Exchange atomically, disregard old value
			atomicExch(max_distance, distance);
		}
	}
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

	//START CLOCK***************************************
	clock_t start, end;
	start = MPI_Wtime();
	//**************************************************
	
	// Constant variables
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(gpu_size, &size, sizeof(int)));
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(gpu_rank, &rank, sizeof(int)));

    // lines = number of points; D = number of dimensions per point
    int N = 0, D = 0;
	float* points;

    // If your rank is 0...
    if (rank == 0) {
        // ...initialize the data
        if (argc !=  7) {
            fprintf(stderr,"EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
            fprintf(stderr,"./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
            fflush(stderr);
            exit(-1);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // Reading the input data
        int error = readInput(argv[1], &N, &D);
        if (error != 0) {
            showFileError(error,argv[1]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        
        points = (float*) calloc(N * D, sizeof(float));
        if (points == NULL) {
            fprintf(stderr, "Memory allocation error.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        error = readInput2(argv[1], points);
        if (error != 0) {
            showFileError(error,argv[1]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
	}

	// Broadcast N and D
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(gpu_n, &N, sizeof(int)));
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(gpu_d, &D, sizeof(int)));

        
	// Parameters from the input args
	int K = atoi(argv[2]);
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(gpu_K, &K, sizeof(int)));


	// Convergence values
	int maxIterations = atoi(argv[3]);
	int minChanges = (int) (N * atof(argv[4]) / 100.0);
	float maxThreshold = atof(argv[5]);

	// GPU allocation values
	int data_size = N * D * sizeof(float);
	int centroids_size = K * D * sizeof(float);
	int local_centroids_size = centroids_size / size;
	
	// Allocate centroids in GPU
	float* centroids = (float*) calloc(K * D, sizeof(float));
	
	if (centroids == NULL) {
		fprintf(stderr, "Memory allocation error.\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		exit(-4);
	}

	float* gpu_centroids;
	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_centroids, centroids_size));

	int* classMap = NULL;

	if (rank == 0) {
		int* centroidPos = (int*) calloc(K, sizeof(int));
		classMap = (int*) calloc(N, sizeof(int));

		if (centroidPos == NULL || classMap == NULL) {
			fprintf(stderr, "Memory allocation error.\n");
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}

		// Initial centrodis
		srand(0);
		for (int i = 0; i < K; i++) 
			centroidPos[i] = rand() % N;
		
		// Loading the array of initial centroids with the data from the array data
		// The centroids are points stored in the data array.
		initCentroids(points, centroids, centroidPos, D, K);
		free(centroidPos);

		// Program properties
		printf("\n    Input properties:");
		printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], N, D);
		printf("\tNumber of clusters: %d\n", K);
		printf("\tMaximum number of iterations: %d\n", maxIterations);
		printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), N);
		printf("\tMaximum centroid precision: %f\n", maxThreshold);

		// Check CUDA device properties
		cudaDeviceProp cuda_prop;

		CHECK_CUDA_CALL(cudaGetDeviceProperties(&cuda_prop, 0));

		printf("\n    Device: %s\n", cuda_prop.name);
		printf("\tCompute Capability: %d.%d\n", cuda_prop.major, cuda_prop.minor);
		printf("\tMax threads / block: %d\n", cuda_prop.maxThreadsPerBlock);
		printf("\tMax threads / SM: %d\n", cuda_prop.maxThreadsPerMultiProcessor);
		printf("\tMax blocks / SM: %d\n", cuda_prop.maxBlocksPerMultiProcessor);
		printf("\tMax grid size: %d x %d x %d\n", cuda_prop.maxGridSize[0], cuda_prop.maxGridSize[1], cuda_prop.maxGridSize[2]);
		printf("\tMax shared memory per SM: %dB\n", cuda_prop.sharedMemPerMultiprocessor);
		printf("\tNumber of SMs: %d\n", cuda_prop.multiProcessorCount);
		printf("\tStarting with following grids and blocks:\n");
		printf("\t    Blocks: 32 x 32\n");
		printf("\t    Grid for points: %d x (32 x 32)\n", N / (32 * 32) + 1);
		printf("\t    Grid for centroids: %d x (32 x 32)\n", K / (32 * 32) + 1);
	}

	MPI_Bcast(centroids, K * D, MPI_FLOAT, 0, MPI_COMM_WORLD);
	CHECK_CUDA_CALL(cudaMemcpy(gpu_centroids, centroids, centroids_size, cudaMemcpyHostToDevice));
	
	//END CLOCK*****************************************
	end = MPI_Wtime();
	printf("\nMemory allocation: %f seconds\n", end - start);
	fflush(stdout);

	// Check assumption on top of file for better explanation on this part
	#if SINGLE_GPU_PER_PROCESS == 0 
		CHECK_CUDA_CALL( cudaSetDevice(0) );
	#elif SINGLE_GPU_PER_PROCESS == 1
		CHECK_CUDA_CALL( cudaSetDevice(rank) );
	#else
		printf("Invalid SINGLE_GPU_PER_PROCESS value. Aborting\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		exit(-4);
	#endif

	CHECK_CUDA_CALL( cudaDeviceSynchronize() );
	MPI_Barrier(MPI_COMM_WORLD);

	//**************************************************
	//START CLOCK***************************************
	start = MPI_Wtime();
	//**************************************************

	char* output_msg = (char*) calloc(100000, sizeof(char));
	char* line = (char*) calloc(100, sizeof(char));

	// Loop variables
	int it = 0;
	int changes;
	float maxDist = FLT_MIN;

	// Allocate memory
	int* pointsPerClass = (int*) malloc(K * sizeof(int)); 
	float* auxCentroids = (float*) malloc(K * D * sizeof(float)); 
	int* local_pointsPerClass = (int*) calloc(K / size, sizeof(int));
	float* local_auxCentroids = (float*) calloc(K * D / size, sizeof(float));
	if (pointsPerClass == NULL || auxCentroids == NULL || local_pointsPerClass == NULL || local_auxCentroids == NULL) {
		fprintf(stderr, "Memory allocation error.\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	float* gpu_auxCentroids;
	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_auxCentroids, centroids_size * sizeof(float)));
	CHECK_CUDA_CALL(cudaMemset(gpu_auxCentroids, 0, centroids_size));

	int* gpu_pointsPerClass;
	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_pointsPerClass, K * sizeof(int)));

	// Allocation of local points and local class map
	int local_n = N / size;
	float* local_points = (float*) calloc(local_n * D, sizeof(float));
	int* local_classMap = (int*) calloc(local_n, sizeof(int));
	if (local_points == NULL || local_classMap == NULL) {
		fprintf(stderr, "Memory allocation error.\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}
	
	float* gpu_local_points;
	int* gpu_local_classMap;
	
	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_local_points, data_size / size));
	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_local_classMap, N / size * sizeof(int)));
	CHECK_CUDA_CALL(cudaMemset(gpu_local_classMap, 0, N / size * sizeof(int)));

	// Scatter the data and allocate on the GPU
	MPI_Scatter(points, (N / size) * D, MPI_FLOAT, local_points, (N / size) * D, MPI_FLOAT, 0, MPI_COMM_WORLD);
	CHECK_CUDA_CALL(cudaMemcpy(gpu_local_points, local_points, data_size / size, cudaMemcpyHostToDevice));


	int local_K = K / size;
	float* local_centroids = (float*) calloc(local_K * D, sizeof(float));
	if (local_centroids == NULL) {
		fprintf(stderr, "Memory allocation error.\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	float* gpu_local_centroids;
	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_local_centroids, centroids_size));
	// Might remove
	CHECK_CUDA_CALL(cudaMemset(gpu_local_centroids, 0, local_centroids_size));



	//  ###################################
    //              CUDA Section
    //  ###################################

	// Set carveout to be of maximum size available
	int carveout = cudaSharedmemCarveoutMaxShared;

	CHECK_CUDA_CALL(cudaFuncSetAttribute(step_1_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, carveout));

	// CUDA related variables
	int pts_grid_size = (N / size) / (32 * 32) + 1;
	int K_grid_size = (K / size) / (32 * 32) + 1;

	dim3 gen_block(32, 32);
	dim3 dyn_grid_pts(pts_grid_size);
	dim3 dyn_grid_cent(K_grid_size);

	int* gpu_local_changes;
	float* gpu_local_maxDistance;

	float* gpu_local_auxCentroids;
	int* gpu_local_pointsPerClass;

	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_local_changes, sizeof(int)));
	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_local_maxDistance, sizeof(float)));
	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_local_auxCentroids, K * D * sizeof(float)));
	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_local_pointsPerClass, K / size * sizeof(int)));


	do {
		it++;

		// Reset the variables
		CHECK_CUDA_CALL(cudaMemset(gpu_local_changes, 0, sizeof(int)));
		CHECK_CUDA_CALL(cudaMemset(gpu_pointsPerClass, 0, K * sizeof(int)));
		CHECK_CUDA_CALL(cudaMemset(gpu_auxCentroids, 0, K * D * sizeof(float)));

		CHECK_CUDA_CALL(cudaDeviceSynchronize());

		// First step of the algorithm, point based
		step_1_kernel<<<dyn_grid_pts, gen_block, centroids_size>>>(gpu_local_points, gpu_centroids, gpu_pointsPerClass, gpu_auxCentroids, gpu_local_classMap, gpu_local_changes);
		CHECK_CUDA_LAST();

		int local_changes = 0;
		CHECK_CUDA_CALL(cudaMemcpy(&local_changes, gpu_local_changes, sizeof(int), cudaMemcpyDeviceToHost));

		// Reduce and send to all the sum of all changes
		MPI_Allreduce(&local_changes, &changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		CHECK_CUDA_CALL(cudaMemcpy(pointsPerClass, gpu_pointsPerClass, K * sizeof(int), cudaMemcpyDeviceToHost));
		CHECK_CUDA_CALL(cudaMemcpy(auxCentroids, gpu_auxCentroids, K * D * sizeof(float), cudaMemcpyDeviceToHost));

		CHECK_CUDA_CALL(cudaMemset(gpu_local_maxDistance, FLT_MIN, sizeof(float)));

		CHECK_CUDA_CALL(cudaDeviceSynchronize());

		MPI_Allreduce(MPI_IN_PLACE, pointsPerClass, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, auxCentroids, K * D, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

		
		// Load part of pointsPerClass, auxCentroids and centroids into their local corrispectives
		memcpy(local_pointsPerClass, pointsPerClass + (K / size * rank), (K / size) * sizeof(int));
		memcpy(local_auxCentroids, auxCentroids + (K * D / size * rank), (K * D / size) * sizeof(float));
		memcpy(local_centroids, centroids + (K * D / size * rank), (K * D / size) * sizeof(float));

		// Load into gpu_local_pointsPerClass (and gpu_local_auxCentroids) a portion of pointsPerClass
		CHECK_CUDA_CALL(cudaMemcpy(gpu_local_pointsPerClass, local_pointsPerClass, (K / size) * sizeof(int), cudaMemcpyHostToDevice));
		CHECK_CUDA_CALL(cudaMemcpy(gpu_local_auxCentroids, local_auxCentroids, (K * D / size) * sizeof(float), cudaMemcpyHostToDevice));
		CHECK_CUDA_CALL(cudaMemcpy(gpu_local_centroids, local_centroids, (K * D / size) * sizeof(float), cudaMemcpyHostToDevice));

		CHECK_CUDA_CALL(cudaDeviceSynchronize());

		step_2_kernel<<<dyn_grid_cent, gen_block>>>(gpu_local_auxCentroids, gpu_local_centroids,
			gpu_local_pointsPerClass, gpu_local_maxDistance);
		CHECK_CUDA_LAST();

		CHECK_CUDA_CALL(cudaDeviceSynchronize());

		float local_maxDistance = 0.0f;
		CHECK_CUDA_CALL(cudaMemcpy(&local_maxDistance, gpu_local_maxDistance, sizeof(float), cudaMemcpyDeviceToHost));
		CHECK_CUDA_CALL(cudaMemcpy(local_centroids, gpu_local_centroids, (K * D / size) * sizeof(float), cudaMemcpyDeviceToHost));

		CHECK_CUDA_CALL(cudaDeviceSynchronize());

		MPI_Allreduce(&local_maxDistance, &maxDist, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
		MPI_Allgather(local_centroids, local_K * D, MPI_FLOAT, centroids, local_K * D, MPI_FLOAT, MPI_COMM_WORLD);


		CHECK_CUDA_CALL(cudaMemcpy(gpu_centroids, centroids, K * D * sizeof(float), cudaMemcpyHostToDevice));

		sprintf(line,"\n[Rank %d/%d] [%d] Cluster changes: %d\tMax. centroid distance: %f", rank + 1, size, it, changes, maxDist);
		output_msg = strcat(output_msg, line);
	} while((changes > minChanges) && (it < maxIterations) && (maxDist > maxThreshold));

	CHECK_CUDA_CALL(cudaMemcpy(local_classMap, gpu_local_classMap, (K / size) * sizeof(int), cudaMemcpyDeviceToHost));

	MPI_Gather(local_classMap, K / size, MPI_INT, classMap, K / size, MPI_INT, 0, MPI_COMM_WORLD);

	//END CLOCK*****************************************
	end = MPI_Wtime();
	printf("\nComputation: %f seconds", end - start);
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	start = MPI_Wtime();
	//**************************************************
	
	printf("%s", output_msg);

	if (rank == 0) {
		// Output and termination conditions

		if (changes <= minChanges) {
			printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, minChanges);
		}
		else if (it >= maxIterations) {
			printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
		}
		else {
			printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
		}	

		// Writing the classification of each point to the output file.
		CHECK_CUDA_CALL(cudaDeviceSynchronize());

		int error = writeResult(classMap, N, argv[6]);
		if(error != 0)
		{
			showFileError(error, argv[6]);
			exit(error);
		}

		//Free memory
		free(points);
		free(classMap);
	}	

	free(local_points);
	free(local_classMap);
	free(local_centroids);
	free(local_auxCentroids);
	free(local_pointsPerClass);
	free(centroids);

	cudaFree(gpu_local_points);
	cudaFree(gpu_centroids);
	cudaFree(gpu_auxCentroids);
	cudaFree(gpu_pointsPerClass);
	cudaFree(gpu_local_changes);
	cudaFree(gpu_local_classMap);
	cudaFree(gpu_local_maxDistance);
	cudaFree(gpu_local_auxCentroids);
	cudaFree(gpu_local_centroids);

	//END CLOCK*****************************************
	end = MPI_Wtime();
	printf("\n\nMemory deallocation: %f seconds\n", end - start);
	fflush(stdout);
	//***************************************************/
    CHECK_MPI_CALL(MPI_Finalize());
	return 0;
}