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
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <cuda.h>


#define MAXLINE 2000
#define MAXCAD 200

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

/*
 * 		CUDA Constants
 */
__constant__ int gpu_K;
__constant__ int gpu_n;
__constant__ int gpu_d;


/*
 *	===	Program Functions ===
 */

 
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
int readInput(char* filename, int *lines, int *samples)
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
        *samples = contsamples;  
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
int writeResult(int* classMap, int lines, const char* filename) {	
    FILE *fp;
    
    if ((fp = fopen(filename, "wt")) != NULL) {
        for (int i = 0; i < lines; i++) {
        	fprintf(fp,"%d\n", classMap[i]);
        }

        fclose(fp);  
   
        return 0;
    } else {
    	return -3; // No file found
	}
}

/*
Function initCentroids: This function copies the values of the initial centroids, using their 
position in the input data structure as a reference map.
*/
void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K)
{
	int i;
	int idx;
	for(i = 0; i < K; i++) {
		idx = centroidPos[i];
		memcpy(&centroids[i * samples], &data[idx * samples], (samples * sizeof(float)));
	}
}

/*
Function euclideanDistance: Euclidean distance
This function could be modified
*/
__device__ void euclideanDistance(float* data, float* centroids, int data_idx, int centroid_idx, float* return_addr) {
	float dist = 0.0;
	for (int dim = 0; dim < gpu_d; dim++) {
		float temp = (data[data_idx + gpu_n * dim] - centroids[centroid_idx * gpu_d + dim]);
		dist += temp * temp;
	}

	*return_addr = dist;
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

/* The function will transpose the array passed as input, so that to make it coalescent for CUDA */
float* transpose(float *array, int N, int D) {
    float* coalescedArray = (float*) malloc(N * D * sizeof(float));

    if (coalescedArray == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(-4);
    }

    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            coalescedArray[d * N + n] = array[n * D + d];
        }
    }
    
	return coalescedArray;
}

/*	Implementation of a custom atomicMax operation for floats. Freely taken from
 *	https://stackoverflow.com/questions/17399119. Credits to vinograd47
 *  (User link: https://stackoverflow.com/users/2451683/vinograd47)
 */
__device__ float custom_atomicMax(float* value_address, float val) {
    int* address_as_int = (int*) value_address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

/* 
 *	===	CUDA Kernels ===
 */

/*  To each thread, a point with D dimensions gets assigned. The thread must compute the l_2 norm and
 *  take the minimum.
 *
 *  Parameters:_m
 * 		- `data`: array of points, on the GPU;
 * 		- `centroids`: array of centroids, on the GPU;
 * 		- `class_map`: array with the classes, on the GPU.
 * 		- `changes_return`: address to which the total changes should be written on;
 * 
 *  Returns:
 * 		- `NULL`
 */
__global__ void assignment_step(float* data, float* centroids, int* class_map, int* changes_return) {
	// Compute thread index
	int thread_index = (blockIdx.y * gridDim.x * blockDim.x * blockDim.y) + (blockIdx.x * blockDim.x * blockDim.y) +
						(threadIdx.y * blockDim.x) +
						threadIdx.x;

	extern __shared__ float shared_centroids[];	// K x D x sizeof(float)
	float distance = 0.0;

	// Define block size and local thread index (index within block)
	int block_size = blockDim.x * blockDim.y;
	int local_thread_index = threadIdx.x + threadIdx.y * blockDim.x;

	int total = gpu_K * gpu_d;
	for (int i = local_thread_index; i < total; i += block_size) {
 	   shared_centroids[i] = centroids[i];
	}

	// Wait until all centroids have been copied
	__syncthreads();

	if (thread_index < gpu_n) {
		// Here thread_index becomes the index of the data (since for each thread -> a point)
		int class_int = 1;
		float min_dist = FLT_MAX;
		
		// For each centroid...
		for (int centroid = 0; centroid < gpu_K; centroid++) {

			// Compute the euclidean distance
			euclideanDistance(data, shared_centroids, thread_index, centroid, &distance);

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
	}
}


/*  To each thread, assign a point. For each such point, get the associated cluster,
 * 	and count the number of points for each cluster. Then, for each point, sum its
 * 	coordinates into a matrix which is used for doing the average of the coordinates.
 * 
 * 	Parameters:
 * 		- `data`: pointer to the data points, on the GPU;
 * 		- `class_map`: pointer to the cluster assignments, on the GPU;
 * 		- `aux_centroids`: pointer to the array for storing the new centroids dimensions, on the GPU;
 * 		- `points_per_class`: pointer to the table storing the amount of points for each class, on the GPU;
 * 
 *  Returns:
 * 		- `NULL`
 */
__global__ void update_step_points(float* data, int* class_map, float* aux_centroids, int* points_per_class) {
	// Index of the thread
	int thread_index = (blockIdx.y * gridDim.x * blockDim.x * blockDim.y) + (blockIdx.x * blockDim.x * blockDim.y) +
						(threadIdx.y * blockDim.x) +
						threadIdx.x;

	extern __shared__ char shared_mem[];
	float* shared_aux_centroids = (float*) shared_mem;
    int* shared_points_per_class = (int*) (shared_mem + sizeof(float) * gpu_K * gpu_d);

	int block_size = blockDim.x * blockDim.y;
	int local_thread_index = threadIdx.x + threadIdx.y * blockDim.x;

	int total = gpu_K * gpu_d;
	for (int i = local_thread_index; i < total; i += block_size) {
 	   shared_aux_centroids[i] = 0.0;
	}

	for (int k = local_thread_index; k < gpu_K; k += block_size) {
        shared_points_per_class[k] = 0;
    }

    __syncthreads();

	if (thread_index < gpu_n) {
		// Get the class assignment for a given point
		int class_assignment = class_map[thread_index];
		int centroid_index = class_assignment - 1;

		// Add 1 to PPC in the corresponding centroid
		atomicAdd(&(shared_points_per_class[centroid_index]), 1);

		for (int dim = 0; dim < gpu_d; dim++) {
			// For each dimension, add to aux_centroids the coordinates
			atomicAdd(&(shared_aux_centroids[centroid_index * gpu_d + dim]), data[thread_index + gpu_n * dim]);
		}
	}

	__syncthreads();

	// Bring back the arrays to the global memory
	for (int k = local_thread_index; k < gpu_K; k += block_size) {
        atomicAdd(&points_per_class[k], shared_points_per_class[k]);
    }

	for (int i = local_thread_index; i < total; i += block_size) {
 		atomicAdd(&aux_centroids[i], shared_aux_centroids[i]) ;
	}

}

/*  To each thread, assign a centroid. The coordinates of each centroid get averaged, and then the
 *  l_2 norm gets computed in order to check the max distance
 *  
 *  
 *  Parameters:
 * 		- `aux_centroids`: an array with all the temporary new coordinates of the centroids, on the GPU;
 * 		- `centroids`: array with the centroids, on the GPU;
 * 		- `points_per_class`: a table enumerating how many points have been assigned for each class, on the GPU;
 * 		- `max_distance`: the pointer to the maxDistance value, which is needed for convergence;
 *  
 *  Returns:
 * 		- `NULL`
 */
__global__ void update_step_centroids(float* aux_centroids, float* centroids, int* points_per_class, float* max_distance) {
	// Index of the thread
	int thread_index = (blockIdx.y * gridDim.x * blockDim.x * blockDim.y) + (blockIdx.x * blockDim.x * blockDim.y) +
						(threadIdx.y * blockDim.x) +
						threadIdx.x;


	int local_thread_index = threadIdx.x + threadIdx.y * blockDim.x;
	extern __shared__ float shared_max_distance[];
	if (local_thread_index == 0){
		shared_max_distance[0] = FLT_MIN;
	}

	// Eventually, make it run such that each thread is a dimensions,
	// the dimensions get averaged and then each thread does the distance or whatever
	if (thread_index < gpu_K) {
		float distance = 0.0;

		for (int dim = 0; dim < gpu_d; dim++) {
			aux_centroids[thread_index * gpu_d + dim] /= (float) points_per_class[thread_index];
			// Compute Euclidean distance (l_2 norm) to check for maximum distance
			distance += \
				(centroids[thread_index * gpu_d + dim] - aux_centroids[thread_index * gpu_d + dim]) * \
				(centroids[thread_index * gpu_d + dim] - aux_centroids[thread_index * gpu_d + dim]);

			// Update centroids within GPU
			centroids[thread_index * gpu_d + dim] = aux_centroids[thread_index * gpu_d + dim];
		}

		if (distance > shared_max_distance[0]) {
			// Atomic Max, disregard old value
			custom_atomicMax(&shared_max_distance[0], distance);
		}
	}

	__syncthreads();

	if (local_thread_index == 0){
		// Save the maxDistance on the GPU through a custom atomicMax for floats
		custom_atomicMax(max_distance, shared_max_distance[0]);
	}

}


int main(int argc, char* argv[]) {
	//START CLOCK***************************************
	clock_t start, end;
	start = clock();
	//**************************************************
	/*
	* PARAMETERS
	*
	* argv[1]: Input data file
	* argv[2]: Number of clusters
	* argv[3]: Maximum number of iterations of the method. Algorithm termination condition.
	* argv[4]: Minimum percfentage of class changes. Algorithm termination condition.
	*          If between one iteration and the next, the percentage of class changes is less than
	*          this percentage, the algorithm stops.
	* argv[5]: Precision in the centroid distance after the update.
	*          It is an algorithm termination condition. If between one iteration of the algorithm 
	*          and the next, the maximum distance between centroids is less than this precision, the
	*          algorithm stops.
	* argv[6]: Output file. Class assigned to each point of the input file.
	* */
	if (argc != 7) {
		fprintf(stderr,"EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
		fprintf(stderr,"./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
		fflush(stderr);
		exit(-1);
	}

	// Reading the input data
	// lines = number of points; samples = number of dimensions per point
	int lines = 0, D= 0;  
	
	int error = readInput(argv[1], &lines, &D);
	if (error != 0) {
		showFileError(error,argv[1]);
		exit(error);
	}
	
	float *data = (float*) calloc(lines * D, sizeof(float));
	if (data == NULL) {
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}

	error = readInput2(argv[1], data);
	if (error != 0) {
		showFileError(error,argv[1]);
		exit(error);
	}

	// Parameters
	int K = atoi(argv[2]); 
	int maxIterations = atoi(argv[3]);
	int minChanges = (int) (lines * atof(argv[4]) / 100.0);
	float maxThreshold = atof(argv[5]);

	int *centroidPos = (int*) calloc(K, sizeof(int));
	float *centroids = (float*) calloc(K * D, sizeof(float));
	int *classMap = (int*) calloc(lines, sizeof(int));

    if (centroidPos == NULL || centroids == NULL || classMap == NULL) {
		fprintf(stderr, "Memory allocation error.\n");
		exit(-4);
	}

	// Initial centrodis
	srand(0);
	int i;
	for (i = 0; i < K; i++) 
		centroidPos[i] = rand() % lines;
	
	// Loading the array of initial centroids with the data from the array data
	// The centroids are points stored in the data array.
	initCentroids(data, centroids, centroidPos, D, K);

	float* coalesced_data = transpose(data, lines, D);
	free(data);


	printf("\n    Input properties:");
	printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, D);
	printf("\tNumber of clusters: %d\n", K);
	printf("\tMaximum number of iterations: %d\n", maxIterations);
	printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
	printf("\tMaximum centroid precision: %f\n", maxThreshold);


	// Check CUDA device properties
	cudaDeviceProp cuda_prop;

	// Adapt to the number of points
	int pts_grid_size = lines / (32 * 32) + 1;
	int K_grid_size = K / (32 * 32) + 1;

	CHECK_CUDA_CALL(cudaGetDeviceProperties(&cuda_prop, 0));

	printf("\n    Device: %s\n", cuda_prop.name);
	printf("\tCompute Capability: %d.%d\n", cuda_prop.major, cuda_prop.minor);
	printf("\tMax threads / block: %d\n", cuda_prop.maxThreadsPerBlock);
	printf("\tMax threads / SM: %d\n", cuda_prop.maxThreadsPerMultiProcessor);
	printf("\tMax blocks / SM: %d\n", cuda_prop.maxBlocksPerMultiProcessor);
	printf("\tMax grid size: %d x %d x %d\n", cuda_prop.maxGridSize[0], cuda_prop.maxGridSize[1], cuda_prop.maxGridSize[2]);
	printf("\tMax shared memory per SM: %zuB\n", cuda_prop.sharedMemPerMultiprocessor);
	printf("\tNumber of SMs: %d\n", cuda_prop.multiProcessorCount);
	printf("\tStarting with following grids and blocks:\n");
	printf("\t    Blocks: 32 x 32\n");
	printf("\t    Grid for points: %d x (32 x 32)\n", pts_grid_size);
	printf("\t    Grid for centroids: %d x (32 x 32)\n", K_grid_size);


	// Output buffer allocation
	char *outputMsg = (char *)calloc(100000,sizeof(char));
	char line[200];

	int it = 0;
	int changes = 0;
	float maxDist = FLT_MIN;

	// pointPerClass: number of points classified in each class
	int *pointsPerClass = (int *) malloc(K * sizeof(int));
	
	// auxCentroids: mean of the points in each class
	float *auxCentroids = (float*) malloc(K * D * sizeof(float));
	
	if (pointsPerClass == NULL || auxCentroids == NULL) {
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}
	
	//END CLOCK*****************************************
	end = clock();
	printf("\nMemory allocation: %f seconds\n", (double) (end - start) / CLOCKS_PER_SEC);
	fflush(stdout);

	CHECK_CUDA_CALL( cudaSetDevice(0) );
	CHECK_CUDA_CALL( cudaDeviceSynchronize() );
	//**************************************************
	//START CLOCK***************************************
	start = clock();
	//**************************************************

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */

	dim3 gen_block(32, 32);
	dim3 points_grid_size(pts_grid_size);
	dim3 centroids_grid_size(K_grid_size);

	int data_size = lines * D * sizeof(float);
	int centroids_size = K * D * sizeof(float);
	int pointperclass_size = K * sizeof(int);

	// GPU pointers
	float* gpu_data;
	float* gpu_centroids;
	int* gpu_class_map;
	float* gpu_aux_centroids;
	int* gpu_points_per_class;

	// Loop-iteration needed vars
	int* gpu_changes;
	float* gpu_max_distance;

	// Load data into the GPU
	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_centroids, centroids_size));
	CHECK_CUDA_CALL(cudaMemcpy(gpu_centroids, centroids, centroids_size, cudaMemcpyHostToDevice));

	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_data, data_size));
	CHECK_CUDA_CALL(cudaMemcpy(gpu_data, coalesced_data, data_size, cudaMemcpyHostToDevice));
	
	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_class_map, lines * sizeof(int)));
	CHECK_CUDA_CALL(cudaMemset(gpu_class_map, 0, lines * sizeof(int)));
	
	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_aux_centroids, centroids_size));
	CHECK_CUDA_CALL(cudaMemset(gpu_aux_centroids, 0, centroids_size));

	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_points_per_class, K * sizeof(int)));
	CHECK_CUDA_CALL(cudaMemset(gpu_points_per_class, 0, K * sizeof(int)));

	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_changes, sizeof(int)));

	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_max_distance, sizeof(float)));

	// Initialize constant vars
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(gpu_K, &K, sizeof(int)));
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(gpu_n, &lines, sizeof(int)));
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(gpu_d, &D, sizeof(int)));

	//END CLOCK*****************************************
	end = clock();
	printf("\nCUDA Allocation: %f seconds\n", (double) (end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	start = clock();
	//**************************************************

	do {
		it++;
		
		// Reset changes and max distance0.0
		CHECK_CUDA_CALL(cudaMemset(gpu_changes, 0, sizeof(int)));
		CHECK_CUDA_CALL(cudaMemset(gpu_max_distance, FLT_MIN, sizeof(float)));
		
		// 1. Calculate the distance from each point to the centroid
		// Assign each point to the nearest centroid.
		assignment_step<<<points_grid_size, gen_block, centroids_size>>>(gpu_data, gpu_centroids, gpu_class_map, gpu_changes);
		
		// Write down to host the changes for checking convergence condition after waiting for GPU
		CHECK_CUDA_CALL(cudaDeviceSynchronize());
		CHECK_CUDA_CALL(cudaMemcpy(&changes, gpu_changes, sizeof(int), cudaMemcpyDeviceToHost));

		// 2. Recalculates the centroids: calculates the mean within each cluster
		// Perform the first update step, on the points
		update_step_points<<<points_grid_size, gen_block, centroids_size + pointperclass_size>>>(gpu_data, gpu_class_map, gpu_aux_centroids, gpu_points_per_class);
		CHECK_CUDA_LAST();

		CHECK_CUDA_CALL(cudaDeviceSynchronize());

		// Perform the second update step, on the centroids
		update_step_centroids<<<centroids_grid_size, gen_block, sizeof(float)>>>(gpu_aux_centroids, gpu_centroids, gpu_points_per_class, gpu_max_distance);
		CHECK_CUDA_LAST();

		CHECK_CUDA_CALL(cudaDeviceSynchronize());

		// Update effectively the positions and take maxDist
		CHECK_CUDA_CALL(cudaMemcpy(&maxDist, gpu_max_distance, sizeof(float), cudaMemcpyDeviceToHost));
		CHECK_CUDA_CALL(cudaMemset(gpu_aux_centroids, 0.0, K * D * sizeof(float)));
		CHECK_CUDA_CALL(cudaMemset(gpu_points_per_class, 0, K * sizeof(int)));

		CHECK_CUDA_CALL(cudaDeviceSynchronize());
		
		sprintf(line, "\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
		outputMsg = strcat(outputMsg,line);
	} while ((changes > minChanges) && (it < maxIterations) && \
			(maxDist > pow(maxThreshold, 2)));


/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */
	// Output and termination conditions
	printf("%s",outputMsg);	

	CHECK_CUDA_CALL( cudaDeviceSynchronize() );

	//END CLOCK*****************************************
	end = clock();
	printf("\nComputation: %f seconds", (double) (end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	start = clock();
	//**************************************************



	if (changes <= minChanges) {
		printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, minChanges);
	} else if (it >= maxIterations) {
		printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
	} else {
		printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
	}	

	// Writing the classification of each point to the output file.
	CHECK_CUDA_CALL(cudaMemcpy(classMap, gpu_class_map, lines * sizeof(int), cudaMemcpyDeviceToHost));
	CHECK_CUDA_CALL(cudaDeviceSynchronize());

	error = writeResult(classMap, lines, argv[6]);
	if(error != 0) {
		showFileError(error, argv[6]);
		exit(error);
	}

	//Free memory
	free(coalesced_data);
	free(classMap);
	free(centroidPos);
	free(centroids);
	free(pointsPerClass);
	free(auxCentroids);

	cudaFree(gpu_data);
	cudaFree(gpu_centroids);
	cudaFree(gpu_aux_centroids);
	cudaFree(gpu_changes);
	cudaFree(gpu_class_map);
	cudaFree(gpu_max_distance);
	cudaFree(gpu_points_per_class);

	//END CLOCK*****************************************
	end = clock();
	printf("\n\nMemory deallocation: %f seconds\n", (double) (end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//***************************************************/
	return 0;
}