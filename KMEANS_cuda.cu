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
__device__ void euclideanDistance(float *point, float *center, int samples, float* return_addr) {
	float dist = 0.0;
	for (int i = 0; i < samples; i++) {
		dist += (point[i] - center[i]) * (point[i] - center[i]);
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


/* 
 *		CUDA Kernels and Variables
 */

__constant__ int gpu_K;
__constant__ int gpu_n;
__constant__ int gpu_d;


/*	Implementation of a custom atomicMax operation for floats. Freely taken from
 *	https://stackoverflow.com/questions/17399119. Credits to vinograd47
 *  (User link: https://stackoverflow.com/users/2451683/vinograd47)
 */
__device__ float custom_atomic_max(float* value_address, float val) {
    int* address_as_int = (int*) value_address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}


/*  To each thread, a point with D dimensions gets assigned. The thread must compute the l_2 norm and
 *  take the minimum.
 *
 *  Parameters:
 * 		- `data`: array of points, on the GPU;
 * 		- `centroids`: array of centroids, on the GPU;
 * 		- `class_map`: array with the classes, on the GPU.
 * 		- `changes_return`: address to which the total changes should be written on;
 * 
 *  Returns:
 * 		- `NULL`
 */
__global__ void step_1_kernel(float* data, float* centroids, int* points_per_class, float* aux_centroids, int* class_map, int* changes_return) {
	// Compute thread index
	int thread_index = (blockIdx.y * gridDim.x * blockDim.x * blockDim.y) + (blockIdx.x * blockDim.x * blockDim.y) +
							(threadIdx.y * blockDim.x) +
							threadIdx.x;

	//extern __shared__ float shared_centroids[];	// K x D x sizeof(float)

	// Define block size and local thread index (index within block)
	//int block_size = blockDim.x * blockDim.y;
	//int local_thread_index = threadIdx.x + threadIdx.y * blockDim.x;

	// Copy centroids data into shared memory
	//for (int portion = 0; portion < (gpu_K * gpu_d) / block_size; portion++) {
	//	int copy_index = local_thread_index + portion * block_size;
	//	shared_centroids[copy_index] = centroids[copy_index];
	//}

	if (thread_index < gpu_n) {
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

		atomicAdd(&(points_per_class[point_index]), 1);

		//if (thread_index == 0)
		//	printf("  PPC: %d\n", points_per_class[thread_index]);

		for (int dim = 0; dim < gpu_d; dim++) {
			int index = point_index * gpu_d + dim;
			atomicAdd(&aux_centroids[index], data[data_index + dim]);
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
__global__ void step_2_kernel(float* centroids_table, float* centroids, int* points_per_class, float* max_distance) {
	// Index of the thread
	int thread_index = (blockIdx.y * gridDim.x * blockDim.x * blockDim.y) + (blockIdx.x * blockDim.x * blockDim.y) +
							(threadIdx.y * blockDim.x) +
							threadIdx.x;
	
	if (thread_index < gpu_K) {
		float distance;
		for (int d = 0; d < gpu_d; d++) {
			centroids_table[thread_index * gpu_d + d] /= (float) points_per_class[thread_index];
			// Compute Euclidean distance (l_2 norm) to check for maximum distance
			distance += pow((centroids[thread_index * gpu_d + d] - centroids_table[thread_index * gpu_d + d]), 2);
		}

		// Perform sqrt of distance
		//distance = sqrt(distance);

		// Exchange atomically, disregard old value
		custom_atomic_max(max_distance, distance);
	}
}


int main(int argc, char* argv[])
{

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
	* argv[4]: Minimum percentage of class changes. Algorithm termination condition.
	*          If between one iteration and the next, the percentage of class changes is less than
	*          this percentage, the algorithm stops.
	* argv[5]: Precision in the centroid distance after the update.
	*          It is an algorithm termination condition. If between one iteration of the algorithm 
	*          and the next, the maximum distance between centroids is less than this precision, the
	*          algorithm stops.
	* argv[6]: Output file. Class assigned to each point of the input file.
	* */
	if(argc !=  7)
	{
		fprintf(stderr,"EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
		fprintf(stderr,"./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
		fflush(stderr);
		exit(-1);
	}

	// Reading the input data
	// lines = number of points; samples = number of dimensions per point
	int n = 0, d = 0;  
	
	int error = readInput(argv[1], &n, &d);
	if(error != 0) {
		showFileError(error,argv[1]);
		exit(error);
	}
	
	float *data = (float*) calloc(n * d, sizeof(float));
	if (data == NULL) {
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}
	
	error = readInput2(argv[1], data);
	if(error != 0) {
		showFileError(error,argv[1]);
		exit(error);
	}

	// Parameters
	int K = atoi(argv[2]); 
	int max_iterations = atoi(argv[3]);
	int min_changes = (int) (n * atof(argv[4]) / 100.0);
	int max_threshold = pow(atof(argv[5]), 2);

	int* centroid_pos = (int*) calloc(K, sizeof(int));
	float* centroids = (float*) calloc(K * d, sizeof(float));
	int* class_map = (int*) calloc(n, sizeof(int));

    if (centroid_pos == NULL || centroids == NULL || class_map == NULL)
	{
		fprintf(stderr, "Memory allocation error.\n");
		exit(-4);
	}

	// Initial centrodis
	srand(0);
	int i;
	for(i = 0; i < K; i++) 
		centroid_pos[i] = rand() % n;
	
	// Loading the array of initial centroids with the data from the array data
	// The centroids are points stored in the data array.
	initCentroids(data, centroids, centroid_pos, d, K);


	printf("\n    Input properties:");
	printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], n, d);
	printf("\tNumber of clusters: %d\n", K);
	printf("\tMaximum number of iterations: %d\n", max_iterations);
	printf("\tMinimum number of changes: %d [%g%% of %d points]\n", min_changes, atof(argv[4]), n);
	printf("\tMaximum centroid precision: %f\n", max_threshold);


	// Check CUDA device properties
	cudaDeviceProp cuda_prop;

	// Adapt to the number of points
	int pts_grid_size = n / (32 * 32) + 1;
	int K_grid_size = K / (32 * 32) + 1;

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
	printf("\t    Grid for points: %d x (32 x 32)\n", pts_grid_size);
	printf("\t    Grid for centroids: %d x (32 x 32)\n", K_grid_size);


	
	//END CLOCK*****************************************
	end = clock();
	printf("\nMemory allocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);

	CHECK_CUDA_CALL( cudaSetDevice(0) );
	CHECK_CUDA_CALL( cudaDeviceSynchronize() );
	//**************************************************
	//START CLOCK***************************************
	start = clock();
	//**************************************************
	char *output_msg = (char*) calloc(100000, sizeof(char));
	char line[100];

	int it = 0;
	int changes = 0;
	float max_dist = FLT_MIN;

	//pointPerClass: number of points classified in each class
	//auxCentroids: mean of the points in each class
	int* points_per_class = (int*) malloc(K * sizeof(int));
	float* aux_centroids = (float*) malloc(K * d * sizeof(float));
	float* dist_centroids = (float*) malloc(K * sizeof(float)); 
	if (points_per_class == NULL || aux_centroids == NULL || dist_centroids == NULL) {
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}


	// Set carveout to be of maximum size available
	int carveout = cudaSharedmemCarveoutMaxShared;

	CHECK_CUDA_CALL(cudaFuncSetAttribute(step_1_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, carveout));

	dim3 gen_block(32, 32);
	dim3 dyn_grid_pts(pts_grid_size);
	dim3 dyn_grid_cent(K_grid_size);

	int data_size = n * d * sizeof(float);
	int centroids_size = K * d * sizeof(float);

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
	CHECK_CUDA_CALL(cudaMemcpy(gpu_data, data, data_size, cudaMemcpyHostToDevice));
	
	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_class_map, n * sizeof(int)));
	CHECK_CUDA_CALL(cudaMemset(gpu_class_map, 0, n * sizeof(int)));
	
	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_aux_centroids, centroids_size));
	CHECK_CUDA_CALL(cudaMemset(gpu_aux_centroids, 0, centroids_size));

	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_points_per_class, K * sizeof(int)));
	CHECK_CUDA_CALL(cudaMemset(gpu_points_per_class, 0, K * sizeof(int)));

	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_changes, sizeof(int)));

	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_max_distance, sizeof(float)));

	// Initialize constant vars
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(gpu_K, &K, sizeof(int)));
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(gpu_n, &n, sizeof(int)));
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(gpu_d, &d, sizeof(int)));

	//END CLOCK*****************************************
	end = clock();
	printf("\nCUDA initialization: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	start = clock();
	//**************************************************

	do {
		it++;

		// Reset changes, max distance, table of centroids and points per class
		CHECK_CUDA_CALL(cudaMemset(gpu_changes, 0, sizeof(int)));
		CHECK_CUDA_CALL(cudaMemset(gpu_max_distance, FLT_MIN, sizeof(float)));
		CHECK_CUDA_CALL(cudaMemset(gpu_aux_centroids, 0, K * d * sizeof(int)));
		CHECK_CUDA_CALL(cudaMemset(gpu_points_per_class, 0, K * sizeof(int)));
		
		// Ensure memory is actually ready for being used
		CHECK_CUDA_CALL(cudaDeviceSynchronize());

		// 1. Calculate the distance from each point to the centroid

		// Assign each point to the nearest centroid.
		step_1_kernel<<<dyn_grid_pts, gen_block, centroids_size>>>(gpu_data, gpu_centroids, gpu_points_per_class,
			gpu_aux_centroids, gpu_class_map, gpu_changes);
		CHECK_CUDA_LAST();

		// Write down to host the changes for checking convergence condition after waiting for GPU
		CHECK_CUDA_CALL(cudaDeviceSynchronize());
		CHECK_CUDA_CALL(cudaMemcpy(&changes, gpu_changes, sizeof(int), cudaMemcpyDeviceToHost));

		// 2. Recalculates the centroids: calculates the mean within each cluster
        
		// Perform the second update step, on the centroids
		step_2_kernel<<<dyn_grid_cent, gen_block>>>(gpu_aux_centroids, gpu_centroids, gpu_points_per_class, gpu_max_distance);
		CHECK_CUDA_LAST();

		// Update effectively the positions and take maxDist
		CHECK_CUDA_CALL(cudaMemcpy(&max_dist, gpu_max_distance, sizeof(float), cudaMemcpyDeviceToHost));
		CHECK_CUDA_CALL(cudaMemcpy(gpu_centroids, gpu_aux_centroids, centroids_size, cudaMemcpyDeviceToDevice));
		
		CHECK_CUDA_CALL(cudaDeviceSynchronize());

		sprintf(line,"\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, max_dist);
		output_msg = strcat(output_msg, line);

	} while((changes > min_changes) && (it < max_iterations) && (max_dist > pow(max_threshold, 2)));


/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */
	// Output and termination conditions
	printf("%s", output_msg);	

	CHECK_CUDA_CALL( cudaDeviceSynchronize() );

	//END CLOCK*****************************************
	end = clock();
	printf("\nComputation: %f seconds", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	start = clock();
	//**************************************************

	

	if (changes <= min_changes) {
		printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, min_changes);
	}
	else if (it >= max_iterations) {
		printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, max_iterations);
	}
	else {
		printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", max_dist, max_threshold);
	}	

	// Writing the classification of each point to the output file.
	CHECK_CUDA_CALL(cudaMemcpy(class_map, gpu_class_map, n * sizeof(int), cudaMemcpyDeviceToHost));
	CHECK_CUDA_CALL(cudaDeviceSynchronize());

	error = writeResult(class_map, n, argv[6]);
	if(error != 0)
	{
		showFileError(error, argv[6]);
		exit(error);
	}

	//Free memory
	free(data);
	free(class_map);
	free(centroid_pos);
	free(centroids);
	free(dist_centroids);
	free(points_per_class);
	free(aux_centroids);

	cudaFree(gpu_data);
	cudaFree(gpu_centroids);
	cudaFree(gpu_aux_centroids);
	cudaFree(gpu_changes);
	cudaFree(gpu_class_map);
	cudaFree(gpu_max_distance);
	cudaFree(gpu_points_per_class);

	//END CLOCK*****************************************
	end = clock();
	printf("\n\nMemory deallocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//***************************************************/
	return 0;
}