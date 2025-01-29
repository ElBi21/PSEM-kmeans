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
__global__ void assignment_step(float* data, float* centroids, int* class_map, int* changes_return) {
	
	// Compute thread index
	int thread_index = (blockIdx.y * gridDim.x * blockDim.x * blockDim.y) + (blockIdx.x * blockDim.x * blockDim.y) +
							(threadIdx.y * blockDim.x) +
							threadIdx.x;

	if (thread_index < gpu_n) {
		int data_index = thread_index * gpu_d;
		int class_int = class_map[thread_index];
		float min_dist = FLT_MAX;
		
		// For each centroid...
		for (int centroid = 0; centroid < gpu_K; centroid++) {
			float distance;

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
		// printf("[K1] Thread %d, I got class %d\n", thread_index, class_map[thread_index]);
	}
}


/*  To each thread, assign a point. For each such point, get the associated cluster,
 * 	and count the number of points for each cluster. Then, for each point, sum its
 * 	coordinates into a matrix which is used for doing the average of the coordinates.
 * 	After that, average all the coordinates and check the maximum distance that changed.
 * 
 * 	Parameters:
 * 		- `data`: pointer to the data points, on the GPU;
 * 		- `class_map`: pointer to the cluster assignments, on the GPU;
 * 		- `centroids_table`: pointer to the table for storing the centroids dimensions, on the GPU;
 * 		- `points_per_class`: pointer to the table storing the amount of points for each class, on the GPU;
 * 
 *  Returns:
 * 		- `NULL`
 */
__global__ void update_step_points(float* data, int* class_map, float* centroids_table, int* points_per_class) {
	// Index of the thread
	int thread_index = (blockIdx.y * gridDim.x * blockDim.x * blockDim.y) + (blockIdx.x * blockDim.x * blockDim.y) +
							(threadIdx.y * blockDim.x) +
							threadIdx.x;

	if (thread_index < gpu_n) {
		int data_index = thread_index * gpu_d;
		int class_assignment = class_map[thread_index];
		int point_index = class_assignment - 1;

		//printf("[K2] Thread %d, with class %d. Points for corresponding class: %d\n", thread_index, class_assignment, points_per_class[point_index]);

		// Pointer arithmetic to atomically add 1 to points_per_class
		//if (thread_index == 0 || thread_index == 1)
		//	printf("Printing the value of PPC: %d\n", points_per_class[class_assignment - 1]);

		atomicAdd(&(points_per_class[point_index]), 1);

		//printf("Class assignments for %d: %d    ", class_assignment, points_per_class[class_assignment - 1]);

		for (int dim = 0; dim < gpu_d; dim++) {
			int index = point_index * gpu_d + dim;
			atomicAdd(&centroids_table[index], data[data_index + dim]);
		}
	}

	//__syncthreads();
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
 *  
 */
__global__ void update_step_centroids(float* centroids_table, float* centroids, int* points_per_class, float* max_distance) {
	// Average the coordinates of the centroids
	int centroid_index = threadIdx.x;
	
	float distance;
	for (int d = 0; d < gpu_d; d++) {
		centroids_table[centroid_index * gpu_d + d] /= (float) points_per_class[centroid_index];
		// Compute Euclidean distance (l_2 norm) to check for maximum distance
		distance += pow((centroids[centroid_index * gpu_d + d] - centroids_table[centroid_index * gpu_d + d]), 2);
	}

	// Perform sqrt of distance
	distance = sqrt(distance);

	if (distance > *max_distance) {
		// Exchange atomically, disregard old value
		atomicExch(max_distance, distance);
	}

	__syncthreads();
}


int main(int argc, char* argv[])
{

	//START CLOCK***************************************
	#ifdef _OPEN_MP
		double start, end;
		start = omp_get_wtime();
	#else
		clock_t start, end;
		start = clock();
	#endif

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
	int lines = 0, samples= 0;  
	
	int error = readInput(argv[1], &lines, &samples);
	if(error != 0)
	{
		showFileError(error,argv[1]);
		exit(error);
	}
	
	float *data = (float*) calloc(lines * samples, sizeof(float));
	if (data == NULL) {
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}
	error = readInput2(argv[1], data);
	if(error != 0)
	{
		showFileError(error,argv[1]);
		exit(error);
	}

	// Parameters
	int K = atoi(argv[2]); 
	int maxIterations = atoi(argv[3]);
	int minChanges = (int) (lines * atof(argv[4]) / 100.0);
	float maxThreshold = atof(argv[5]);

	int *centroidPos = (int*) calloc(K, sizeof(int));
	float *centroids = (float*) calloc(K * samples, sizeof(float));
	int *classMap = (int*) calloc(lines, sizeof(int));

    if (centroidPos == NULL || centroids == NULL || classMap == NULL)
	{
		fprintf(stderr, "Memory allocation error.\n");
		exit(-4);
	}

	// Initial centrodis
	srand(0);
	int i;
	for(i = 0; i < K; i++) 
		centroidPos[i] = rand() % lines;
	
	// Loading the array of initial centroids with the data from the array data
	// The centroids are points stored in the data array.
	initCentroids(data, centroids, centroidPos, samples, K);


	printf("\n    Input properties:");
	printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
	printf("\tNumber of clusters: %d\n", K);
	printf("\tMaximum number of iterations: %d\n", maxIterations);
	printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
	printf("\tMaximum centroid precision: %f\n", maxThreshold);


	// Check CUDA device properties
	cudaDeviceProp cuda_prop;
	//CHECK_CUDA_CALL(cudaGetDevice(0));
	CHECK_CUDA_CALL(cudaGetDeviceProperties(&cuda_prop, 0));

	printf("\n    Device: %s\n", cuda_prop.name);
	printf("\tMax threads / block: %d\n", cuda_prop.maxThreadsPerBlock);
	printf("\tMax threads / SM: %d\n", cuda_prop.maxThreadsPerMultiProcessor);
	printf("\tMax blocks / SM: %d\n", cuda_prop.maxBlocksPerMultiProcessor);
	printf("\tMax grid size: %d x %d x %d\n", cuda_prop.maxGridSize[0], cuda_prop.maxGridSize[1], cuda_prop.maxGridSize[2]);
	printf("\tNumber of SMs: %d\n", cuda_prop.multiProcessorCount);
	
	//END CLOCK*****************************************
	#ifdef _OPEN_MP
		end = omp_get_wtime();
	#else
		end = clock();
	#endif
	printf("\nMemory allocation: %f seconds\n", end - start);
	fflush(stdout);

	CHECK_CUDA_CALL( cudaSetDevice(0) );
	CHECK_CUDA_CALL( cudaDeviceSynchronize() );
	//**************************************************
	//START CLOCK***************************************
	#ifdef _OPEN_MP
		start = omp_get_wtime();
	#else
		start = clock();
	#endif
	//**************************************************
	char *outputMsg = (char *)calloc(100000,sizeof(char));
	char line[100];

	//int j;
	//int classInt;
	//float dist;
	//float minDist = FLT_MAX;
	int it = 0;
	int changes = 0;
	float maxDist = FLT_MIN;

	//pointPerClass: number of points classified in each class
	//auxCentroids: mean of the points in each class
	int *pointsPerClass = (int *) malloc(K * sizeof(int));
	float *auxCentroids = (float*) malloc(K * samples * sizeof(float));
	float *distCentroids = (float*) malloc(K * sizeof(float)); 
	if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */

	// Adapt to the number of points
	int grid_size = lines / (32 * 32) + 1;

	dim3 point_block(32, 32);
	dim3 temp_grid(grid_size);

	int data_size = lines * samples * sizeof(float);
	int centroids_size = K * samples * sizeof(float);

	// GPU pointers
	float* gpu_data;
	float* gpu_centroids;
	int* gpu_class_map;
	float* gpu_centroids_temp;
	int* gpu_points_per_class;

	// Loop-iteration needed vars
	int* gpu_changes;
	float* gpu_max_distance;

	// Load data into the GPU
	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_centroids, centroids_size));
	CHECK_CUDA_CALL(cudaMemcpy(gpu_centroids, centroids, centroids_size, cudaMemcpyHostToDevice));

	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_data, data_size));
	CHECK_CUDA_CALL(cudaMemcpy(gpu_data, data, data_size, cudaMemcpyHostToDevice));
	
	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_class_map, lines * sizeof(int)));
	CHECK_CUDA_CALL(cudaMemset(gpu_class_map, 0, lines * sizeof(int)));
	
	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_centroids_temp, centroids_size));
	CHECK_CUDA_CALL(cudaMemset(gpu_centroids_temp, 0, centroids_size));

	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_points_per_class, K * sizeof(int)));
	CHECK_CUDA_CALL(cudaMemset(gpu_points_per_class, 0, K * sizeof(int)));

	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_changes, sizeof(int)));

	CHECK_CUDA_CALL(cudaMalloc((void**) &gpu_max_distance, sizeof(float)));

	// Initialize constant vars
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(gpu_K, &K, sizeof(int)));
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(gpu_n, &lines, sizeof(int)));
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(gpu_d, &samples, sizeof(int)));

	do {
		it++;

		// Reset changes and max distance
		CHECK_CUDA_CALL(cudaMemset(gpu_changes, 0, sizeof(int)));
		CHECK_CUDA_CALL(cudaMemset(gpu_max_distance, FLT_MIN, sizeof(float)));

		// 1. Calculate the distance from each point to the centroid

		// Assign each point to the nearest centroid.
		assignment_step<<<temp_grid, point_block>>>(gpu_data, gpu_centroids, gpu_class_map, gpu_changes);

		// Write down to host the changes for checking convergence condition after waiting for GPU
		CHECK_CUDA_CALL(cudaDeviceSynchronize());
		CHECK_CUDA_CALL(cudaMemcpy(&changes, gpu_changes, sizeof(int), cudaMemcpyDeviceToHost));

		// 2. Recalculates the centroids: calculates the mean within each cluster
		
		// Ensure memory is actually ready for being used
		CHECK_CUDA_CALL(cudaMemset(gpu_centroids_temp, 0, K * samples * sizeof(int)));
		CHECK_CUDA_CALL(cudaMemset(gpu_points_per_class, 0, K * sizeof(int)));
		CHECK_CUDA_CALL(cudaDeviceSynchronize());

		// Perform the first update step, on the points
		update_step_points<<<temp_grid, point_block>>>(gpu_data, gpu_class_map, gpu_centroids_temp, gpu_points_per_class);
		CHECK_CUDA_LAST();

		// Perform the second update step, on the centroids
		update_step_centroids<<<1, K>>>(gpu_centroids_temp, gpu_centroids, gpu_points_per_class, gpu_max_distance);
		CHECK_CUDA_LAST();

		// Update effectively the positions and take maxDist
		CHECK_CUDA_CALL(cudaMemcpy(&maxDist, gpu_max_distance, sizeof(float), cudaMemcpyDeviceToHost));
		CHECK_CUDA_CALL(cudaMemcpy(gpu_centroids, gpu_centroids_temp, centroids_size, cudaMemcpyDeviceToDevice));
		
		sprintf(line,"\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
		outputMsg = strcat(outputMsg,line);

	} while((changes > minChanges) && (it < maxIterations) && (maxDist > maxThreshold));

/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */
	// Output and termination conditions
	printf("%s",outputMsg);	

	CHECK_CUDA_CALL( cudaDeviceSynchronize() );

	//END CLOCK*****************************************
	#ifdef _OPEN_MP
		end = omp_get_wtime();
	#else
		end = clock();
	#endif
	printf("\nComputation: %f seconds", end - start);
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	#ifdef _OPEN_MP
		start = omp_get_wtime();
	#else
		start = clock();
	#endif
	//**************************************************

	

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
	error = writeResult(classMap, lines, argv[6]);
	if(error != 0)
	{
		showFileError(error, argv[6]);
		exit(error);
	}

	//Free memory
	free(data);
	free(classMap);
	free(centroidPos);
	free(centroids);
	free(distCentroids);
	free(pointsPerClass);
	free(auxCentroids);

	cudaFree(gpu_data);
	cudaFree(gpu_centroids);
	cudaFree(gpu_centroids_temp);
	cudaFree(gpu_changes);
	cudaFree(gpu_class_map);
	cudaFree(gpu_max_distance);
	cudaFree(gpu_points_per_class);

	//END CLOCK*****************************************
	#ifdef _OPEN_MP
		end = omp_get_wtime();
	#else
		end = clock();
	#endif
	printf("\n\nMemory deallocation: %f seconds\n", end - start);
	fflush(stdout);
	//***************************************************/
	return 0;
}
