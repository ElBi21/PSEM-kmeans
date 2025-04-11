/*
 * k-Means clustering algorithm
 *
 * Reference sequential version (Do not modify this code)
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
#define _XOPEN_SOURCE 600

// #include <bits/pthreadtypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <limits.h>

#define MAXLINE 2000
#define MAXCAD 200

//Macros
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/* 
Function showFileError: It displays the corresponding error during file reading.
*/
void showFileError(int error, char* filename) {
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
int readInput(char* filename, int *lines, int *samples) {
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
int readInput2(char* filename, float* data) {
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
int writeResult(int *classMap, int lines, const char* filename) {	
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
void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K) {
	int i;
	int idx;
	for(i=0; i<K; i++)
	{
		idx = centroidPos[i];
		memcpy(&centroids[i*samples], &data[idx*samples], (samples*sizeof(float)));
	}
}

/*
Function euclideanDistance: Euclidean distance
This function could be modified
*/
float euclideanDistance(float* point, float* center, int samples) {
	float dist = 0.0;
	for (int d = 0; d < samples; d++) {
		dist += (point[d] - center[d]) * (point[d] - center[d]);
	}
	//dist = sqrt(dist);
	return(dist);
}

/*
Function zeroFloatMatriz: Set matrix elements to 0
This function could be modified
*/
void zeroFloatMatriz(float *matrix, int total_size, int t_rank, int t_count) {
	int local_size = floor((float) total_size / t_count);
	memset(matrix + local_size * t_rank, 0.0, local_size * sizeof(float));
	
	// In case of non-integer local sizes, make rank 0 fill in
	if ((total_size % t_count) != 0 && t_rank == 0) {
		memset(matrix + local_size * t_count, 0.0, (total_size - local_size * t_count) * sizeof(float));
	}
}

/*
Function zeroIntArray: Set array elements to 0
This function could be modified
*/
void zeroIntArray(int *array, int total_size, int t_rank, int t_count) {
	int local_size = floor((float) total_size / t_count);
	memset(array + t_rank * local_size, 0, local_size * sizeof(int));

	// In case of non-integer local sizes, make rank 0 fill in
	if ((total_size % t_count) != 0 && t_rank == t_count - 1) {
		memset(array + t_count * local_size, 0, (total_size - local_size * t_count) * sizeof(int));
	}
}

/* Struct for the thread arguments*/
struct thread_args {
	// Useful params
	long tsf_rank;
	long tsf_count;
	
	// Struct with all the global parameters
	struct global_params* global_params_struct;
};

/* Struct for storing global parameters. To be considered as constants */
struct global_params {
	// Pointers to data
	float* data;
	float* centroids;
	int* class_map;
	int* points_per_class;
	float* aux_centroids;

	// Parameters pointers
	int* min_changes_ptr;
	int* max_iterations_ptr;
	float* max_threshold_ptr;

	// Return pointers
	int* changes_return_ptr;
	int* iterations_return_ptr;
	float* max_dist_return_ptr;

	char* line_ptr;
	char* output_message_ptr;
	
	// Problem data params
	int n;
	int d;
	int k;

	// Pthreads data
	pthread_mutex_t* return_sync_mutex;
	pthread_mutex_t* step_1_mutex;
	pthread_barrier_t* return_sync_barrier;
	pthread_barrier_t* step_1_barrier;
	pthread_barrier_t* final_barrier;
	long* in_charge;
	long* min_in_charge;
};


void step1(int local_n, int k, int dims, int* assigned_class, float* local_min_dist, int* local_changes, int* local_points_per_class, 
	float* local_aux_centroids, float* global_data, float* global_centroids, int* global_class_map, long thread_rank) {

	for (int point_index = 0; point_index < local_n; point_index++) {
		*assigned_class = 1;
		*local_min_dist = FLT_MAX;

		// For each centroid...
		for (int centr_index = 0; centr_index < k; centr_index++) {
			// Compute l_2 squared (without root)
			float dist = euclideanDistance(
				&global_data[(local_n * dims * thread_rank) + point_index * dims], 
				&global_centroids[centr_index * dims], 
				dims);

			// If the distance is the smallest (locally), update the smallest distance found so far
			if (dist < *local_min_dist) {
				*local_min_dist = dist;
				*assigned_class = centr_index + 1;
			}
		}

		// If the class assigned is different, add 1 to local_changes
		if (global_class_map[thread_rank * local_n + point_index] != *assigned_class) {
			*(local_changes) = *(local_changes) + 1;
		}

		// Assign the new class
		global_class_map[thread_rank * local_n + point_index] = *assigned_class;

		// Step 2
		// Add 1 to the local points_per_class
		local_points_per_class[*assigned_class - 1] += 1;

		// Add the coordinates
		for (int dim_index = 0; dim_index < dims; dim_index++) {
			local_aux_centroids[(*assigned_class - 1) * dims + dim_index] += global_data[(local_n * dims * thread_rank) + point_index * dims + dim_index];
		}
	}
}


void step2(int local_k, int dims, float* global_centroids, float* local_max_dist, float* global_aux_centroids,
	int* global_points_per_class, long thread_rank) {
	*local_max_dist = FLT_MIN;
	for (int centroid_index = 0; centroid_index < local_k; centroid_index++) {
		// Average all dimensions
		for (int dimension_index = 0; dimension_index < dims; dimension_index++) {
			global_aux_centroids[thread_rank * local_k * dims + centroid_index * dims + dimension_index] /= global_points_per_class[thread_rank * local_k + centroid_index];
		}

		// Compute distance from old centroid to new centroid
		float dist_centroids = euclideanDistance(
			global_centroids + (thread_rank * local_k * dims) + centroid_index * dims, 
			global_aux_centroids + (thread_rank * local_k * dims) + centroid_index * dims, 
			dims
		);

		// Get new maximum distance
		*local_max_dist = MAX(*local_max_dist, dist_centroids);
	}

	for (int i = 0; i < local_k * dims; i++) {
		global_centroids[thread_rank * local_k * dims + i] = global_aux_centroids[thread_rank * local_k * dims + i];
	}
}


void* kernel(void* args) {
	struct thread_args* kernel_args = (struct thread_args*) args;
	struct global_params* global_params = kernel_args -> global_params_struct;

	// Dump some data from the struct
	long thread_rank = kernel_args -> tsf_rank;
	long thread_count = kernel_args -> tsf_count;
	long* in_charge = global_params -> in_charge;
	long* min_in_charge = global_params -> min_in_charge;
	// printf("Moving test from thread %ld! :D\n", thread_rank);
	
	int dims = global_params -> d;
	int k = global_params -> k;
	int n = global_params -> n;
	int local_n = floor((float) n / thread_count);
	int local_k = MAX(floor((float) k / thread_count), 1);

	int n_to_solve = n - local_n * *(min_in_charge);
	int k_to_solve = k - local_k * *(min_in_charge);

	float* global_centroids = global_params -> centroids;
	float* global_data = global_params -> data;
	int* global_class_map = global_params -> class_map;
	int* global_points_per_class = global_params -> points_per_class;
	float* global_aux_centroids = global_params -> aux_centroids;

	int* local_points_per_class = (int*) calloc(k, sizeof(int));
	float* local_aux_centroids = (float*) calloc(k * dims, sizeof(float));

	// Vars
	int iteration = 0;
	float local_min_dist = FLT_MAX;
	float local_max_dist = FLT_MIN;
	int local_changes;
	int assigned_class;

	// Initialize the arrays
	zeroIntArray(global_points_per_class, k, thread_rank, thread_count);
	zeroFloatMatriz(global_aux_centroids, k * dims, thread_rank, thread_count);
	memset(local_points_per_class, 0, k * sizeof(int));
	memset(local_aux_centroids, 0.0, k * dims * sizeof(float));

	if (thread_rank == 0)
		printf("Rank %ld:\n\tLocal n: %d (Remaining: %d)\n\tLocal k: %d (Remaining: %d)\n", thread_rank, local_n, n_to_solve, local_k, k_to_solve);

	// HERE AT MOST k + 1 THREADS ENTER
	if ((k >= thread_count) || ((k < thread_count) && (thread_rank <= (k + 1)))) {
		do {
			iteration++;
			local_changes = 0;
	
			if (thread_rank == 3)
				printf("Passed here for iteration: %d\n", iteration);
			
			// Step 1: compute the assignments
			printf("%ld ", thread_rank);
			if (thread_rank != *in_charge)
				step1(local_n, k, dims, &assigned_class, &local_min_dist, &local_changes, local_points_per_class, local_aux_centroids, global_data, global_centroids, global_class_map, thread_rank);
	
			if ((n_to_solve != 0) && (thread_rank == *in_charge)) {
				for (int point_index = 0; point_index < n_to_solve; point_index++) {
					assigned_class = 1;
					local_min_dist = FLT_MAX;
			
					// For each centroid...
					for (int centr_index = 0; centr_index < k; centr_index++) {
						// Compute l_2 squared (without root)
						float dist = euclideanDistance(
							&global_data[(local_n * dims * thread_count) + point_index * dims], 
							&global_centroids[centr_index * dims], 
							dims);
			
						// If the distance is the smallest (locally), update the smallest distance found so far
						if (dist < local_min_dist) {
							local_min_dist = dist;
							assigned_class = centr_index + 1;
						}
					}
			
					// If the class assigned is different, add 1 to local_changes
					if (global_class_map[thread_count * local_n + point_index] != assigned_class) {
						local_changes++;
					}
			
					// Assign the new class
					global_class_map[thread_count * local_n + point_index] = assigned_class;
			
					// Step 2
					// Add 1 to the local points_per_class
					local_points_per_class[assigned_class - 1] += 1;
			
					// Add the coordinates
					for (int dim_index = 0; dim_index < dims; dim_index++) {
						local_aux_centroids[(assigned_class - 1) * dims + dim_index] += global_data[(local_n * dims * thread_count) + point_index * dims + dim_index];
					}
				}
			}
	
			// Sum up all local points per class and auxiliary centroids
			pthread_mutex_lock(global_params -> step_1_mutex);
				// Reset global max dist and changes
				if (thread_rank == 0) {
					*(global_params -> changes_return_ptr) = 0;
					*(global_params -> max_dist_return_ptr) = FLT_MIN;
	
					// Choose next thread to be in charge of remaining items
					/**in_charge = *in_charge + 1;
					if (*in_charge == k) {
						*in_charge = 0;
					}*/
				}
	
				// Build global_aux_centroids from local parts
				if ((thread_rank != k) || (n_to_solve != 0) && (thread_rank == k)) {
					for (int centroid_index = 0; centroid_index < k; centroid_index++) {
						global_points_per_class[centroid_index] += local_points_per_class[centroid_index];
						
						for (int dimension_index = 0; dimension_index < dims; dimension_index++) {
							global_aux_centroids[centroid_index * dims + dimension_index] += local_aux_centroids[centroid_index * dims + dimension_index];
						}
					}
				}
			pthread_mutex_unlock(global_params -> step_1_mutex);
			
			// Wait for all threads to compute all centroids
			pthread_barrier_wait(global_params -> step_1_barrier);
	
			if (thread_rank != *in_charge)
				step2(local_k, dims, global_centroids, &local_max_dist, global_aux_centroids, global_points_per_class, thread_rank);
	
			if ((k_to_solve != 0) && (thread_rank == *in_charge)) {
				for (int centroid_index = 0; centroid_index < (k_to_solve); centroid_index++) {
					for (int dimension_index = 0; dimension_index < dims; dimension_index++) {
						global_aux_centroids[(thread_count * local_k * dims) + centroid_index * dims + dimension_index] /= global_points_per_class[thread_count * local_k + centroid_index];
					}
	
					float dist_centroids = euclideanDistance(
						global_centroids + (thread_count * local_k * dims) + centroid_index * dims, 
						global_aux_centroids + (thread_count * local_k * dims) + centroid_index * dims, 
						dims
					);
	
					local_max_dist = MAX(local_max_dist, dist_centroids);
				}
	
				// Replace previous centroids with new ones
				for (int i = 0; i < k_to_solve * dims; i++) {
					global_centroids[thread_count * local_k * dims + i] = global_aux_centroids[thread_count * local_k * dims + i];
				}
			}
	
			// Critical zone: update return parameters
			pthread_mutex_lock(global_params -> return_sync_mutex);
			*(global_params -> changes_return_ptr) += local_changes;
			*(global_params -> iterations_return_ptr) = iteration;
			*(global_params -> max_dist_return_ptr) = MAX(local_max_dist, *(global_params -> max_dist_return_ptr));
			pthread_mutex_unlock(global_params -> return_sync_mutex);
			
			if (iteration < 20 && thread_rank == 0) {
				printf("\nIteration %d\n\tCentroids\n", iteration);
				for (int i = 0; i < k; i++)
					printf("%f ", global_centroids[k * dims - k + i]);
				printf("\n\tAuxCentroids\n");
				for (int i = 0; i < k; i++)
					printf("%f ", global_aux_centroids[k * dims - k + i]);
				printf("\n\tPPC\n");
				for (int i = 0; i < k; i++)
					printf("%d ", global_points_per_class[i]);
				printf("\n\tIn charge: %ld\n\n", *in_charge);
	
				fflush(stdout);
			}
	
			pthread_barrier_wait(global_params -> final_barrier);
	
			// Print message
			if (thread_rank == 0) {
				sprintf(global_params -> line_ptr, "\n[%d] Cluster changes: %d\tMax. centroid distance: %f", iteration, *(global_params -> changes_return_ptr), *(global_params -> max_dist_return_ptr));
				global_params -> output_message_ptr = strcat(global_params -> output_message_ptr, global_params -> line_ptr);
			}
	
			// Reset variables used
			memset(local_points_per_class, 0, k * sizeof(int));
			memset(local_aux_centroids, 0.0, k * dims * sizeof(float));
			zeroIntArray(global_points_per_class, k, thread_rank, thread_count);
			zeroFloatMatriz(global_aux_centroids, k * dims, thread_rank, thread_count);
	
			pthread_barrier_wait(global_params -> final_barrier);
		} while (
			(iteration < *(global_params -> max_iterations_ptr)) && \
			(*(global_params -> changes_return_ptr) > *(global_params -> min_changes_ptr)) && \
			(pow(*(global_params -> max_threshold_ptr), 2) < *(global_params -> max_dist_return_ptr))
		);
	}

	free(local_points_per_class);
	free(local_aux_centroids);

	return NULL;
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
	* argv[4]: Minimum percentage of class changes. Algorithm termination condition.
	*          If between one iteration and the next, the percentage of class changes is less than
	*          this percentage, the algorithm stops.
	* argv[5]: Precision in the centroid distance after the update.
	*          It is an algorithm termination condition. If between one iteration of the algorithm 
	*          and the next, the maximum distance between centroids is less than this precision, the
	*          algorithm stops.
	* argv[6]: Output file. Class assigned to each point of the input file.
	* */
	if(argc != 8)
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
	if(error != 0) {
		showFileError(error,argv[1]);
		exit(error);
	}
	
	float* data = (float*) calloc(lines*samples, sizeof(float));
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
	int maxIterations = atoi(argv[3]);
	int minChanges = (int) (lines * atof(argv[4])/100.0);
	float maxThreshold = atof(argv[5]);

	int* centroidPos = (int*) calloc(K,sizeof(int));
	float* centroids = (float*) calloc(K * samples,sizeof(float));
	int* classMap = (int*) calloc(lines,sizeof(int));

    if (centroidPos == NULL || centroids == NULL || classMap == NULL) {
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}

	// Initial centrodis
	srand(0);
	int i;
	for (i = 0; i < K; i++) 
		centroidPos[i] = rand()%lines;
	
	// Loading the array of initial centroids with the data from the array data
	// The centroids are points stored in the data array.
	initCentroids(data, centroids, centroidPos, samples, K);

	long threads_count = (long) atoi(argv[7]);

	printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
	printf("\tNumber of clusters: %d\n", K);
	printf("\tMaximum number of iterations: %d\n", maxIterations);
	printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
	printf("\tMaximum centroid precision: %f\n", maxThreshold);
	printf("\tNumber of threads: %ld\n", threads_count);
	
	char *outputMsg = (char*) calloc(100000, sizeof(char));
	char line[1000];

	//pointPerClass: number of points classified in each class
	//auxCentroids: mean of the points in each class
	int* pointsPerClass = (int *) malloc(K*sizeof(int));
	float* auxCentroids = (float*) malloc(K*samples*sizeof(float));
	float* distCentroids = (float*) malloc(K*sizeof(float)); 
	if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL) {
		fprintf(stderr, "Memory allocation error.\n");
		exit(-4);
	}

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */

	int it, changes;
	float maxDist;
	long in_charge = 0;
	long min_in_charge;

	it = 0;
	changes = 0;
	maxDist = FLT_MIN;

	pthread_t* thread_handles;
	pthread_mutex_t return_sync_mutex;
	pthread_mutex_t step_1_mutex;
	pthread_barrier_t return_sync_barrier;
	pthread_barrier_t step_1_barrier;
	pthread_barrier_t final_barrier;

	thread_handles = malloc(threads_count * sizeof(pthread_t));
	pthread_mutex_init(&return_sync_mutex, NULL);
	pthread_mutex_init(&step_1_mutex, NULL);
	pthread_barrier_init(&step_1_barrier, NULL, (unsigned int) threads_count);
	pthread_barrier_init(&return_sync_barrier, NULL, (unsigned int) threads_count);
	pthread_barrier_init(&final_barrier, NULL, threads_count);

	// Define global parameters
	struct global_params g_params = {
		.data = data,
		.centroids = centroids,
		.class_map = classMap,
		.points_per_class = pointsPerClass,
		.aux_centroids = auxCentroids,

		.d = samples,
		.n = lines,
		.k = K,
		
		.max_threshold_ptr = &maxThreshold,
		.max_iterations_ptr = &maxIterations,
		.min_changes_ptr = &minChanges,

		.changes_return_ptr = &changes,
		.iterations_return_ptr = &it,
		.max_dist_return_ptr = &maxDist,

		.output_message_ptr = outputMsg,
		.line_ptr = line,

		.return_sync_mutex = &return_sync_mutex,
		.return_sync_barrier = &return_sync_barrier,
		.step_1_mutex = &step_1_mutex,
		.step_1_barrier = &step_1_barrier,
		.final_barrier = &final_barrier,
		.in_charge = &in_charge,
		.min_in_charge = &min_in_charge
	};

	//END CLOCK*****************************************
	end = clock();
	printf("\nMemory allocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	start = clock();
	//**************************************************

	if (K < threads_count) {
		// Only K threads will compute stuff
		min_in_charge = K;
	} else if (K >= threads_count) {
		// If there is some imbalance, it will be spread among all threads
		min_in_charge = 0;
	}
	in_charge = min_in_charge;


	for (long t = 0; t < threads_count; t++) {
		struct thread_args* thread_data = malloc(sizeof(struct thread_args));
		thread_data -> tsf_rank = t;
		thread_data -> tsf_count = threads_count;
		thread_data -> global_params_struct = &g_params;

		pthread_create(&thread_handles[t], NULL, kernel, (void*) thread_data);
	}	
	
	for (long t = 0; t < threads_count; t++) {
		pthread_join(thread_handles[t], NULL);
	}
/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */
	// Output and termination conditions
	printf("%s", outputMsg);	

	//END CLOCK*****************************************
	end = clock();
	printf("\nComputation: %f seconds", (double)(end - start) / CLOCKS_PER_SEC / threads_count);
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
	error = writeResult(classMap, lines, argv[6]);
	if (error != 0) {
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

	free(thread_handles);

	pthread_mutex_destroy(&return_sync_mutex);
	pthread_mutex_destroy(&step_1_mutex);
	pthread_barrier_destroy(&return_sync_barrier);
	pthread_barrier_destroy(&step_1_barrier);

	//END CLOCK*****************************************
	end = clock();
	printf("\n\nMemory deallocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//***************************************************/
	return 0;
}