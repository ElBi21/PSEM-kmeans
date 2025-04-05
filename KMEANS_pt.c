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
float euclideanDistance(float *point, float *center, int samples) {
	float dist = 0.0;
	for (int d = 0; d < samples; d++) {
		dist += (point[d] - center[d]) * (point[d] - center[d]);
	}
	dist = sqrt(dist);
	return(dist);
}

/*
Function zeroFloatMatriz: Set matrix elements to 0
This function could be modified
*/
void zeroFloatMatriz(float *matrix, int rows, int columns) {
	int i,j;
	for (i=0; i<rows; i++)
		for (j=0; j<columns; j++)
			matrix[i*columns+j] = 0.0;	
}

/*
Function zeroIntArray: Set array elements to 0
This function could be modified
*/
void zeroIntArray(int *array, int size) {
	int i;
	for (i=0; i<size; i++)
		array[i] = 0;	
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

	// Parameters pointers
	int* min_changes_ptr;
	int* max_iterations_ptr;
	float* max_threashold_ptr;

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
	pthread_barrier_t* return_sync_barrier;
};


void* kernel(void* args) {
	struct thread_args* kernel_args = (struct thread_args*) args;
	struct global_params* global_params = kernel_args -> global_params_struct;

	// Dump some data from the struct
	long thread_rank = kernel_args -> tsf_rank;
	long thread_count = kernel_args -> tsf_count;
	// printf("Moving test from thread %ld! :D\n", thread_rank);
	
	int dimensions = global_params -> d;
	int k = global_params -> k;
	int local_n = (global_params -> n) / thread_count;
	float* centroids = global_params -> centroids;
	int* class_map = global_params -> class_map;
	int* points_per_class = global_params -> points_per_class;

	int local_data_offset = thread_rank * local_n * dimensions;
	float* local_data = (global_params -> data) + (local_data_offset);

	int* global_changes = global_params -> changes_return_ptr;

	// Vars
	int iteration = 0;
	float min_dist, max_dist;

	// TEMP
	max_dist = FLT_MIN;

	int local_changes, assigned_class;

	// Test for checking arrays
	printf("Rank %ld, starting index: %d (local_n: %d), data: %f, global changes: %d\n", thread_rank, local_data_offset, local_n, local_data[dimensions], *global_changes);


	do {
		iteration++;
		local_changes = 0;
		
		for (int pt = 0; pt < local_n; pt++) {
			assigned_class = 1;
			min_dist = FLT_MAX;
			for (int centr = 0; centr < k; centr++) {
				float dist = euclideanDistance(local_data + (pt * dimensions + local_n), centroids + (centr * dimensions), dimensions);

				if (dist < min_dist) {
					min_dist = dist;
					assigned_class = centr + 1;
				}
			}
			
			if (class_map[pt] != assigned_class) {
				local_changes++;
				class_map[pt] = assigned_class;
			}
		}

		zeroIntArray(points_per_class, k);

		printf("[T%ld] Iteration %d, changes: %d\n", thread_rank, iteration, local_changes);

		// Critical zone: update return parameters
		pthread_mutex_lock(global_params -> return_sync_mutex);
		*(global_params -> changes_return_ptr) += local_changes;
		*(global_params -> iterations_return_ptr) = iteration;
		*(global_params -> max_dist_return_ptr) = MAX(max_dist, *(global_params -> max_dist_return_ptr));
		pthread_mutex_unlock(global_params -> return_sync_mutex);

		pthread_barrier_wait(global_params -> return_sync_barrier);

		// Print message
		if (thread_rank == 0) {
			sprintf(global_params -> line_ptr, "\n[%d] Cluster changes: %d\tMax. centroid distance: %f", iteration, *(global_params -> changes_return_ptr), max_dist);
			global_params -> output_message_ptr = strcat(global_params -> output_message_ptr, global_params -> line_ptr);
		}
	} while (
		(iteration < *(global_params -> max_iterations_ptr)) && \
		(local_changes < *(global_params -> min_changes_ptr)) && \
		(*(global_params -> max_threashold_ptr) > max_dist)
	);

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
	float* centroids = (float*) calloc(K*samples,sizeof(float));
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


	printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
	printf("\tNumber of clusters: %d\n", K);
	printf("\tMaximum number of iterations: %d\n", maxIterations);
	printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
	printf("\tMaximum centroid precision: %f\n", maxThreshold);
	
	//END CLOCK*****************************************
	end = clock();
	printf("\nMemory allocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	start = clock();
	//**************************************************
	char *outputMsg = (char*) calloc(10000, sizeof(char));
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

	long thread_counts = 4;

	int it, changes;
	float maxDist;

	changes = 0;

	pthread_t* thread_handles;
	pthread_mutex_t return_sync_mutex;
	pthread_barrier_t return_sync_barrier;

	thread_handles = malloc(thread_counts * sizeof(pthread_t));
	pthread_mutex_init(&return_sync_mutex, NULL);
	pthread_barrier_init(&return_sync_barrier, NULL, (unsigned int) thread_counts);
	
	printf("CIAOOOOO\n");
	// Define global parameters
	struct global_params g_params = {
		.data = data,
		.centroids = centroids,
		.class_map = classMap,
		.points_per_class = pointsPerClass,

		.d = samples,
		.n = lines,
		.k = K,
		
		.max_threashold_ptr = &maxThreshold,
		.max_iterations_ptr = &maxIterations,
		.min_changes_ptr = &minChanges,

		.changes_return_ptr = &changes,
		.iterations_return_ptr = &it,
		.max_dist_return_ptr = &maxDist,

		.output_message_ptr = outputMsg,
		.line_ptr = line,

		.return_sync_mutex = &return_sync_mutex,
		.return_sync_barrier = &return_sync_barrier
	};

	for (long t = 0; t < thread_counts; t++) {
		struct thread_args* thread_data = malloc(sizeof(struct thread_args));
		thread_data -> tsf_rank = t;
		thread_data -> tsf_count = thread_counts;
		thread_data -> global_params_struct = &g_params;

		pthread_create(&thread_handles[t], NULL, kernel, (void*) thread_data);
	}
	
	for (long t = 0; t < thread_counts; t++) {
		pthread_join(thread_handles[t], NULL);
	}

	// Recover parameters
	/*changes = *(g_params.changes_return_ptr);
	it = *(g_params.iterations_return_ptr);
	maxDist = *(g_params.max_dist_return_ptr);*/
/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */
	// Output and termination conditions
	printf("%s", outputMsg);	

	//END CLOCK*****************************************
	end = clock();
	printf("\nComputation: %f seconds", (double)(end - start) / CLOCKS_PER_SEC);
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
	pthread_barrier_destroy(&return_sync_barrier);

	//END CLOCK*****************************************
	end = clock();
	printf("\n\nMemory deallocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//***************************************************/
	return 0;
}

/* 

struct global_params* g_params = malloc(sizeof(struct global_params));
	g_params -> data = data;
	g_params -> centroids = centroids;

	g_params -> d = samples;
	g_params -> n = lines;
	g_params -> k = K;
		
	g_params -> max_threashold_ptr = &maxThreshold;
	g_params -> max_iterations_ptr = &maxIterations;
	g_params -> min_changes_ptr = &minChanges;
	g_params -> total_changes_ptr = &total_changes;

	g_params -> changes_return_ptr = &changes;
	g_params -> iterations_return_ptr = &it;
	g_params -> max_dist_return_ptr = &maxDist;

	g_params -> output_message_ptr = outputMsg;
	g_params -> line_ptr = line;

*/