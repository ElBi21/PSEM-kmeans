/*
 * k-Means clustering algorithm
 *
 * MPI version
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
#include <mpi.h>

#define MAXLINE 2000
#define MAXCAD 200

//Macros
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))


//	UNCHANGABLE FUNCTIONS DECLARATION

//Function showFileError: It displays the corresponding error during file reading.
void showFileError(int error, char* filename);

//Function readInput: It reads the file to determine the number of rows and columns.
int readInput(char* filename, int *lines, int *samples);


//Function readInput2: It loads data from file.
int readInput2(char* filename, float* data);


//Function writeResult: It writes in the output file the cluster of each sample (point).
int writeResult(int *classMap, int lines, const char* filename);

/*Function initCentroids: This function copies the values of the initial centroids, using their 
position in the input data structure as a reference map.*/
void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K);


//	CHANGABLE FUNCTIONS DEFINITION

/*Function euclideanDistance: Euclidean distance
This function could be modified*/
float euclideanDistance(float *point, float *center, int samples)
{
	float dist=0.0;
	for(int i=0; i<samples; i++) 
	{
		dist+= (point[i]-center[i])*(point[i]-center[i]);
	}
	dist = sqrt(dist);
	return(dist);
}

/*Function zeroIntArray: Set array elements to 0
This function could be modified*/
void zeroIntArray(int *array, int size)
{
	int i;
	for (i=0; i<size; i++)
		array[i] = 0;	
}

/*Function zeroFloatMatriz: Set matrix elements to 0
This function could be modified*/
void zeroFloatMatriz(float *matrix, int rows, int columns)
{
	int i,j;
	for (i=0; i<rows; i++)
		for (j=0; j<columns; j++)
			matrix[i*columns+j] = 0.0;	
}

//	MPI PROGRAM
int main(int argc, char* argv[])
{
	/* 0. Initialize MPI */
	MPI_Init( &argc, &argv );
	int rank, size;
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

//	START CLOCK
	double start, end;
	start = MPI_Wtime();

// 	READING PARAMETERS
	if(argc !=  7)
	{
		fprintf(stderr,"EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
		fprintf(stderr,"./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
		fflush(stderr);
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	int N = 0, D = 0;  
	float *points = NULL;
	
	if (rank == 0) {
		int error = readInput(argv[1], &N, &D);
		if(error != 0)
		{
			showFileError(error,argv[1]);
			MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
		}

		points = (float*)calloc(N*D, sizeof(float));
		if (points == NULL)
		{
			fprintf(stderr,"Memory allocation error.\n");
			MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
		}

		error = readInput2(argv[1], points);
		if(error != 0)
		{
			showFileError(error,argv[1]);
			MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
		}
	}

	// Broadcast the values of N and D
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Everyone gets the arguments of the program
	int K = atoi(argv[2]); 
	int maxIterations = atoi(argv[3]);
	int minChanges = (int) (N * atof(argv[4]) / 100.0);
	float maxThreshold = atof(argv[5]);
	
	float *centroids = (float*) calloc(K * D, sizeof(float));
	if (centroids == NULL) {
		fprintf(stderr,"Memory allocation error.\n");
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}
	int *classMap = NULL;

	// Rank 0 must initialize centroids, all other processes will get the array from it
	if (rank == 0) {
		int *centroidPos = (int*)calloc(K,sizeof(int));
		classMap = (int*)calloc(N,sizeof(int));

		if (centroidPos == NULL || classMap == NULL) {
			fprintf(stderr,"Memory allocation error.\n");
			MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
		}

		srand(0);
		for (int i = 0; i < K; i++) 
			centroidPos[i] = rand() % N;
		
		initCentroids(points, centroids, centroidPos, D, K);
		free(centroidPos);
		
		printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], N, D);
		printf("\tNumber of clusters: %d\n", K);
		printf("\tMaximum number of iterations: %d\n", maxIterations);
		printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), N);
		printf("\tMaximum centroid precision: %f\n", maxThreshold);
	}

	// Broadcast the centroids to all the processes
	MPI_Bcast(centroids, K*D, MPI_FLOAT, 0, MPI_COMM_WORLD);

//	END CLOCK
	end = MPI_Wtime();
	printf("\nRank %d | Memory allocation: %f seconds\n", rank, end - start);
	fflush(stdout);
	MPI_Barrier(MPI_COMM_WORLD);

//	START CLOCK
	start = MPI_Wtime();

	char* outputMsg;
	char* line;

	if (rank == 0) {
		outputMsg = (char*) calloc(10000, sizeof(char));
		line = (char*) calloc(100, sizeof(char));
	}

	int it = 0;
	int changes;
	float maxDist;
	int *pointsPerClass = (int *)malloc(K * sizeof(int)); 
	float *auxCentroids = (float*)malloc(K * D * sizeof(float)); 
	if (pointsPerClass == NULL || auxCentroids == NULL) {
		fprintf(stderr, "Memory allocation error.\n");
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}
	
	
//  VALUES NEEDED FOR STEP 1
	int *sendcounts = (int *)malloc(size * sizeof(int));
	int *displs = (int *)malloc(size * sizeof(int));
	int remainder = N % size;
	int sum = 0;
	for (int i = 0; i < size; ++i) {
		sendcounts[i] = (N / size) * D;
		if (i < remainder) sendcounts[i] += D;
		displs[i] = sum;
		sum += sendcounts[i];
	}

	// Works also with odd number of processes / points
	int local_n = sendcounts[rank] / D;
	float *local_points = (float*) calloc(local_n * D, sizeof(float));
	int *local_classMap = (int*) calloc(local_n, sizeof(int));
	if (local_points == NULL || local_classMap == NULL) {
		fprintf(stderr,"Memory allocation error.\n");
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	// Send part of data to all the processes
	MPI_Scatterv(points, sendcounts, displs, MPI_FLOAT, local_points, sendcounts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

//  VALUES NEEDED FOR STEP 2
	int *centroid_sendcounts = (int *)malloc(size * sizeof(int));
	int *centroid_displs = (int *)malloc(size * sizeof(int));
	int centroid_remainder = K % size;
	sum = 0;
	for (int i = 0; i < size; ++i) {
		centroid_sendcounts[i] = (K / size) * D;
		if (i < centroid_remainder) centroid_sendcounts[i] += D;
		centroid_displs[i] = sum;
		sum += centroid_sendcounts[i];
	}

	int local_k = centroid_sendcounts[rank] / D;
	float* local_centroids = (float*) calloc(local_k * D, sizeof(float));

	int terminate;
//	DO START
	do {
		it++;

		// Step 1. Calculate the distance from each point to the centroid

		int local_changes = 0;
		// For each local point...
		for (int i = 0; i < local_n; i++) {	
			int class = 1;
			float minDist = FLT_MAX;
			
			// For each centroid...
			for(int j = 0; j < K; j++) {
				// Compute l_2
				float dist = euclideanDistance(&local_points[i * D], &centroids[j*D], D);
				
				// If the distance is smallest so far, update minDist and the class of the point
				if (dist < minDist) {
					minDist = dist;
					class = j+1;
				}
			}

			// If the class changed, add 1 to the changes
			if (local_classMap[i] != class) {
				local_changes++;
			}

			// Change effectively the class
			local_classMap[i] = class;
		}

		// Gather all the changes for summing
		MPI_Request MPI_REQUEST;
		MPI_Ireduce(&local_changes, &changes, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD, &MPI_REQUEST);

		// Initialize pointsPerClass and the centroid auxiliary tables
		zeroIntArray(pointsPerClass, K);
		zeroFloatMatriz(auxCentroids, K, D);

		// Sum the coordinate of all local points
		for (int i = 0; i < local_n; i++) {
			int class = local_classMap[i];
			pointsPerClass[class-1]++;
			for(int j = 0; j < D; j++) {
				auxCentroids[(class - 1) * D + j] += local_points[i * D + j];
			}
		}

		// All the processes receive the other pointsPerClass and auxiliary centroids
		MPI_Allreduce(MPI_IN_PLACE, pointsPerClass, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, auxCentroids, K * D, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

		// Step 2. Update the centroids

		float local_maxDist = 0.0f;

		// For each local centroid...
		for (int i = 0; i < local_k; i++) {
			// Used for querying the global centroids table
			int global_idx = centroid_displs[rank] / D + i;
			if (global_idx >= K) break;
			
			float distance = 0.0f;
			
			if (pointsPerClass[global_idx] == 0) continue;
			// For each dimension...
			for (int j = 0; j < D; j++) {
				// Average the coordinates
				float centroid_val = auxCentroids[global_idx * D + j] / pointsPerClass[global_idx];
				// Compute the difference with the previous value
				float diff = centroids[global_idx * D + j] - centroid_val;
				distance += diff * diff;
				// And store the new coordinate
				local_centroids[i * D + j] = centroid_val;
			}

			distance = sqrt(distance);

			// Save value for later convergence check
			if(distance > local_maxDist) {
				local_maxDist = distance;
			}
		}

		// Store the greatest maxDist among all processes
		MPI_Reduce(&local_maxDist, &maxDist, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

		// Gather all local centroids and send them the updated table with all the new centroids
		MPI_Allgatherv(local_centroids, local_k * D, MPI_FLOAT, centroids, centroid_sendcounts, centroid_displs, MPI_FLOAT, MPI_COMM_WORLD);

		if (rank == 0) {
			MPI_Wait(&MPI_REQUEST, MPI_STATUS_IGNORE);
			sprintf(line,"\n[%d] Cluster changes: %d\nMax. centroid distance: %f", it, changes, maxDist);
			outputMsg = strcat(outputMsg, line);
			terminate = (changes > minChanges) && (it < maxIterations) && (maxDist > maxThreshold);
		}

		// Check if all processes must exit the loop
		MPI_Bcast(&terminate, 1, MPI_INT, 0, MPI_COMM_WORLD);
	} while (terminate);

	// Gather classMap using varying counts
	int *recvcounts = (int*) malloc(size * sizeof(int));
	int *rdispls = (int*) malloc(size * sizeof(int));
	sum = 0;
	for (int i = 0; i < size; ++i) {
		recvcounts[i] = sendcounts[i] / D;
		rdispls[i] = sum;
		sum += recvcounts[i];
	}

	// Actual MPI call for gathering the class mappings
	MPI_Gatherv(local_classMap, local_n, MPI_INT, classMap, recvcounts, rdispls, MPI_INT, 0, MPI_COMM_WORLD);

// 	Output and termination conditions
	if (rank == 0){
		printf("%s",outputMsg);
	}	

//	END CLOCK
	end = MPI_Wtime();
	printf("\n%d |Computation: %f seconds", rank, end - start);
	fflush(stdout);
	MPI_Barrier(MPI_COMM_WORLD);

//	START CLOCK
	start = MPI_Wtime();

	if (rank == 0){
		if (changes <= minChanges) {
			printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, minChanges);
		}
		else if (it >= maxIterations) {
			printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
		}
		else {
			printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
		}	

		int error = writeResult(classMap, N, argv[6]);
		if(error != 0)
		{
			showFileError(error, argv[6]);
			MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
		}
		fflush(stdout);
	}
	
//	FREE LOCAL ARRAYS
	free(local_points);
	free(local_classMap);
	free(local_centroids);
	free(sendcounts);
	free(displs);
	free(centroid_sendcounts);
	free(centroid_displs);
	free(recvcounts);
	free(rdispls);

//	Free memory
	if (rank == 0){
		free(points);
		free(classMap);
		free(outputMsg);
		free(line);
	}

	free(centroids);
	free(pointsPerClass);
    free(auxCentroids);

//	END CLOCK
	end = MPI_Wtime();
	printf("\n\n%d |Memory deallocation: %f seconds\n", rank, end - start);
	fflush(stdout);

//	FINALIZE
	MPI_Finalize();
	return 0;
}



//	UNCHANGABLE FUNCTIONS DEFINITIONS
void showFileError(int error, char* filename) {
	printf("Error: %d\n", error);
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

void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K) {
	int i;
	int idx;
	for(i=0; i<K; i++)
	{
		idx = centroidPos[i];
		memcpy(&centroids[i*samples], &data[idx*samples], (samples*sizeof(float)));
	}
}

/*//////////////////////////////////////////////////
	char fuori2[10000];  // Large enough to hold the formatted string
    strcpy(fuori2, "[ "); // Start the string with "["

    // Build the formatted string
    for (int i = 0; i < K; i++) {
        char temp[20];  // Temporary buffer for individual values
        sprintf(temp, "%d", pointsPerClass[i]);  // Convert number to string
        strcat(fuori2, temp);  // Append to the main string

        if (i < K - 1) {  // Add a comma and space if not the last element
            strcat(fuori2, ", ");
        }
		if (i%10 == 0 && i!=0){strcat(fuori2, "\n");}
    }
    strcat(fuori2, " ]");  // Close the string with "]"
    // Print the final formatted string
    printf("%d |My pointsPerClass: %s\n", rank, fuori2);
	fflush(stdout);
	MPI_Barrier(MPI_COMM_WORLD);
////////////////////////////////////////////////////////////////////////////*/