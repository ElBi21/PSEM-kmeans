/*
 * k-Means clustering algorithm
 *
 * OpenMP version
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
#include <omp.h>
#include <mpi.h>

#define MAXLINE 2000
#define MAXCAD 200

// Macros
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/*
Function showFileError: It displays the corresponding error during file reading.
*/
void showFileError(int error, char *filename)
{
	printf("Error\n");
	switch (error)
	{
	case -1:
		fprintf(stderr, "\tFile %s has too many columns.\n", filename);
		fprintf(stderr, "\tThe maximum number of columns has been exceeded. MAXLINE: %d.\n", MAXLINE);
		break;
	case -2:
		fprintf(stderr, "Error reading file: %s.\n", filename);
		break;
	case -3:
		fprintf(stderr, "Error writing file: %s.\n", filename);
		break;
	}
	fflush(stderr);
}

/*
Function readInput: It reads the file to determine the number of rows and columns.
*/
int readInput(char *filename, int *lines, int *samples)
{
	FILE *fp;
	char line[MAXLINE] = "";
	char *ptr;
	const char *delim = "\t";
	int contlines, contsamples = 0;

	contlines = 0;

	if ((fp = fopen(filename, "r")) != NULL)
	{
		while (fgets(line, MAXLINE, fp) != NULL)
		{
			if (strchr(line, '\n') == NULL)
			{
				return -1;
			}
			contlines++;
			ptr = strtok(line, delim);
			contsamples = 0;
			while (ptr != NULL)
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
int readInput2(char *filename, float *data)
{
	FILE *fp;
	char line[MAXLINE] = "";
	char *ptr;
	const char *delim = "\t";
	int i = 0;

	if ((fp = fopen(filename, "rt")) != NULL)
	{
		while (fgets(line, MAXLINE, fp) != NULL)
		{
			ptr = strtok(line, delim);
			while (ptr != NULL)
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
		return -2; // No file found
	}
}

/*
Function writeResult: It writes in the output file the cluster of each sample (point).
*/
int writeResult(int *classMap, int lines, const char *filename)
{
	FILE *fp;

	if ((fp = fopen(filename, "wt")) != NULL)
	{
		for (int i = 0; i < lines; i++)
		{
			fprintf(fp, "%d\n", classMap[i]);
		}
		fclose(fp);

		return 0;
	}
	else
	{
		return -3; // No file found
	}
}

/*

Function initCentroids: This function copies the values of the initial centroids, using their
position in the input data structure as a reference map.
*/
void initCentroids(const float *data, float *centroids, int *centroidPos, int samples, int K)
{
	int i;
	int idx;
	for (i = 0; i < K; i++)
	{
		idx = centroidPos[i];
		memcpy(&centroids[i * samples], &data[idx * samples], (samples * sizeof(float)));
	}
}

/*
Function euclideanDistance: Euclidean distance
This function could be modified
*/
float euclideanDistance(float *point, float *center, int samples)
{
	float dist = 0.0;
	for (int i = 0; i < samples; i++)
	{
		dist += (point[i] - center[i]) * (point[i] - center[i]);
	}
	return (dist);
}

/*
Function zeroFloatMatriz: Set matrix elements to 0
This function could be modified
*/
void zeroFloatMatriz(float *matrix, int rows, int columns)
{
	memset(matrix, 0, rows * columns * sizeof(float));
}

/*
Function zeroIntArray: Set array elements to 0
This function could be modified
*/
void zeroIntArray(int *array, int size)
{
	memset(array, 0, size * sizeof(int));
}

int main(int argc, char *argv[])
{

	// START CLOCK***************************************
	double start, end;
	start = omp_get_wtime();
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
	 * argv[7]: (OPTIONAL) Number of threads for OpenMP
	 * */
	if ((argc != 7) && !(argc == 8))
	{
		fprintf(stderr, "EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
		fprintf(stderr, "./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
		fflush(stderr);
		exit(-1);
	}

	// Reading the input data
	// lines = number of points; samples = number of dimensions per point
	int lines = 0, samples = 0;

	int error = readInput(argv[1], &lines, &samples);
	if (error != 0)
	{
		showFileError(error, argv[1]);
		exit(error);
	}

	float *data = (float *)calloc(lines * samples, sizeof(float));
	if (data == NULL)
	{
		fprintf(stderr, "Memory allocation error.\n");
		exit(-4);
	}
	error = readInput2(argv[1], data);
	if (error != 0)
	{
		showFileError(error, argv[1]);
		exit(error);
	}

	// Parameters
	int K = atoi(argv[2]);
	int maxIterations = atoi(argv[3]);
	int minChanges = (int)(lines * atof(argv[4]) / 100.0);
	float maxThreshold = atof(argv[5]);

	int *centroidPos = (int *)calloc(K, sizeof(int));
	float *centroids = (float *)calloc(K * samples, sizeof(float));
	int *classMap = (int *)calloc(lines, sizeof(int));

	if (centroidPos == NULL || centroids == NULL || classMap == NULL)
	{
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
	initCentroids(data, centroids, centroidPos, samples, K);

	printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
	printf("\tNumber of clusters: %d\n", K);
	printf("\tMaximum number of iterations: %d\n", maxIterations);
	printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
	printf("\tMaximum centroid precision: %f\n", maxThreshold);

	// END CLOCK*****************************************
	end = omp_get_wtime();
	printf("\nMemory allocation: %f seconds\n", end - start);
	fflush(stdout);
	//**************************************************
	// START CLOCK***************************************
	start = omp_get_wtime();
	//**************************************************
	char *outputMsg = (char *)calloc(10000, sizeof(char));
	char line[100];

	int j;
	int class;
	float dist, minDist;
	int it = 0;
	int changes = 0;
	float maxDist;

	// pointPerClass: number of points classified in each class
	// auxCentroids: mean of the points in each class
	int *pointsPerClass = (int *)malloc(K * sizeof(int));
	float *auxCentroids = (float *)malloc(K * samples * sizeof(float));
	float *distCentroids = (float *)malloc(K * sizeof(float));
	if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL)
	{
		fprintf(stderr, "Memory allocation error.\n");
		exit(-4);
	}

	/*
	 *
	 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
	 *
	 */

	int threads = 8;
	if (argc == 8)
	{
		threads = atoi(argv[7]);
	}
	omp_set_num_threads(threads);

	int rank, size, provided;
	MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
	// Check provided thread level
	if (provided < MPI_THREAD_FUNNELED)
	{
		fprintf(stderr, "Error: MPI does not provide required thread support level\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Bcast(centroids, K * samples, MPI_FLOAT, 0, MPI_COMM_WORLD);

	int local_lines = lines / size;

	float *local_data = (float *)malloc(local_lines * samples * sizeof(float));
	int *local_classMap = (int *)calloc(local_lines, sizeof(int));

	// Scatter the original data from the root (rank 0) to all processes
	MPI_Scatter(
		data,
		local_lines * samples,
		MPI_FLOAT,
		local_data,
		local_lines * samples,
		MPI_FLOAT,
		0,
		MPI_COMM_WORLD);

	// The root not need anymore data, all threads have local_data
	if (rank == 0)
	{
		free(data);
		data = NULL;
	}

	// Local buffers
	int *local_pointsPerClass = (int *)calloc(K, sizeof(int));
	float *local_auxCentroids = (float *)calloc(K * samples, sizeof(float));
	do
	{
		it++;

		// 1. Calculate the distance from each point to the centroid
		// Assign each point to the nearest centroid.
		int local_changes = 0;
#pragma omp parallel for shared(local_data, local_classMap, centroids) private(i, class, minDist, j, dist) reduction(+ : local_changes)
		for (i = 0; i < local_lines; i++)
		{
			class = 1;
			minDist = FLT_MAX;
			// Find the nearest centroid
			for (j = 0; j < K; j++)
			{
				dist = euclideanDistance(&local_data[i * samples], &centroids[j * samples], samples);
				if (dist < minDist)
				{
					minDist = dist;
					class = j + 1;
				}
			}
			if (local_classMap[i] != class)
			{
				local_changes++;
			}
			local_classMap[i] = class;
		}

		// 2. Recalculates the centroids: calculates the mean within each cluster
		zeroIntArray(local_pointsPerClass, K);
		zeroFloatMatriz(local_auxCentroids, K, samples);

#pragma omp parallel
		{
			int *thread_points = (int *)calloc(K, sizeof(int));
			float *thread_aux = (float *)calloc(K * samples, sizeof(float));

#pragma omp for
			for (int i = 0; i < local_lines; i++)
			{
				class = local_classMap[i];
				thread_points[class - 1] += 1;
				for (int j = 0; j < samples; j++)
				{
					thread_aux[(class - 1) * samples + j] += local_data[i * samples + j];
				}
			}

#pragma omp critical
			{
				for (int k = 0; k < K; k++)
				{
					local_pointsPerClass[k] += thread_points[k];
					for (int j = 0; j < samples; j++)
					{
						local_auxCentroids[k * samples + j] += thread_aux[k * samples + j];
					}
				}
			}
			free(thread_points);
			free(thread_aux);
		}

		MPI_Allreduce(&local_changes, &changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(local_pointsPerClass, pointsPerClass, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(local_auxCentroids, auxCentroids, K * samples, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

		// Calculate the mean for each centroid 
#pragma omp parallel for
		for (i = 0; i < K; i++)
		{
			for (j = 0; j < samples; j++)
			{
				auxCentroids[i * samples + j] /= (float)pointsPerClass[i];
			}
		}
		// Compute the maximum distance between old and new centroids
		float local_maxDist = 0.0f;
		//	#pragma omp parallel for reduction(max : maxDist)
		for (int i = 0; i < K; i++)
		{
			float d = euclideanDistance(&centroids[i * samples],
										&auxCentroids[i * samples],
										samples);

			if (d > local_maxDist) // Correct reduction usage
			{
				local_maxDist = d;
			}
		}

		MPI_Allreduce(&local_maxDist, &maxDist, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

		// Copy new centroids into the real centroids array
		memcpy(centroids, auxCentroids, (K * samples * sizeof(float)));

	} while ((changes > minChanges) && (it < maxIterations) && (maxDist > (maxThreshold * maxThreshold)));

	MPI_Gather(
		local_classMap, 
		local_lines,	
		MPI_INT,		
		classMap,		
		local_lines,	
		MPI_INT,		
		0,				
		MPI_COMM_WORLD);

	/*
	 *
	 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
	 *
	 */
	// Output and termination conditions
	printf("%s", outputMsg);

	// END CLOCK*****************************************
	end = omp_get_wtime();
	printf("\nComputation: %f seconds", end - start);
	fflush(stdout);
	//**************************************************
	// START CLOCK***************************************
	start = omp_get_wtime();
	//**************************************************

	if (rank == 0)
	{
		if (changes <= minChanges)
		{
			printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, minChanges);
		}
		else if (it >= maxIterations)
		{
			printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
		}
		else
		{
			printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
		}

		// Writing the classification of each point to the output file.
		error = writeResult(classMap, lines, argv[6]);
		if (error != 0)
		{
			showFileError(error, argv[6]);
			exit(error);
		}
	}

	// Free memory
	// Clean up
	free(local_data);
	free(local_classMap);
	free(local_pointsPerClass);
	free(local_auxCentroids);
	free(pointsPerClass);
	free(auxCentroids);
	free(distCentroids);
	free(centroids);

	free(classMap);
	MPI_Finalize(); // Finalize MPI

	// END CLOCK*****************************************
	end = omp_get_wtime();
	printf("\n\nMemory deallocation: %f seconds\n", end - start);
	fflush(stdout);
	//***************************************************/
	return 0;
}