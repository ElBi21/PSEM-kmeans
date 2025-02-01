/*
 * k-Means clustering algorithm
 *
 * MPI+PThreads version
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2022/2023
 *
 * Version: 1.1
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
#include <pthread.h>

#define MAXLINE 2000
#define MAXCAD 200

// Macros
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

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
        return -2; // File not found
    }
}

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
        return -3; // File open error
    }
}

void initCentroids(const float *data, float *centroids, int *centroidPos, int samples, int K)
{
    int i, idx;
    for (i = 0; i < K; i++)
    {
        idx = centroidPos[i];
        memcpy(&centroids[i * samples], &data[idx * samples], (samples * sizeof(float)));
    }
}

float euclideanDistance(float *point, float *center, int samples)
{
    float dist = 0.0;
    for (int i = 0; i < samples; i++)
    {
        dist += (point[i] - center[i]) * (point[i] - center[i]);
    }
    return dist; // Squared distance
}

void zeroFloatMatriz(float *matrix, int rows, int columns)
{
    memset(matrix, 0, rows * columns * sizeof(float));
}

void zeroIntArray(int *array, int size)
{
    memset(array, 0, size * sizeof(int));
}

/* ============================================
   Pthreads data structures and functions
   ============================================ */

/* --- Structure and function for Step 1: Assigning points to centroids --- */
typedef struct
{
    int thread_id;
    int num_threads;
    int local_n;
    int D;
    int K;
    float *local_points;
    int *local_classMap;
    float *centroids;
    int local_changes; // To accumulate number of changed assignments in this thread
    int start;         // starting index in local_points array (each point has D values)
    int end;           // one past the last index (exclusive)
} assign_thread_data_t;

void *assign_points_thread(void *arg)
{
    assign_thread_data_t *data = (assign_thread_data_t *)arg;
    int start = data->start;
    int end = data->end;
    int D = data->D;
    int K = data->K;
    data->local_changes = 0;
    for (int i = start; i < end; i++)
    {
        int new_class = 1;
        float minDist = FLT_MAX;
        for (int j = 0; j < K; j++)
        {
            float dist = 0.0f;
            for (int d = 0; d < D; d++)
            {
                float diff = data->local_points[i * D + d] - data->centroids[j * D + d];
                dist += diff * diff;
            }
            if (dist < minDist)
            {
                minDist = dist;
                new_class = j + 1; // classes are 1-indexed
            }
        }
        if (data->local_classMap[i] != new_class)
        {
            data->local_changes++;
        }
        data->local_classMap[i] = new_class;
    }
    return NULL;
}

/* --- Structure and function for Step 2: Accumulating centroid updates --- */
typedef struct
{
    int thread_id;
    int num_threads;
    int local_n;
    int D;
    int K;
    float *local_points;
    int *local_classMap;
    int *partial_counts;      // This thread’s partial count for each centroid (length K)
    float *partial_centroids; // This thread’s partial sums for each centroid (length K*D)
    int start;                // starting index in local_points for this thread
    int end;                  // ending index (exclusive)
} centroid_thread_data_t;

void *centroid_accumulate_thread(void *arg)
{
    centroid_thread_data_t *data = (centroid_thread_data_t *)arg;
    int start = data->start;
    int end = data->end;
    int D = data->D;
    int K = data->K;
    // Initialize the thread’s partial arrays to zero
    for (int k = 0; k < K; k++)
    {
        data->partial_counts[k] = 0;
        for (int d = 0; d < D; d++)
        {
            data->partial_centroids[k * D + d] = 0.0f;
        }
    }
    // Accumulate contributions from the assigned points
    for (int i = start; i < end; i++)
    {
        int cls = data->local_classMap[i]; // classes are 1-indexed
        int idx = cls - 1;
        data->partial_counts[idx]++;
        for (int d = 0; d < D; d++)
        {
            data->partial_centroids[idx * D + d] += data->local_points[i * D + d];
        }
    }
    return NULL;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    double start, end;
    start = MPI_Wtime();

    if ((argc != 7) && !(argc == 8))
    {
        if (rank == 0)
        {
            fprintf(stderr, "EXECUTION ERROR MPI+OpenMP: Parameters are not correct.\n");
            fprintf(stderr, "./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file] Optional: [Number of Threads]\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Set the number of Pthreads threads
	int threads = 8;
	if (argc == 8)
	{
		threads = atoi(argv[7]); // Set thread count from command-line argument
	}

    int N = 0, D = 0;
    float *points = NULL;
    if (rank == 0)
    {
        int error = readInput(argv[1], &N, &D);
        if (error != 0)
        {
            showFileError(error, argv[1]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        points = (float *)calloc(N * D, sizeof(float));
        if (points == NULL)
        {
            fprintf(stderr, "Memory allocation error.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        error = readInput2(argv[1], points);
        if (error != 0)
        {
            showFileError(error, argv[1]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int K = atoi(argv[2]);
    int maxIterations = atoi(argv[3]);
    int minChanges = (int)(N * atof(argv[4]) / 100.0);
    float maxThreshold = atof(argv[5]);

    float *centroids = (float *)calloc(K * D, sizeof(float));
    if (centroids == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    int *classMap = NULL;
    if (rank == 0)
    {
        int *centroidPos = (int *)calloc(K, sizeof(int));
        classMap = (int *)calloc(N, sizeof(int));
        if (centroidPos == NULL || classMap == NULL)
        {
            fprintf(stderr, "Memory allocation error.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
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
    MPI_Bcast(centroids, K * D, MPI_FLOAT, 0, MPI_COMM_WORLD);

    end = MPI_Wtime();
    printf("\n%d |Memory allocation: %f seconds\n", rank, end - start);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    char *outputMsg = NULL;
    if (rank == 0)
    {
        outputMsg = (char *)calloc(10000, sizeof(char));
    }

    int it = 0;
    int changes;
    float maxDist;
    int *pointsPerClass = (int *)malloc(K * sizeof(int));
    float *auxCentroids = (float *)malloc(K * D * sizeof(float));
    if (pointsPerClass == NULL || auxCentroids == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* === Prepare data distribution among MPI processes === */
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    int remainder = N % size;
    int sum = 0;
    for (int i = 0; i < size; ++i)
    {
        sendcounts[i] = (N / size) * D;
        if (i < remainder)
            sendcounts[i] += D;
        displs[i] = sum;
        sum += sendcounts[i];
    }
    int local_n = sendcounts[rank] / D;
    float *local_points = (float *)calloc(local_n * D, sizeof(float));
    int *local_classMap = (int *)calloc(local_n, sizeof(int));
    if (local_points == NULL || local_classMap == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    MPI_Scatterv(points, sendcounts, displs, MPI_FLOAT, local_points, sendcounts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    int *centroid_sendcounts = (int *)malloc(size * sizeof(int));
    int *centroid_displs = (int *)malloc(size * sizeof(int));
    int centroid_remainder = K % size;
    sum = 0;
    for (int i = 0; i < size; ++i)
    {
        centroid_sendcounts[i] = (K / size) * D;
        if (i < centroid_remainder)
            centroid_sendcounts[i] += D;
        centroid_displs[i] = sum;
        sum += centroid_sendcounts[i];
    }
    int local_k = centroid_sendcounts[rank] / D;
    float *local_centroids = (float *)calloc(local_k * D, sizeof(float));
    if (local_centroids == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* === Main iteration loop === */
    do
    {
        it++;

        /* -----------------------------------------------------
           STEP 1: Assign points to nearest centroid (using Pthreads)
           ----------------------------------------------------- */
        pthread_t assign_threads[threads];
        assign_thread_data_t assign_data[threads];
        int points_per_thread = local_n / threads;
        int extra = local_n % threads;
        int current_start = 0;
        for (int t = 0; t < threads; t++)
        {
            assign_data[t].thread_id = t;
            assign_data[t].num_threads = threads;
            assign_data[t].local_n = local_n;
            assign_data[t].D = D;
            assign_data[t].K = K;
            assign_data[t].local_points = local_points;
            assign_data[t].local_classMap = local_classMap;
            assign_data[t].centroids = centroids;
            assign_data[t].start = current_start;
            int count = points_per_thread + (t < extra ? 1 : 0);
            assign_data[t].end = current_start + count;
            current_start += count;
            pthread_create(&assign_threads[t], NULL, assign_points_thread, &assign_data[t]);
        }
        int local_changes = 0;
        for (int t = 0; t < threads; t++)
        {
            pthread_join(assign_threads[t], NULL);
            local_changes += assign_data[t].local_changes;
        }

        /* Nonblocking reduction to sum changes across processes */
        MPI_Request MPI_REQUEST;
        MPI_Iallreduce(&local_changes, &changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &MPI_REQUEST);

        /* -----------------------------------------------------
           STEP 2: Recalculate centroids (accumulate local contributions)
           ----------------------------------------------------- */
        // We use threads to accumulate contributions from local points.
        // Each thread will have its own partial arrays.
        int *partial_counts_all = malloc(threads * K * sizeof(int));
        float *partial_centroids_all = malloc(threads * K * D * sizeof(float));
        pthread_t centroid_threads[threads];
        centroid_thread_data_t centroid_data[threads];
        int points_per_thread2 = local_n / threads;
        extra = local_n % threads;
        current_start = 0;
        for (int t = 0; t < threads; t++)
        {
            centroid_data[t].thread_id = t;
            centroid_data[t].num_threads = threads;
            centroid_data[t].local_n = local_n;
            centroid_data[t].D = D;
            centroid_data[t].K = K;
            centroid_data[t].local_points = local_points;
            centroid_data[t].local_classMap = local_classMap;
            centroid_data[t].partial_counts = partial_counts_all + t * K;
            centroid_data[t].partial_centroids = partial_centroids_all + t * K * D;
            centroid_data[t].start = current_start;
            int count = points_per_thread2 + (t < extra ? 1 : 0);
            centroid_data[t].end = current_start + count;
            current_start += count;
            pthread_create(&centroid_threads[t], NULL, centroid_accumulate_thread, &centroid_data[t]);
        }
        for (int t = 0; t < threads; t++)
        {
            pthread_join(centroid_threads[t], NULL);
        }
        // Combine partial results from threads into pointsPerClass and auxCentroids
        zeroIntArray(pointsPerClass, K);
        zeroFloatMatriz(auxCentroids, K, D);
        for (int t = 0; t < threads; t++)
        {
            for (int k = 0; k < K; k++)
            {
                pointsPerClass[k] += partial_counts_all[t * K + k];
                for (int d = 0; d < D; d++)
                {
                    auxCentroids[k * D + d] += partial_centroids_all[t * K * D + k * D + d];
                }
            }
        }
        free(partial_counts_all);
        free(partial_centroids_all);

        /* MPI_Allreduce the centroid accumulations across processes */
        MPI_Allreduce(MPI_IN_PLACE, pointsPerClass, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, auxCentroids, K * D, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        /* -----------------------------------------------------
           STEP 3: Update centroids and check convergence
           ----------------------------------------------------- */
        float local_maxDist = 0.0f;
        // Each process is assigned a portion of the centroids for update.
        for (int i = 0; i < local_k; i++)
        {
            int global_idx = centroid_displs[rank] / D + i;
            if (global_idx >= K)
                break;
            float distance = 0.0f;
            if (pointsPerClass[global_idx] == 0)
                continue;
            for (int j = 0; j < D; j++)
            {
                float new_val = auxCentroids[global_idx * D + j] / pointsPerClass[global_idx];
                float diff = centroids[global_idx * D + j] - new_val;
                distance += diff * diff;
                local_centroids[i * D + j] = new_val;
            }
            if (distance > local_maxDist)
                local_maxDist = distance;
        }
        MPI_Allreduce(&local_maxDist, &maxDist, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allgatherv(local_centroids, local_k * D, MPI_FLOAT,
                       centroids, centroid_sendcounts, centroid_displs, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Wait(&MPI_REQUEST, MPI_STATUS_IGNORE);

    } while ((changes > minChanges) && (it < maxIterations) && (maxDist > maxThreshold * maxThreshold));

    /* === Gather final class assignments === */
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *rdispls = (int *)malloc(size * sizeof(int));
    sum = 0;
    for (int i = 0; i < size; ++i)
    {
        recvcounts[i] = sendcounts[i] / D;
        rdispls[i] = sum;
        sum += recvcounts[i];
    }
    MPI_Gatherv(local_classMap, local_n, MPI_INT,
                classMap, recvcounts, rdispls, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("%s", outputMsg);
    }

    end = MPI_Wtime();
    double computation_time = end - start;
    double max_computation_time;
    MPI_Reduce(&computation_time, &max_computation_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        printf("\n Computation: %f seconds\n", max_computation_time);
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    if (rank == 0)
    {
        if (changes <= minChanges)
        {
            printf("\n\nTermination condition: Minimum number of changes reached: %d [%d]", changes, minChanges);
        }
        else if (it >= maxIterations)
        {
            printf("\n\nTermination condition: Maximum number of iterations reached: %d [%d]", it, maxIterations);
        }
        else
        {
            printf("\n\nTermination condition: Centroid update precision reached: %g [%g]", maxDist, maxThreshold);
        }
        int error = writeResult(classMap, N, argv[6]);
        if (error != 0)
        {
            showFileError(error, argv[6]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        fflush(stdout);
    }

    /* Free memory */
    free(local_points);
    free(local_classMap);
    free(local_centroids);
    free(sendcounts);
    free(displs);
    free(centroid_sendcounts);
    free(centroid_displs);
    free(recvcounts);
    free(rdispls);
    if (rank == 0)
    {
        free(points);
        free(classMap);
        free(outputMsg);
    }
    free(centroids);
    free(pointsPerClass);
    free(auxCentroids);

    end = MPI_Wtime();
    printf("\n\n%d |Memory deallocation: %f seconds\n", rank, end - start);
    fflush(stdout);

    MPI_Finalize();
    return 0;
}