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
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <cuda.h>
#include <cooperative_groups.h>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

#define MAXLINE 2000
#define MAXCAD 200

// Macros
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/* 
Function showFileError: It displays the corresponding error during file reading.
*/
void showFileError(int error, char* filename) {
    printf("Error\n");
    switch (error) {
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
int readInput(char* filename, int *N, int *D) {
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int contN = 0, contD = 0;

    if ((fp = fopen(filename, "r")) != NULL) {
        while (fgets(line, MAXLINE, fp) != NULL) {
            if (strchr(line, '\n') == NULL) {
                return -1;
            }
            contN++;
            ptr = strtok(line, delim);
            contD = 0;
            while (ptr != NULL) {
                contD++;
                ptr = strtok(NULL, delim);
            }
        }
        fclose(fp);
        *N = contN;
        *D = contD;
        return 0;
    } else {
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

    if ((fp = fopen(filename, "rt")) != NULL) {
        while (fgets(line, MAXLINE, fp) != NULL) {
            ptr = strtok(line, delim);
            while (ptr != NULL) {
                data[i] = atof(ptr);
                i++;
                ptr = strtok(NULL, delim);
            }
        }
        fclose(fp);
        return 0;
    } else {
        return -2;
    }
}

/* 
Function writeResult: It writes in the output file the cluster of each sample (point).
*/
int writeResult(int* classMap, int N, const char* filename) {
    FILE *fp;
    if ((fp = fopen(filename, "wt")) != NULL) {
        for (int i = 0; i < N; i++) {
            fprintf(fp, "%d\n", classMap[i]);
        }
        fclose(fp);
        return 0;
    } else {
        return -3;
    }
}

/* 
Function initCentroids
*/
void initCentroids(const float* data, float* centroids, int* centroidPos, int D, int K) {
    for (int i = 0; i < K; i++) {
        int idx = centroidPos[i];
        memcpy(&centroids[i * D], &data[idx * D], D * sizeof(float));
    }
}

/* 
Euclidean Distance
*/
float euclideanDistance(float *point, float *center, int D) {
    float dist = 0.0;
    for (int i = 0; i < D; i++) {
        dist += (point[i] - center[i]) * (point[i] - center[i]);
    }
    return dist;
}

/* 
Zero Float Matrix
*/
void zeroFloatMatriz(float *matrix, int rows, int columns) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < columns; j++)
            matrix[i * columns + j] = 0.0;
}

/* 
Zero Int Array
*/
void zeroIntArray(int *array, int size) {
    for (int i = 0; i < size; i++)
        array[i] = 0;
}

/* 
Transpose a matrix
*/
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

// Allocate constant memory on the device
__constant__ int const_N;
__constant__ int const_K;
__constant__ int const_D;
__constant__ int const_maxIterations;
__constant__ int const_minChanges;
__constant__ float const_maxThreshold;
__constant__ int const_IpTN;
__constant__ int const_RN;
__constant__ int const_IpTK;
__constant__ int const_RK;
__constant__ int const_TotThreads;

__global__ void kmeans(float*, float*, int*, int*, int*, float*, float*, float*, float*, int*, float*, float*, int*);

__device__ void get_IpTpos(int ID, int IpT, int R, int * myIpT,int * posP);


int main(int argc, char* argv[]) {
    clock_t start, end;
    start = clock();

    if (argc != 7) {
        fprintf(stderr, "EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
        fprintf(stderr, "./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
        fflush(stderr);
        exit(-1);
    }

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess || deviceCount == 0) {
        printf("Failed to get CUDA device count or no CUDA device available.\n");
        return 1;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("\nDevice %d: %s\n", 0, deviceProp.name);
    printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("  Multiprocessors (SMs): %d\n", deviceProp.multiProcessorCount);
    printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("  Max threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("  Warp size: %d\n", deviceProp.warpSize);

    int N = 0, D = 0;
    int error = readInput(argv[1], &N, &D);
    if (error != 0) {
        showFileError(error, argv[1]);
        exit(error);
    }

    float* points = (float*) calloc(N * D, sizeof(float));
    if (points == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(-4);
    }

    error = readInput2(argv[1], points);
    if (error != 0) {
        showFileError(error, argv[1]);
        exit(error);
    }

    int K = atoi(argv[2]);
    int maxIterations = atoi(argv[3]);
    int minChanges = (int)(N * atof(argv[4]) / 100.0);
    float maxThreshold = atof(argv[5]);

    int* centroidPos = (int*) calloc(K, sizeof(int));
    float* centroids = (float*) calloc(K * D, sizeof(float));
    int* classMap = (int*) calloc(N, sizeof(int));

    if (centroidPos == NULL || centroids == NULL || classMap == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(-4);
    }

    // Initial centrodis
    srand(0);
    for (int i = 0; i < K; i++) 
        centroidPos[i] = rand() % N;

    initCentroids(points, centroids, centroidPos, D, K);

    printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], N, D);
    printf("\tNumber of clusters: %d\n", K);
    printf("\tMaximum number of iterations: %d\n", maxIterations);
    printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), N);
    printf("\tMaximum centroid precision: %f\n", maxThreshold);

    //Make centroids and points transpose
    //This is done to make the memory access coalesced
    // Transpose and free original data
    float *transposed_points = transpose(points, N, D);
    free(points);  // Free original allocation
    points = transposed_points;

    float *transposed_centroids = transpose(centroids, K, D);
    free(centroids);  // Free original allocation
    centroids = transposed_centroids;
    //Now we have DxN arrays

    //SETTING CUDA
    int TpB = deviceProp.maxThreadsPerBlock;
    int SMs = deviceProp.multiProcessorCount;

    // Set GRID and BLOCK size
    dim3 BlockSize = TpB; //Threads per Block
    dim3 numBlocks = SMs;
    int TotThreads = TpB*SMs;

    //ALLOCATE MEMORY FOR CUDA
    // Allocate memory on the device
    float *d_points;
    float *d_centroids;
    int *d_classMap;
    int *d_centroidPos;
    int *d_pointsPerClass;
    float *d_auxCentroids;
    float *d_distCentroids;
    float *d_dimdistKN;
    float *d_distKN;
    int *d_bchanges;
    float *d_dimdistK;
    float *d_distK;
    int *d_it;




    CUDA_CHECK(cudaMalloc((void**)&d_points, N*D*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_centroids, K*D*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_classMap, N*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_centroidPos, K*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_pointsPerClass, K*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_auxCentroids, K*D*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_distCentroids, K*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dimdistKN,  D*K*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_distKN,  K*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_bchanges, SMs*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_dimdistK,  D*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_distK,  K*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_it,  sizeof(int)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_points, points, N*D*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, centroids, K*D*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(const_N, &N, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_K, &K, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_D, &D, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_maxIterations, &maxIterations, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_minChanges, &minChanges, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_maxThreshold, &maxThreshold, sizeof(float)));

    int IpTN = (N*D)/(SMs*TpB); // Iterations per Threads 
    int RN = (N*D)%(SMs*TpB); //  Iterations per Threads rest
    int IpTK = (K*D)/(SMs*TpB); // Iterations per Threads
    int RK = (K*D)%(SMs*TpB); //  Iterations per Threads rest

    CUDA_CHECK(cudaMemcpyToSymbol(const_IpTN, &IpTN, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_RN, &RN, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_IpTK, &IpTK, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_RK, &RK, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_TotThreads, &TotThreads, sizeof(int)));

    void *kernelArgs[] = {
        d_points,
        d_centroids,
        d_classMap,
        d_centroidPos,
        d_pointsPerClass,
        d_auxCentroids,
        d_distCentroids,
        d_bchanges,
        d_dimdistKN,
        d_distKN,
        d_dimdistK,
        d_distK,
        d_it
    };



    end = clock();
    printf("\nMemory allocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    fflush(stdout);
    start = clock();
    char *outputMsg = (char *)calloc(100000,sizeof(char));
    char line[1000];


    //  START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
    printf("\nLaunching kernel with %d blocks and %d threads per block...\n\n", deviceProp.multiProcessorCount , deviceProp.maxThreadsPerBlock);

    size_t sharedMemSize; ///////////////////////////////////////////////////////
    // Launch kernel
    cudaLaunchCooperativeKernel((void*)kmeans, numBlocks, BlockSize, kernelArgs, sharedMemSize);
    CUDA_CHECK(cudaDeviceSynchronize())

    float maxDist;
    int changes;
    int it;

    CUDA_CHECK(cudaMemcpy(centroids, d_centroids, K*D*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(classMap, d_classMap, N*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&maxDist, d_distK, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&changes, d_bchanges, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&it, d_it, sizeof(int), cudaMemcpyDeviceToHost));

    // Ripetuto a riga 290
    // float* transposed_centroids = transpose(centroids, K, D);

    free(centroids);  // Free original allocation
    centroids = transposed_centroids;
   
/*
*
* STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
*
*/


    printf("%s",outputMsg);	

    end = clock();
    printf("\nComputation: %f seconds", (double)(end - start) / CLOCKS_PER_SEC);
    fflush(stdout);
    start = clock();

    if (changes <= minChanges) {
        printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, minChanges);
    }
    else if (it >= maxIterations) {
        printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
    }
    else {
        printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
    }	

    error = writeResult(classMap, N, argv[6]);
    if(error != 0)
    {
        showFileError(error, argv[6]);
        exit(error);
    }

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_classMap));
    CUDA_CHECK(cudaFree(d_centroidPos));
    CUDA_CHECK(cudaFree(d_pointsPerClass));
    CUDA_CHECK(cudaFree(d_auxCentroids));
    CUDA_CHECK(cudaFree(d_distCentroids));
    CUDA_CHECK(cudaFree(d_dimdistKN));
    CUDA_CHECK(cudaFree(d_distKN));
    CUDA_CHECK(cudaFree(d_bchanges));
    CUDA_CHECK(cudaFree(d_dimdistK));
    CUDA_CHECK(cudaFree(d_distK));
    CUDA_CHECK(cudaFree(d_it));
    
    free(points);
    free(classMap);
    free(centroidPos);
    free(centroids);
    // free(distCentroids);
    // free(pointsPerClass);
    // free(auxCentroids);

    end = clock();
    printf("\n\nMemory deallocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    fflush(stdout);
    return 0;
}

__device__ void get_IpTpos(int ID, int IpT, int R, int * myIpT,int * pos){
    *myIpT = IpT;
    *pos = 0;

    if (ID < R) {
        (*myIpT)++;
        *pos = IpT * ID;
    } else {
        *pos = R * (IpT + 1) + IpT * (ID - R);
    }
}


// KERNEL
__global__ void kmeans(
    float* d_points,
    float* d_centroids,
    int* d_classMap,
    int* d_centroidPos,
    int* d_pointsPerClass,
    float* d_auxCentroids,
    float* d_distCentroids,
    float *d_dimdistKN,
    float *d_distKN,
    int* d_bchanges,
    float *d_dimdistK,
    float *d_distK,
    int *d_it) {


    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    g.sync();
    int myId = blockIdx.x * blockDim.x + threadIdx.x ;

    //sec 0
    int myIpTN;
    int posP;
    // position in points
    get_IpTpos(myId, const_IpTN, const_RN, &myIpTN, &posP);

    extern __shared__ float sh_centroids[];

    int t1 = blockIdx.x * blockDim.x;
    int t1_myIpT, t1_posP;
    int t2 = blockIdx.x * blockDim.x + blockDim.x;
    int t2_myIpT, t2_posP;

                                // Prima era RN ↓
    get_IpTpos(t1, const_IpTN, const_RN, &t1_myIpT, &t1_posP); 
    get_IpTpos(t2, const_IpTN, const_RN, &t2_myIpT, &t2_posP);

    int d1= t1_posP / const_N;
    int d2= t2_posP / const_N;
    int total = (d2+1) * const_K - d1 * const_K;
    int tocopyvals = total / blockDim.x;
    int rest = total % blockDim.x;
    int posC;
    if (threadIdx.x < rest){
        tocopyvals++;
        posC =  threadIdx.x * tocopyvals;
    }else{
        posC =  rest * (tocopyvals+1) * ((threadIdx.x-rest) * tocopyvals);
    }

    int d, n, idx;
    int posk;
    int dist;

    //sec 1
    int times_1 = const_N*const_D / const_TotThreads;
    int rest_1 = const_N*const_D % const_TotThreads;
    if (myId < rest_1) {

        // Cosa va fatto qua?
        times_1 += 1;
    }
    float agg = 0.0f; 

    //sec 2
    int times_2 = n / const_TotThreads;
    int rest_2 = n % const_TotThreads;
    if (myId < rest_2) { 

        // Cosa va fatto qua?
        times_2 += 1;
    }
    float mindist;
    float agg_dist;
    int cluster_class;

    extern __shared__ int sh_changes[]; //BlockDim.x

    //sec3
    int times_3 = const_D*const_K / const_TotThreads;
    int rest_3 = const_D*const_K % const_TotThreads;
    if (myId < rest_3) { 

        // Cosa va fatto qua?
        times_3 += 1;
    }

    extern __shared__ int sh_pointsPerClass[]; //K

    extern __shared__ int sh_distK[]; //K


    //sec 4
    int myIpTK;
    int posK;
    //position in centroids
    get_IpTpos(myId, const_IpTK, const_RK, &myIpTK, &posK);

    int times_4 = const_K / gridDim.x;
    int rest_4 = const_K % gridDim.x;
    if (myId < rest_4) {

        // Cosa va fatto qua?
        times_4 += 1;
    }
    int posT= threadIdx.x+(blockDim.x*32);

    float maxDist;
    int changes;
    int it= 0;   
    int i, k, t;
    do { 
        //sec 0 
        it++;
        for (i=0; i<tocopyvals; i++){
            sh_centroids[posC + i] = d_centroids[d1*const_K + posC + i];
        }
        __syncthreads();

        for (i=0; i<myIpTN; i++){
            d = (posP + i) / const_N;
            n = (posP + i) % const_N;
            posk = (d * const_K) - (d1 * const_K);
            for (k=0; k<const_K; k++){
                dist = d_points[posP + i] - sh_centroids[posk + k];
                idx= (d*const_N*const_K) + (k*const_N) + n;
                d_dimdistKN[idx] = dist*dist;
            }
        }
        __syncthreads();

        //section1
        for (t=0; t < times_1; t++){
            k = myId / const_N;
            n = (t+1) * (myId % const_N);
            agg= 0.0f;
            for (d=0; d < const_D; d++){
                agg += d_dimdistKN[d * const_K * const_N + k * const_N + n];
            }
            d_distKN[(t+1)*(myId % const_N)] = agg;
        }
        __syncthreads();

        //section2

        changes = 0; 
        for (t=0; t<times_2; t++){
            mindist = FLT_MAX;
            cluster_class = 1;
            for (k=0; k<const_K; k++){
                agg_dist = d_distKN[k*n + ((t+1)*myId)];
                if (agg_dist < mindist) {
                    mindist =  agg_dist;
                    cluster_class = k+1;
                }
            }

            if (d_classMap[(t+1)*myId] != cluster_class) {
                d_classMap[(t+1)*myId] = cluster_class;
                changes++;
            }
        }

        sh_changes[threadIdx.x] = changes;
        __syncthreads();

        //INTER BLOCK REDUCTION
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                sh_changes[threadIdx.x] += sh_changes[threadIdx.x + stride];
            }
            __syncthreads();
        }
        
        if (threadIdx.x == 0) {
            d_bchanges[blockIdx.x] =  sh_changes[0];
        }
        //INTRA BLOCKS REDUCTION
        g.sync();
        if (blockIdx.x == 0){
            if (threadIdx.x < gridDim.x) {
                sh_changes[threadIdx.x] = d_bchanges[threadIdx.x];
            } 
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride && threadIdx.x + stride < gridDim.x) {
                    sh_changes[threadIdx.x] += sh_changes[threadIdx.x + stride];
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                d_bchanges[0] = sh_changes[0];
            }
        }

        //sec 3
        if (threadIdx.x<const_K) {
            sh_pointsPerClass[threadIdx] = 0;
            if (blockDim.x==blockIdx.x){
                d_pointsPerClass[threadIdx] = 0;
            }
        }
        
        for (t=0; t<times_3; t++){
            d_auxCentroids[(t+1)*myId] = 0;
        }

        g.sync();
        
        for (i=0; i<myIpTN; i++){
            d = (posP + i) / const_N;
            n = (posP + i) % const_N;
            k = d_classMap[n] - 1 ;
            atomicAdd(d_auxCentroids[k * d + d], d_points[posP + i]);
            if (d == const_D-1){
                atomicAdd(&sh_pointsPerClass[k], 1);
            }
        }
        __syncthreads();
        if (threadIdx.x<const_K){
            atomicAdd(d_pointsPerClass[threadIdx.x], sh_pointsPerClass[threadIdx.x]);
        }

        g.sync();

        if (threadIdx.x<const_K){
            sh_pointsPerClass[threadIdx.x]= d_pointsPerClass[threadIdx.x];
        }

        __syncthreads();

        for (i=0; i<myIpTK; i++){
            d = (posK + i) / const_K;
            k = (posK + i) % const_K;
            if (sh_pointsPerClass[k] > 0) {  // Prevent division by zero
                d_auxCentroids[posK + i] /= sh_pointsPerClass[k];
            }
            dist = d_centroids[posK + i] - d_auxCentroids[posK + k];
            d_dimdistK[posK + i] = dist*dist;
            d_centroids[posK + i] = d_auxCentroids[posK+ i];
        }
        g.sync();


        if (threadIdx.x<32 && posT<const_K){
            
            for (d=0; d<const_D; d++){
                if (d==0){
                    d_distK[posT] = 0;
                }
                d_distK[posT] += d_dimdistK[const_K*d + posT];
            }
        }

        g.sync();

        if (blockIdx.x == 0){
            if (threadIdx.x < const_K) {
                sh_distK[threadIdx.x] = d_distK[threadIdx.x];
            } 
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride && threadIdx.x + stride < const_K) {
                    sh_distK[threadIdx.x] += sh_distK[threadIdx.x + stride];
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                d_distK[0] = sh_distK[0];
            }
        }

        g.sync();

        maxDist = d_distK[0];
        changes = d_bchanges[0];
    } while ((changes > const_minChanges) && (it < const_maxIterations) && (maxDist > const_maxThreshold*const_maxThreshold));

    if (myId==0){
        d_it[0] = it;
    }
}