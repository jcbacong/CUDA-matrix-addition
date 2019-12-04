#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CPU 
void randomArray(float *cpu_arrayA,float *cpu_arrayB, unsigned long SQWIDTH) {
	srand((unsigned) time(NULL));
	for(unsigned long i = 0; i<SQWIDTH*SQWIDTH; ++i){
		cpu_arrayA[i] = ((float)rand()/(float)(RAND_MAX)) * 100;
		cpu_arrayB[i] = ((float)rand()/(float)(RAND_MAX)) * 100;
		//printf("Matrx [%d][%d]: %.2f\n", i,j, cpu_array[i][j]);
	}
}

void printResults(float *h_matA, float *h_matB, float *h_matC,int SQWIDTH){
	printf("Matrix A:\n");
	for(int i=0; i< SQWIDTH*SQWIDTH; i++){
		// int id = i + floor(i / (int)SQWIDTH )* (int)SQWIDTH;
		printf("%.2f	", h_matA[i]);
		if( (i+1) % SQWIDTH  == 0 ){
			printf("\n");
		}

	}
	printf("Matrix B:\n");
	for(int i=0; i< SQWIDTH*SQWIDTH; i++){
		// int id = i + floor(i / (int)SQWIDTH )* (int)SQWIDTH;
		printf("%.2f	", h_matB[i]);
		if( (i+1) % SQWIDTH  == 0 ){
			printf("\n");
		}

	}
	printf("Matrix C:\n");
	for(int i=0; i< SQWIDTH*SQWIDTH; i++){
		// int id = i + floor(i / (int)SQWIDTH )* (int)SQWIDTH;
		printf("%.2f	", h_matC[i]);
		if( (i+1) % SQWIDTH  == 0 ){
			printf("\n");
		}

	}
}

// GPU
__global__ void kernel_1t1e(float *A, float *B, float *C, unsigned long WIDTH) {
	// To DO: Device a row major indexing
	int rowID = threadIdx.y + blockIdx.y * blockDim.y; 	// Row address
	int colID = threadIdx.x + blockIdx.x * blockDim.x;	// Column Address
	int elemID;											// Element address

    // a_ij = a[i][j], where a is in row major order
	if(rowID < WIDTH && colID < WIDTH){
		elemID = colID + rowID * WIDTH; 				
		C[elemID] = A[elemID] + B[elemID];
	}
}

__global__ void kernel_1t1r(float *A, float *B, float *C, unsigned long WIDTH) {
	// To DO: Each thread = 1 output row
	int rowID = threadIdx.y + blockIdx.y * blockDim.y;	// Row address

	if(rowID < WIDTH) {
		for(int i = 0; i<WIDTH; i++){
			//elemID = colID + rowID * WIDTH; 
			C[i + rowID*WIDTH] = A[i + rowID*WIDTH] + B[i + rowID*WIDTH];
		}
	}
}

__global__ void kernel_1t1c(float *A, float *B, float *C, unsigned long WIDTH) {
	// To DO: Each thread = 1 output row
	int colID = threadIdx.x + blockIdx.x * blockDim.x;	// Row address

	if(colID < WIDTH) {
		for(int i = 0; i<WIDTH; i++){
			//elemID = colID + rowID * WIDTH; 
			C[colID + i*WIDTH] = A[colID + i*WIDTH] + B[colID + i*WIDTH];
		}
	}
}


int main(int argv, int* argc[]) {
	// Memory specification
	unsigned long SQWIDTH;
	scanf("%lu", &SQWIDTH);

	const size_t d_size = sizeof(float) * size_t(SQWIDTH*SQWIDTH);

	// Multiprocessing constants
	const dim3 threadsPerBlock(32,32); 	// Must not exceed 1024 (max thread per block)
	const dim3 blocksPerGrid(ceil(SQWIDTH/32),ceil(SQWIDTH/32));		// Number of blocks that will be used

	// CUDA TIME
	float ms;
	float avems = 0.0;
	cudaEvent_t start,end;



	// Initialize host matrices
	clock_t h_alloctime = clock();
	float *h_matA = (float*) malloc(SQWIDTH*SQWIDTH * sizeof(float));
	float *h_matB = (float*) malloc(SQWIDTH*SQWIDTH * sizeof(float));
	float *h_matC = (float*) malloc(SQWIDTH*SQWIDTH * sizeof(float));
	randomArray(h_matA, h_matB, SQWIDTH);
	printf("[**] CPU Allocation time for %dx%d matrix: %.6fsec \n",SQWIDTH,SQWIDTH,(double)(clock()-h_alloctime)/CLOCKS_PER_SEC );
	// Initialize device matrices
	float *d_matA, *d_matB, *d_matC;
	
	clock_t d_alloctime = clock();
	cudaMalloc((void **) &d_matA, d_size);
	cudaMalloc((void **) &d_matB, d_size);
	cudaMalloc((void **) &d_matC, d_size);
	cudaMemcpy(d_matA, h_matA, d_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matB, h_matB, d_size, cudaMemcpyHostToDevice); 
	printf("[**] GPU Allocation time for %lux%lu matrix: %.6fsec \n",SQWIDTH,SQWIDTH,(double)(clock()-d_alloctime)/CLOCKS_PER_SEC );


	// Number of threads = SQWIDTH*SQWIDTH
	printf("[**] Starting kernel program 'kernel_1t1e' execution\n");
	for(int i = 0; i<10; i++){
		// ELEMENT
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start, 0);

		kernel_1t1e<<< blocksPerGrid, threadsPerBlock >>>(d_matA,d_matB,d_matC,SQWIDTH);


		cudaEventRecord(end, 0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&ms, start, end);

		printf("\tIteration no. %d: %.6fsecs\n", i, ms);
		avems+=ms;
		cudaMemcpy(h_matC, d_matC, d_size, cudaMemcpyDeviceToHost); 

		cudaEventDestroy(start);
		cudaEventDestroy(end);
	}
	printf("[**] Average kernel execution time: %.2fsec.\n\n", avems/10.0);
	

	printf("[] Starting kernel program 'kernel_1t1r'.execution\n");
	avems = 0.0;
	for(int i = 0; i<10; i++){
		// ELEMENT
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start, 0);

		kernel_1t1r<<< blocksPerGrid, threadsPerBlock >>>(d_matA,d_matB,d_matC,SQWIDTH);


		cudaEventRecord(end, 0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&ms, start, end);

		printf("\tIteration no. %d: %.6fsecs\n", i, ms);
		avems+=ms;
		cudaMemcpy(h_matC, d_matC, d_size, cudaMemcpyDeviceToHost); 
		cudaEventDestroy(start);
		cudaEventDestroy(end);
	}
	printf("[**] Average kernel execution time: %.2fsec.\n\n", avems/10.0);
	
	printf("[**] Starting kernel program 'kernel_1t1c' execution\n");
	avems = 0;
	for(int i = 0; i<10; i++){
		// ELEMENT
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start, 0);

		kernel_1t1c<<< blocksPerGrid, threadsPerBlock >>>(d_matA,d_matB,d_matC,SQWIDTH);


		cudaEventRecord(end, 0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&ms, start, end);

		printf("\tIteration no. %d: %.6fsecs\n", i, ms);
		avems+=ms;
		cudaMemcpy(h_matC, d_matC, d_size, cudaMemcpyDeviceToHost); 

		cudaEventDestroy(start);
		cudaEventDestroy(end);
	}
	printf("[**] Average kernel execution time: %.2fsec.\n", avems/10.0);
	
	cudaFree(d_matA);
	cudaFree(d_matB);
	cudaFree(d_matC);
	free(h_matA);
	free(h_matB);
	free(h_matC);

	return 0;
}
