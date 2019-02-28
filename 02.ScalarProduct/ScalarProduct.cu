#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <iostream>
#include <string>
#include <time.h>
#include <numeric> 

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define THREADS_PER_BLOCK 512

#define CHECK(call) \
do { \
	if (cudaSuccess != call) { \
		fprintf(stderr, ("CUDA ERROR! file: %s[%i] -> %s\n"), __FILE__, __LINE__, cudaGetErrorString(call)); \
		exit(0); \
	} \
} while (0)

void checkResult(float *hostRef, float *gpuRef, const int N)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	if (abs(hostRef - gpuRef) > epsilon)
	{
		match = 0;
		printf("Arrays do not match!\n");
		printf("host %5.2f gpu %5.2f", hostRef,
			gpuRef);
	}
	if (match) printf("Arrays match.\n\n");

	return;
}

__global__ void scalarProductOnDevice(float *A, float *B, float *C, const int N)
{
	// multiplication 
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int temp[THREADS_PER_BLOCK];
	if (i < N) temp[threadIdx.x] = A[i] * B[i];

	__syncthreads();

	if (0 == threadIdx.x) {
		int sum = 0;

		for (int i = 0; i < blockDim.x; i++)
			sum += temp[i];
		atomicAdd(C, sum);
	}

}


void scalarProductOnHost(float *A, float *B, float *C, const int N)
{
	float product = 0;
	for (int idx = 0; idx < N; idx++)
	{
		product += (A[idx] * B[idx]);
	}
	*C = product;
}

void initialData(float *ip, int size)
{
	// generate different seed for random number
	time_t t;
	srand((unsigned)time(&t));

	for (int i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}

	return;
}

int main(int argc, char **argv)
{
	printf("%s Starting...\n", argv[0]);

	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	// set up data size of vectors
	int nElem = 1 << 10;
	printf("Vector size %d\n", nElem);

	// malloc host memory
	size_t nBytes = nElem * sizeof(float);

	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes);

	// initialize data at host side
	initialData(h_A, nElem);
	initialData(h_B, nElem);
	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	// add vector at host side for result checks
	scalarProductOnHost(h_A, h_B, hostRef, nElem);


	// malloc device global memory
	float *d_A, *d_B, *d_C;
	CHECK(cudaMalloc((float**)&d_A, nBytes));
	CHECK(cudaMalloc((float**)&d_B, nBytes));
	CHECK(cudaMalloc((float**)&d_C, nBytes));

	// transfer data from host to device
	CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

	// invoke kernel at host side
	//int iLen = 512;
	dim3 blockDim(THREADS_PER_BLOCK);
	dim3 gridDim((nElem + blockDim.x - 1) / blockDim.x);



	// --------KERNEL coalesced sum
	scalarProductOnDevice << < gridDim, blockDim >> > (d_A, d_B, d_C, nElem);
	CHECK(cudaDeviceSynchronize());

	// check kernel error
	CHECK(cudaGetLastError());

	// copy kernel result back to host side
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	// check device results
	checkResult(hostRef, gpuRef, nElem);

	// free device global memory
	CHECK(cudaFree(d_A));
	CHECK(cudaFree(d_B));
	CHECK(cudaFree(d_C));

	// free host memory
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	return(0);
}