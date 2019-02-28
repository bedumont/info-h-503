#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <iostream>
#include <string>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CHECK(call) \
do { \
	if (cudaSuccess != call) { \
		fprintf(stderr, ("CUDA ERROR! file: %s[%i] -> %s\n"), __FILE__, __LINE__, cudaGetErrorString(call)); \
		exit(0); \
	} \
} while (0)

void checkResult(float *hostRef, float *gpuRef, const int Nx, const int Ny)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	for (int i = 0; i < Nx*Ny; i++)
	{
		if (abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			match = 0;
			printf("Matrices do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
				gpuRef[i], i);
			break;
		}
	}

	if (match) printf("Matrices match.\n\n");

	return;
}

void initialData(float *ip, int Nx, int Ny)
{
	// generate different seed for random number
	time_t t;
	srand((unsigned)time(&t));

	for (int i = 0; i < Nx*Ny; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}

	return;
}

void multiplyMatricesOnHost(float *A, float *B, float *C, const int Nx, const int Ny)
{
	// TODO
	return;
}

__global__ void multiplyMatricesOnGPU(float *A, float *B, float *C, const int Nx, const int Ny)
{
	// TODO
	C[threadIdx.x] = 1.0f;
	return;
}

__global__ void multiplyMatricesOnGPUWithSharedMemory(float *A, float *B, float *C, const int Nx, const int Ny)
{
	// TODO
	C[threadIdx.x] = 1.0f;
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

	// set up data size of matrices
	int nElemx = 1 << 16;
	int nElemy = 1 << 16;
	printf("Matrix size %dx%d\n", nElemx, nElemy);

	// malloc host memory
	size_t nBytes = nElemx*nElemy * sizeof(float);

	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes);

	// initialize data at host side
	initialData(h_A, nElemx, nElemy);
	initialData(h_B, nElemx, nElemy);
	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	// Multiply matrices at host side for result checks
	multiplyMatricesOnHost(h_A, h_B, hostRef, nElemx, nElemy);


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
	int iLen = 512;
	dim3 blockDim(iLen);
	dim3 gridDim((nElemx*nElemy + blockDim.x - 1) / blockDim.x);

	// multiply kernel
	multiplyMatricesOnGPU << <gridDim, blockDim >> >(d_A, d_B, d_C, nElemx, nElemy);
	CHECK(cudaDeviceSynchronize());

	// check kernel error
	CHECK(cudaGetLastError());

	// copy kernel result back to host side
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	// check device results
	checkResult(hostRef, gpuRef, nElemx, nElemy);

	// Cleanup GPU memor
	CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

	// multiply kernel with shared memory
	multiplyMatricesOnGPUWithSharedMemory << <gridDim, blockDim >> >(d_A, d_B, d_C, nElemx, nElemy);
	CHECK(cudaDeviceSynchronize());

	// check kernel error
	CHECK(cudaGetLastError());

	// copy kernel result back to host side
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	// check device results
	checkResult(hostRef, gpuRef, nElemx, nElemy);



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

