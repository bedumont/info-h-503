
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <iostream>
#include <string>

// From imagePPM
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <cassert>
#include <exception>
#include <vector>


/*
* Various memory access pattern optimizations applied to a matrix transpose
* kernel.
*/

#define BDIMX 16
#define BDIMY 16

#define IPAD 2


// From imagePPM
class Image
{
public:

	struct Rgb
	{
		Rgb() : r(0), g(0), b(0) {}
		Rgb(float c) : r(c), g(c), b(c) {}
		Rgb(float _r, float _g, float _b) : r(_r), g(_g), b(_b) { }
		bool operator != (const Rgb &c) const { return c.r != r && c.g != g && c.b != b; }
		Rgb& operator *= (const Rgb &rgb) { r *= rgb.r, g *= rgb.g, b *= rgb.b; return *this; }
		Rgb& operator += (const Rgb &rgb) { r += rgb.r, g += rgb.g, b += rgb.b; return *this; }
		friend float& operator += (float &f, const Rgb rgb)
		{
			f += (rgb.r + rgb.g + rgb.b) / 3.f; return f;
		}
		float r, g, b;
	};

	Image() : w(0), h(0), pixels(NULL)
	{ // empty image 
	}
	Image(const unsigned int &_w, const unsigned int &_h, const Rgb &c = kBlack) : w(_w), h(_h), pixels(NULL)
	{
		pixels = new Rgb[w * h];
		for (int i = 0; i < w * h; ++i) pixels[i] = c;
	}
	Image(const Image &img) : w(img.w), h(img.h), pixels(NULL)
	{
		pixels = new Rgb[w * h];
		memcpy(pixels, img.pixels, sizeof(Rgb) * w * h);
	}
	// move constructor
	Image(Image &&img) : w(0), h(0), pixels(NULL)
	{
		w = img.w;
		h = img.h;
		pixels = img.pixels;
		img.pixels = NULL;
		img.w = img.h = 0;
	}
	// move assignment operator
	Image& operator = (Image &&img)
	{
		//printf("in move assignment\n");
		if (this != &img) {
			if (pixels != NULL) delete[] pixels;
			w = img.w, h = img.h;
			pixels = img.pixels;
			img.pixels = NULL;
			img.w = img.h = 0;
		}
		return *this;
	}
	Rgb& operator () (const unsigned &x, const unsigned int &y) const
	{
		assert(x < w && y < h);
		return pixels[y * w + x];
	}
	Image& operator *= (const Rgb &rgb)
	{
		for (int i = 0; i < w * h; ++i) pixels[i] *= rgb;
		return *this;
	}
	Image& operator += (const Image &img)
	{
		for (int i = 0; i < w * h; ++i) pixels[i] += img[i];
		return *this;
	}
	Image& operator /= (const float &div)
	{
		float invDiv = 1 / div;
		for (int i = 0; i < w * h; ++i) pixels[i] *= invDiv;
		return *this;
	}
	friend Image operator * (const Rgb &rgb, const Image &img)
	{
		Image tmp(img);
		tmp *= rgb;
		return tmp;
	}
	Image operator * (const Image &img)
	{
		Image tmp(*this);
		// multiply pixels together
		for (int i = 0; i < w * h; ++i) tmp[i] *= img[i];
		return tmp;
	}
	static Image circshift(const Image &img, const std::pair<int, int> &shift)
	{
		Image tmp(img.w, img.h);
		int w = img.w, h = img.h;
		for (int j = 0; j < h; ++j) {
			int jmod = (j + shift.second) % h;
			for (int i = 0; i < w; ++i) {
				int imod = (i + shift.first) % w;
				tmp[jmod * w + imod] = img[j * w + i];
			}
		}
		return tmp;
	}
	const Rgb& operator [] (const unsigned int &i) const { return pixels[i]; }
	Rgb& operator [] (const unsigned int &i) { return pixels[i]; }
	~Image() { if (pixels != NULL) delete[] pixels; }
	unsigned int w, h;
	Rgb *pixels;
	static const Rgb kBlack, kWhite, kRed, kGreen, kBlue;

};

const Image::Rgb Image::kBlack = Image::Rgb(0);
const Image::Rgb Image::kWhite = Image::Rgb(1);
const Image::Rgb Image::kRed = Image::Rgb(1, 0, 0);
const Image::Rgb Image::kGreen = Image::Rgb(0, 1, 0);
const Image::Rgb Image::kBlue = Image::Rgb(0, 0, 1);


void savePPM(const Image &img, const char *filename)
{
	if (img.w == 0 || img.h == 0) { fprintf(stderr, "Can't save an empty image\n"); return; }
	std::ofstream ofs;
	try {
		ofs.open(filename, std::ios::binary); // need to spec. binary mode for Windows users
		if (ofs.fail()) throw("Can't open output file");
		ofs << "P6\n" << img.w << " " << img.h << "\n255\n";
		unsigned char r, g, b;
		// loop over each pixel in the image, clamp and convert to byte format
		for (int i = 0; i < img.w * img.h; ++i) {
			r = static_cast<unsigned char>(std::min(1.f, img.pixels[i].r) * 255);
			g = static_cast<unsigned char>(std::min(1.f, img.pixels[i].g) * 255);
			b = static_cast<unsigned char>(std::min(1.f, img.pixels[i].b) * 255);
			ofs << r << g << b;
		}
		ofs.close();
	}
	catch (const char *err) {
		fprintf(stderr, "%s\n", err);
		ofs.close();
	}
}


Image readPPM(const char *filename)
{
	std::ifstream ifs;
	ifs.open(filename, std::ios::binary); // need to spec. binary mode for Windows users
	Image img;
	try {
		if (ifs.fail()) { throw("Can't open input file"); }
		//std::string header;
		char header[3];
		ifs >> header;
		std::string headerStr = std::string(header);
		if (strcmp(headerStr.c_str(), "P6") != 0) throw("Can't read input file");
		int w, h, b;
		ifs >> w >> h >> b;
		img.w = w; img.h = h;
		img.pixels = new Image::Rgb[w * h]; // this is throw an exception if bad_alloc
		ifs.ignore(256, '\n'); // skip empty lines is necessary until we get to the binary data
		unsigned char pix[3];
		// read each pixel one by one and convert bytes to floats
		for (int i = 0; i < w * h; ++i) {
			ifs.read(reinterpret_cast<char *>(pix), 3);
			img.pixels[i].r = pix[0] / 255.f; if (img.pixels[i].r>0.7) img.pixels[i].r *= 3;
			img.pixels[i].g = pix[1] / 255.f; if (img.pixels[i].g>0.7) img.pixels[i].g *= 3;
			img.pixels[i].b = pix[2] / 255.f; if (img.pixels[i].b>0.7) img.pixels[i].b *= 3;
		}
		ifs.close();
	}
	catch (const char *err) {
		fprintf(stderr, "%s\n", err);
		ifs.close();
	}

	return img;
}

/*********************** Slightly modified readPPM to have directly a 1D buffer output **********************/

unsigned char* readPPM_to_Buffer(const char *filename, long& SIZE)
{
	std::ifstream ifs;
	ifs.open(filename, std::ios::binary); // need to spec. binary mode for Windows users
										  //Image img;
	if (ifs.fail()) {
		printf("Can't open input file \n");
		exit(1);
	}
	//std::string header;
	char header[3];
	ifs >> header;
	std::string headerStr = std::string(header);
	if (strcmp(headerStr.c_str(), "P6") != 0) throw("Can't read input file");
	int w, h, b;
	ifs >> w >> h >> b;
	//img.w = w; img.h = h;
	SIZE = w * h; // grey-scale only: use (w*h*3) for RGB images
	unsigned char *data = (unsigned char*)malloc(SIZE);
	//img.pixels = new Image::Rgb[w * h]; // this is throw an exception if bad_alloc
	ifs.ignore(256, '\n'); // skip empty lines is necessary until we get to the binary data
	unsigned char pix[3];
	unsigned char greyPix;
	// read each pixel one by one and convert bytes to floats
	for (int i = 0; i < w * h; ++i) {
		ifs.read(reinterpret_cast<char *>(pix), 3);
		greyPix = (pix[0] + pix[1] + pix[2]) / 3;
		data[i] = greyPix;
		/*
		img.pixels[i].r = pix[0] / 255.f; if (img.pixels[i].r>0.7) img.pixels[i].r *= 3;
		img.pixels[i].g = pix[1] / 255.f; if (img.pixels[i].g>0.7) img.pixels[i].g *= 3;
		img.pixels[i].b = pix[2] / 255.f; if (img.pixels[i].b>0.7) img.pixels[i].b *= 3;
		*/
	}
	ifs.close();

	return data;
}


void initialData(float *in, const int size)
{
	for (int i = 0; i < size; i++)
	{
		in[i] = (float)(rand() & 0xFF) / 10.0f; //100.0f;
	}

	return;
}

void printData(float *in, const int size)
{
	for (int i = 0; i < size; i++)
	{
		printf("%dth element: %f\n", i, in[i]);
	}

	return;
}

#define CHECK(call) \
do { \
	if (cudaSuccess != call) { \
		fprintf(stderr, ("CUDA ERROR! file: %s[%i] -> %s\n"), __FILE__, __LINE__, cudaGetErrorString(call)); \
		exit(0); \
	} \
} while (0)

void checkResult(float *hostRef, float *gpuRef, const int size, int showme)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	for (int i = 0; i < size; i++)
	{
		if (abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			match = 0;
			printf("different on %dth element: host %f gpu %f\n", i, hostRef[i],
				gpuRef[i]);
			break;
		}

		if (showme && i > size / 2 && i < size / 2 + 5)
		{
			// printf("%dth element: host %f gpu %f\n",i,hostRef[i],gpuRef[i]);
		}
	}

	if (!match)  printf("Arrays do not match.\n\n");
}

void transposeHost(float *out, float *in, const int nx, const int ny)
{
	for (int iy = 0; iy < ny; ++iy)
	{
		for (int ix = 0; ix < nx; ++ix)
		{
			out[ix * ny + iy] = in[iy * nx + ix];
		}
	}
}

__global__ void warmup(float *out, float *in, const int nx, const int ny)
{
	unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		out[iy * nx + ix] = in[iy * nx + ix];
	}
}

// case 0 copy kernel: access data in rows
__global__ void copyRow(float *out, float *in, const int nx, const int ny)
{
	unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		out[iy * nx + ix] = in[iy * nx + ix];
	}
}

// case 1 copy kernel: access data in columns
__global__ void copyCol(float *out, float *in, const int nx, const int ny)
{
	unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		out[ix * ny + iy] = in[ix * ny + iy];
	}
}

// case 2 transpose kernel: read in rows and write in columns
__global__ void transposeNaiveRow(float *out, float *in, const int nx,
	const int ny)
{
	unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		out[ix * ny + iy] = in[iy * nx + ix];
	}
}

// case 3 transpose kernel: read in columns and write in rows
__global__ void transposeNaiveCol(float *out, float *in, const int nx,
	const int ny)
{
	unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		out[iy * nx + ix] = in[ix * ny + iy];
	}
}

// case 4 transpose kernel: read in rows and write in columns + unroll 4 blocks
__global__ void transposeUnroll4Row(float *out, float *in, const int nx,
	const int ny)
{
	unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int ti = iy * nx + ix; // access in rows
	unsigned int to = ix * ny + iy; // access in columns

	if (ix + 3 * blockDim.x < nx && iy < ny)
	{
		out[to] = in[ti];
		out[to + ny * blockDim.x] = in[ti + blockDim.x];
		out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
		out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
	}
}

__global__ void transposeRow(float *out, float *in, const int nx,
	const int ny)
{
	unsigned int thd_ix = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int thd_iy = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int thd_1D = thd_iy * gridDim.x * blockDim.x + thd_ix;

	if (thd_1D < ny)
	{
		printf("ix, iy, row = %d, %d, %d \n", thd_ix, thd_iy, thd_1D);
		int row_start = thd_1D;     //thd_1D * nx;
		int row_end = thd_1D + nx;  //(thd_1D + 1) * nx;
		int col_index = thd_1D;
		for (int i = row_start; i < row_end; i++) {
			out[col_index] = in[i];
			col_index += nx;
		}
	}
}

__global__ void transposeUnroll8Row(float *out, float *in, const int nx,
	const int ny)
{
	unsigned int ix = blockDim.x * blockIdx.x * 8 + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int ti = iy * nx + ix; // access in rows
	unsigned int to = ix * ny + iy; // access in columns

	if (ix + 7 * blockDim.x < nx && iy < ny)
	{
		out[to] = in[ti];
		out[to + ny * blockDim.x] = in[ti + blockDim.x];
		out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
		out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
		out[to + ny * 4 * blockDim.x] = in[ti + 4 * blockDim.x];
		out[to + ny * 5 * blockDim.x] = in[ti + 5 * blockDim.x];
		out[to + ny * 6 * blockDim.x] = in[ti + 6 * blockDim.x];
		out[to + ny * 7 * blockDim.x] = in[ti + 7 * blockDim.x];
	}
}

// case 5 transpose kernel: read in columns and write in rows + unroll 4 blocks
__global__ void transposeUnroll4Col(float *out, float *in, const int nx,
	const int ny)
{
	unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int ti = iy * nx + ix; // access in rows
	unsigned int to = ix * ny + iy; // access in columns

	if (ix + 3 * blockDim.x < nx && iy < ny)
	{
		out[ti] = in[to];
		out[ti + blockDim.x] = in[to + blockDim.x * ny];
		out[ti + 2 * blockDim.x] = in[to + 2 * blockDim.x * ny];
		out[ti + 3 * blockDim.x] = in[to + 3 * blockDim.x * ny];
	}
}

/*
* case 6 :  transpose kernel: read in rows and write in colunms + diagonal
* coordinate transform
*/
__global__ void transposeDiagonalRow(float *out, float *in, const int nx,
	const int ny)
{
	unsigned int blk_y = blockIdx.x;
	unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

	unsigned int ix = blockDim.x * blk_x + threadIdx.x;
	unsigned int iy = blockDim.y * blk_y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		out[ix * ny + iy] = in[iy * nx + ix];
	}
}

/*
* case 7 :  transpose kernel: read in columns and write in row + diagonal
* coordinate transform.
*/
__global__ void transposeDiagonalCol(float *out, float *in, const int nx,
	const int ny)
{
	unsigned int blk_y = blockIdx.x;
	unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

	unsigned int ix = blockDim.x * blk_x + threadIdx.x;
	unsigned int iy = blockDim.y * blk_y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		out[iy * nx + ix] = in[ix * ny + iy];
	}
}

__global__ void transposeDiagonalColUnroll4(float *out, float *in, const int nx,
	const int ny)
{
	unsigned int blk_y = blockIdx.x;
	unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

	unsigned int ix_stride = blockDim.x * blk_x;
	unsigned int ix = ix_stride * 4 + threadIdx.x;
	unsigned int iy = blockDim.y * blk_y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		out[iy * nx + ix] = in[ix * ny + iy];
		out[iy * nx + ix + blockDim.x] = in[(ix + blockDim.x) * ny + iy];
		out[iy * nx + ix + 2 * blockDim.x] =
			in[(ix + 2 * blockDim.x) * ny + iy];
		out[iy * nx + ix + 3 * blockDim.x] =
			in[(ix + 3 * blockDim.x) * ny + iy];
	}
}

__global__ void transposeSmem(float *out, float *in, int nx, int ny)
{
	// static shared memory
	__shared__ float tile[BDIMY][BDIMX];

	// coordinate in original matrix
	unsigned int ix, iy, ti, to;
	ix = blockDim.x * blockIdx.x + threadIdx.x;
	iy = blockDim.y * blockIdx.y + threadIdx.y;

	// linear global memory index for original matrix
	ti = iy * nx + ix;

	// thread index in transposed block
	unsigned int bidx, irow, icol;
	bidx = threadIdx.y * blockDim.x + threadIdx.x;
	irow = bidx / blockDim.y;
	icol = bidx % blockDim.y;

	// coordinate in transposed matrix
	ix = blockDim.y * blockIdx.y + icol;
	iy = blockDim.x * blockIdx.x + irow;

	// linear global memory index for transposed matrix
	to = iy * ny + ix;

	// transpose with boundary test
	if (ix < nx && iy < ny)
	{
		// load data from global memory to shared memory
		tile[threadIdx.y][threadIdx.x] = in[ti];

		// thread synchronization
		__syncthreads();

		// store data to global memory from shared memory
		out[to] = tile[icol][irow];
	}
}

__global__ void transposeSmemPad(float *out, float *in, int nx, int ny)
{
	// static shared memory with padding
	__shared__ float tile[BDIMY][BDIMX + IPAD];

	// coordinate in original matrix
	unsigned int  ix, iy, ti, to;
	ix = blockDim.x * blockIdx.x + threadIdx.x;
	iy = blockDim.y * blockIdx.y + threadIdx.y;

	// linear global memory index for original matrix
	ti = iy * nx + ix;

	// thread index in transposed block
	unsigned int bidx, irow, icol;
	bidx = threadIdx.y * blockDim.x + threadIdx.x;
	irow = bidx / blockDim.y;
	icol = bidx % blockDim.y;

	// coordinate in transposed matrix
	ix = blockDim.y * blockIdx.y + icol;
	iy = blockDim.x * blockIdx.x + irow;

	// linear global memory index for transposed matrix
	to = iy * ny + ix;

	// transpose with boundary test
	if (ix < nx && iy < ny)
	{
		// load data from global memory to shared memory
		tile[threadIdx.y][threadIdx.x] = in[ti];

		// thread synchronization
		__syncthreads();

		// store data to global memory from shared memory
		out[to] = tile[icol][irow];
	}
}

// main functions
int main(int argc, char **argv)
{
	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s starting transpose at ", argv[0]);
	printf("device %d: %s ", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	// set up array size 2048
	int nx = 1 << 11;
	int ny = 1 << 11;

	// select a kernel and block size
	int blockx = 16;
	int blocky = 16;

	if (argc > 2) blockx = atoi(argv[2]);

	if (argc > 3) blocky = atoi(argv[3]);

	if (argc > 4) nx = atoi(argv[4]);

	if (argc > 5) ny = atoi(argv[5]);

	printf(" with matrix nx %d ny %d\n", nx, ny);
	size_t nBytes = nx * ny * sizeof(float);

	// execution configuration
	dim3 block(blockx, blocky);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	dim3 grid4((nx + block.x - 1) / (block.x * 4), (ny + block.y - 1) /
		(block.y * 4));
	dim3 grid8((nx + block.x - 1) / (block.x * 8), (ny + block.y - 1) /
		(block.y * 8));

	// allocate host memory
	float *h_A = (float *)malloc(nBytes);
	float *hostRef = (float *)malloc(nBytes);
	float *gpuRef = (float *)malloc(nBytes);

	// initialize host array
	initialData(h_A, nx * ny);

	// transpose at host side
	transposeHost(hostRef, h_A, nx, ny);

	// allocate device memory
	float *d_A, *d_C;
	CHECK(cudaMalloc((float**)&d_A, nBytes));
	CHECK(cudaMalloc((float**)&d_C, nBytes));

	// copy data from host to device
	CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

	// warmup to avoide startup overhead
	warmup << <grid, block >> >(d_C, d_A, nx, ny);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());

	// transposeNaiveRow
	transposeNaiveRow << <grid, block >> >(d_C, d_A, nx, ny);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	checkResult(hostRef, gpuRef, nx * ny, 1);

	// transposeNaiveCol
	transposeNaiveCol << <grid, block >> >(d_C, d_A, nx, ny);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	checkResult(hostRef, gpuRef, nx * ny, 1);

	// transposeUnroll4Row
	transposeUnroll4Row << <grid4, block >> >(d_C, d_A, nx, ny);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	checkResult(hostRef, gpuRef, nx * ny, 1);

	/*
	// transposeRow
	transposeRow << <grid, block >> >(d_C, d_A, nx, ny);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	checkResult(hostRef, gpuRef, nx * ny, 1);
	*/

	// transposeUnroll8Row
	transposeUnroll8Row << <grid8, block >> >(d_C, d_A, nx, ny);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	checkResult(hostRef, gpuRef, nx * ny, 1);

	// transpose Shared Memory
	transposeSmem << <grid, block >> >(d_C, d_A, nx, ny);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	checkResult(hostRef, gpuRef, nx * ny, 1);

	// transpose Shared Memory with padding
	transposeSmemPad << <grid, block >> >(d_C, d_A, nx, ny);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	checkResult(hostRef, gpuRef, nx * ny, 1);

	// transposeUnroll4Col
	transposeUnroll4Col << <grid4, block >> >(d_C, d_A, nx, ny);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	checkResult(hostRef, gpuRef, nx * ny, 1);

	// transposeDiagonalRow
	transposeDiagonalRow << <grid, block >> >(d_C, d_A, nx, ny);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	checkResult(hostRef, gpuRef, nx * ny, 1);

	// transposeDiagonalCol
	transposeDiagonalCol << <grid, block >> >(d_C, d_A, nx, ny);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	checkResult(hostRef, gpuRef, nx * ny, 1);

	// transposeDiagonalColUnroll4
	transposeDiagonalColUnroll4 << <grid4, block >> >(d_C, d_A, nx, ny);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	checkResult(hostRef, gpuRef, nx * ny, 1);


	// free host and device memory
	CHECK(cudaFree(d_A));
	CHECK(cudaFree(d_C));
	free(h_A);
	free(hostRef);
	free(gpuRef);

	// reset device
	CHECK(cudaDeviceReset());
	return EXIT_SUCCESS;
}

