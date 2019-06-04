#include "test.cuh"

#define LDIM 144
#define TDIM 32
#define NMB1 33
#define NMB  32

#define CHECK(call) \
do { \
	if (cudaSuccess != call) { \
		fprintf(stderr, ("CUDA ERROR! file: %s[%i] -> %s\n"), __FILE__, __LINE__, cudaGetErrorString(call)); \
		exit(0); \
	} \
} while (0)

void cuTest() {
	std::cout << " Everything Seems to work" << std::endl;
}


__global__ void directGrad(float *imR, float  *imG, float *imB, float *grad, int w, int h) {
	__shared__ float gv[LDIM+2];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int N = w * h;
	int col = i % w;
	if (i < N)
	{
		if (threadIdx.x==0 && i!=0)
		{
			gv[0]= (float)(6969 * imR[i-1] + 23434 * imG[i-1] + 2365 * imB[i-1]) / 32768;
		}
		else if (threadIdx.x+1==blockDim.x && i+1 !=N)
		{
			gv[LDIM+1]= (float)(6969 * imR[i + 1] + 23434 * imG[i + 1] + 2365 * imB[i + 1]) / 32768;
		}
		gv[threadIdx.x+1] = (float)(6969 * imR[i] + 23434 * imG[i] + 2365 * imB[i]) / 32768;

		__syncthreads();

		if (col == 0)
		{
			grad[i] = gv[threadIdx.x + 2] - gv[threadIdx.x+1];
		}
		else if (col == w - 1)
		{
			grad[i] = gv[threadIdx.x + 1] - gv[threadIdx.x];
		}
		else
		{
			grad[i] = 0.5f*(gv[threadIdx.x + 2] - gv[threadIdx.x]);
		}


	}
}



__global__ void costVol(float *im1R, float  *im1G, float *im1B, float *grad1,
						float *im2R, float  *im2G, float *im2B, float *grad2, 
						int w, int h, int d,
						float maxCol, float maxGrad, float alpha,
						float *dCost) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	int N = w * h;
	int col = i % w;
	if (i < N) {
		float costColor = maxCol; // Color L1 distance
		float costGrad = maxGrad; // x-deriv abs diff
		if (0 <= col + d && col + d < w) {
			
			costGrad = grad1[i] - grad2[i + d];
			if (costGrad < 0)
				costGrad = -costGrad;
			if (costGrad > maxGrad) // Eq. (6)
				costGrad = maxGrad;

			//
			float col1[3] = { im1R[i], im1G[i], im1B[i] };
			float col2[3] = { im2R[i + d], im2G[i + d], im2B[i + d] };
			costColor = 0;
			for (int ix = 0; ix < 3; ix++) { // Eq. (2)
				float tmp = col1[ix] - col2[ix];
				if (tmp < 0) tmp = -tmp;
				costColor += tmp;
			}
			costColor /= 3;
			if (costColor > maxCol) // Eq. (3)
				costColor = maxCol;
			
		}
		dCost[i] = (1 - alpha)*costColor + alpha*costGrad;
	}
}

__global__ void rowSum(float *in, float *out, int w, int h, int ipt, int rem) {
	extern __shared__ float temp[];
	//int i = blockIdx.x * blockDim.x + 2*threadIdx.x;
	int nt = blockDim.x;
	int tdx = threadIdx.x;
	int pl = ipt * tdx;
	int pad = pl / NMB;
	int bk = blockIdx.x;
	if (pl+ipt<=w)
	{
		float csum = 0.0f;
		for (int ix = 0; ix < ipt; ix++)
		{
			csum += in[pl + ix + bk * w];

			temp[(pl + ix+pad)] = csum;
		}
		float toSum;
		for (int step = 1; step < nt; step*=2)
		{
			
			__syncthreads();
			if (tdx >= step)
			{
				toSum = temp[((pl - ipt * (step - 1) - 1)*(NMB1))>>5];

				__syncthreads();
				for (int ix = 0; ix < ipt; ix++)
				{

					temp[pl + ix+pad] += toSum;
				}
			}
		}

		__syncthreads();

		if (tdx==0)
		{
			out[bk*w] = 0.0f;
			for (int ix = 0; ix < ipt; ix++)
			{
				out[bk*w + ix+1] = temp[ix];
			}
		}
		else if (tdx+1==nt)
		{
			for (int ix = 0; ix < ipt-1; ix++)
			{
				out[pl + bk * w + ix + 1] = temp[pl + ix + pad];
			}
		}
		else
		{
			for (int ix = 0; ix < ipt; ix++)
			{
				out[pl+bk*w + ix+1] = temp[pl+ix+pad];
			}
		}

		if (rem!=0 && tdx+1==nt)
		{
			csum = temp[pl + ipt +pad];
			for (int ix = 0; ix < rem-1; ix++)
			{
				csum += in[pl +ipt+ ix + bk * w];
				out[pl + ipt + ix + bk * w+1] = csum;
			}
		}
		
	}
}


__global__ void rowSumNoPad(float *in, float *out, int w, int h, int ipt, int rem) {
	extern __shared__ float temp[];
	//int i = blockIdx.x * blockDim.x + 2*threadIdx.x;
	int nt = blockDim.x;
	int tdx = threadIdx.x;
	int pl = ipt * tdx;
	int bk = blockIdx.x;
	if (pl + ipt <= w)
	{
		float csum = 0.0f;
		for (int ix = 0; ix < ipt; ix++)
		{
			csum += in[pl + ix + bk * w];

			temp[pl + ix] = csum;
		}
		float toSum;
		for (int step = 1; step < nt; step *= 2)
		{

			__syncthreads();
			if (tdx >= step)
			{
				toSum = temp[pl - ipt * (step - 1) - 1];

				__syncthreads();
				for (int ix = 0; ix < ipt; ix++)
				{

					temp[pl + ix] += toSum;
				}
			}
		}

		__syncthreads();

		if (tdx == 0)
		{
			out[bk*w] = 0.0f;
			for (int ix = 0; ix < ipt; ix++)
			{
				out[bk*w + ix + 1] = temp[ix];
			}
		}
		else if (tdx + 1 == nt)
		{
			for (int ix = 0; ix < ipt - 1; ix++)
			{
				out[pl + bk * w + ix + 1] = temp[pl + ix];
			}
		}
		else
		{
			for (int ix = 0; ix < ipt; ix++)
			{
				out[pl + bk * w + ix + 1] = temp[pl + ix];
			}
		}

		if (rem != 0 && tdx + 1 == nt)
		{
			csum = temp[pl + ipt];
			for (int ix = 0; ix < rem - 1; ix++)
			{
				csum += in[pl + ipt + ix + bk * w];
				out[pl + ipt + ix + bk * w + 1] = csum;
			}
		}

	}
}

__global__ void rowSumNoShared(float *in, float *out, int w, int h, int ipt, int rem) {
	//int i = blockIdx.x * blockDim.x + 2*threadIdx.x;
	int nt = blockDim.x;
	int tdx = threadIdx.x;
	int pl = ipt * tdx;
	int bk = blockIdx.x;
	int ln = bk * w;
	if (pl + ipt <= w)
	{
		float csum = 0.0f;
		if (tdx==0)
		{
			out[ln] = 0.0f;
			for (int ix = 0; ix < ipt-1; ix++)
			{
				csum += in[pl + ix + ln];
				out[(pl + ix + 1 + ln)] = csum;
			}
		}
		else 
		{
			for (int ix = 0; ix < ipt ; ix++)
			{
				csum += in[pl + ix + ln-1];
				out[pl + ln + ix] = csum;
			}
		}
		
		float toSum;
		for (int step = 1; step < nt; step *= 2)
		{

			__syncthreads();
			if (tdx >= step)
			{
				toSum = out[pl - ipt * (step - 1) - 1+ln];

				__syncthreads();
				for (int ix = 0; ix < ipt; ix++)
				{
					out[pl + ix+ln] += toSum;
				}
			}
		}
		__syncthreads();

		if (rem != 0 && tdx + 1 == nt)
		{
			csum = out[pl + ipt - 1 +ln];
			for (int ix = 0; ix < rem; ix++)
			{
				csum += in[pl + ipt -1+ ix +ln];
				out[pl + ipt + ix +ln] = csum;
			}
		}
	}
}


__global__ void scan(float *input, float *output, int n) {
	extern __shared__ float temp[];   
	int tdx = threadIdx.x;
	int pl = tdx << 1;
	int bk = blockIdx.x;
	int offset = 1;

	temp[pl] = input[pl+bk*n];
	temp[pl + 1] = input[pl + 1 +bk*n];

	for (int d = n >> 1; d > 0; d >>= 1) {
		__syncthreads();        
		if (tdx < d) 
		{ 
			int ai = offset * (pl + 1) - 1;        
			int bi = offset * (pl + 2) - 1;       
			temp[bi] += temp[ai]; 
		}     
		offset *= 2; 
	}

	if (tdx == 0) temp[n - 1] = 0;

	for (int d = 1; d < n; d *= 2) {
		offset >>= 1; 
		__syncthreads();     
		if (tdx < d) {
			int ai = offset * (pl + 1) - 1; 
			int bi = offset * (pl + 2) - 1;

			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}   
	__syncthreads();   
	output[pl+bk*n] = temp[pl];   
	output[pl + 1+bk*n] = temp[pl + 1];
}

__global__ void scanPad(float *input, float *output, int n) {
	extern __shared__ float temp[];
	int tdx = threadIdx.x;
	int pl = tdx << 1;
	int bk = blockIdx.x;
	int offset = 1;

	temp[((pl)*(NMB1))>>5] = input[pl + bk * n];
	temp[((pl + 1)*(NMB1))>>5] = input[pl + 1 + bk * n];

	for (int d = n >> 1; d > 0; d >>= 1) {
		__syncthreads();
		if (tdx < d)
		{
			int ai = ((offset * (pl + 1) - 1)*(NMB1))>>5;
			int bi = ((offset * (pl + 2) - 1)*(NMB1))>>5;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (tdx == 0) temp[((n - 1)*(NMB1)) >> 5] = 0;

	for (int d = 1; d < n; d *= 2) {
		offset >>= 1;
		__syncthreads();
		if (tdx < d) {
			int ai = ((offset * (pl + 1) - 1)*(NMB1)) >> 5;
			int bi = ((offset * (pl + 2) - 1)*(NMB1)) >> 5;

			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	output[pl + bk * n] = temp[((pl)*(NMB1)) >> 5];
	output[pl + 1 + bk * n] = temp[((pl+1)*(NMB1)) >> 5];
}

__global__ void transpose(float *input, float *output, int width, int height) {

	__shared__ float temp[TDIM][TDIM + 1];
	int xIndex = blockIdx.x*TDIM + threadIdx.x;
	int yIndex = blockIdx.y*TDIM + threadIdx.y;

	if ((xIndex < width) && (yIndex < height)) 
	{ 
		int id_in = yIndex * width + xIndex;
		temp[threadIdx.y][threadIdx.x] = input[id_in];
	}

	__syncthreads();

	xIndex = blockIdx.y * TDIM + threadIdx.x;
	yIndex = blockIdx.x * TDIM + threadIdx.y;

	if ((xIndex < height) && (yIndex < width)) 
	{ 
		int id_out = yIndex * height + xIndex;
		output[id_out] = temp[threadIdx.x][threadIdx.y]; 
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

__global__ void matSetup(float *input1,float *input2, int w, int h) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int N = w * h;
	if (i<N)
	{
		input1[i] = 1.0f;
		input2[i] = (float)(i%w);
	}
}


__global__ void boxFilter(float *intIm, float *meanIm, int w, int h, int r) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int col = i % w;
	int ln = i % h;
	int xmin = col - r;
	int xmax = col + r;
	int ymin = ln - r;
	int ymax = ln + r;
	int N = w * h;
	float A =0.0f;
	float B=0.0f;
	float C=0.0f;
	float D=0.0f;
	if (i<N)
	{
		if (xmin <=0 || ymin <=0)
		{
			if (xmin<=0)
			{
				xmin = 0;
				B = 0.0f;
			}
			else if (ymin<=0)
			{
				ymin = 0;
			}
			
			D = 0.0f;
		}
	}
}

void rSumCheck() {
	int w = 2048;
	int h = 1024;
	size_t Nbytes = w * h * sizeof(float);
	float *d_ones, *d_ref, *d_rS;
	CHECK(cudaMalloc((float**)&d_ones, Nbytes));
	CHECK(cudaMalloc((float**)&d_ref, Nbytes));
	CHECK(cudaMalloc((float**)&d_rS, Nbytes));
	float *h_ref, *h_rS;
	h_ref= (float *)malloc(Nbytes);
	h_rS=(float *)malloc(Nbytes);
	dim3 blockDim(h);
	dim3 gridDim((h*w + blockDim.x - 1) / blockDim.x);
	matSetup << < gridDim, blockDim >> > (d_ones, d_ref, w, h);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(h_ref, d_ref, Nbytes, cudaMemcpyDeviceToHost));
	int ipt = w / blockDim.x;
	int rem = w % blockDim.x;
	dim3 gd(h);
	printf("Number of Shared indexes : %d\n", ((w*(NMB1)) / (NMB)));
	rowSum << <gd, blockDim, ((w*(NMB1)) / (NMB)) * sizeof(float) >> > (d_ones, d_rS, w, h, ipt, rem);
	CHECK(cudaDeviceSynchronize());

	rowSumNoPad << <gd, blockDim, w * sizeof(float) >> > (d_ones, d_rS, w, h, ipt, rem);
	CHECK(cudaDeviceSynchronize());
	scan << <gd, blockDim, w * sizeof(float) >> > (d_ones, d_rS, w);
	CHECK(cudaDeviceSynchronize());

	scanPad << <gd, blockDim, ((w*(NMB1)) / (NMB)) * sizeof(float) >> > (d_ones, d_rS, w);
	CHECK(cudaDeviceSynchronize());
	//cudaFuncSetCacheConfig(rowSumNoShared, cudaFuncCachePreferL1);
	rowSumNoShared << <gd, blockDim >> > (d_ones, d_rS, w, h, ipt, rem);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(h_rS, d_rS, Nbytes, cudaMemcpyDeviceToHost));

	double epsilon = 1.0E-8;
	bool match = 1;

	for (int i = 0; i < w*h; i++)
	{
		if (abs(h_ref[i] - h_rS[i]) > epsilon)
		{
			match = 0;
			printf("Matrices do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n", h_ref[i],
				h_rS[i], i);
			break;
		}
	}

	if (match) printf("Matrices match.\n\n");

}


Image fcv(float * im1Color, float * im2Color, int w, int h, int dispMin, int dispMax, const ParamGuidedFilter & param)
{
	
	float *d_im1R, *d_im1G, *d_im1B, *d_im2R, *d_im2G, *d_im2B;
	size_t Nbytes = w * h * sizeof(float);
	CHECK(cudaMalloc((float**)&d_im1R, Nbytes));
	CHECK(cudaMalloc((float**)&d_im1G, Nbytes));
	CHECK(cudaMalloc((float**)&d_im1B, Nbytes));
	CHECK(cudaMalloc((float**)&d_im2R, Nbytes));
	CHECK(cudaMalloc((float**)&d_im2G, Nbytes));
	CHECK(cudaMalloc((float**)&d_im2B, Nbytes));

	CHECK(cudaMemcpy(d_im1R, im1Color, Nbytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_im1G, im1Color+w*h, Nbytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_im1B, im1Color+2*w*h, Nbytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_im2R, im2Color, Nbytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_im2G, im2Color + w * h, Nbytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_im2B, im2Color + 2 * w*h, Nbytes, cudaMemcpyHostToDevice));

	float *d_disp, *h_disp, *h_cost, *d_cost;
	CHECK(cudaMalloc((float**)&d_cost, Nbytes));
	//h_disp=(float *)malloc(Nbytes);
	//h_cost= (float *)malloc(Nbytes);


	//for (size_t i = 0; i < w*h; i++)
	//{
	//	h_disp[i] = static_cast<float>(dispMin - 1);
	//	h_cost[i] = std::numeric_limits<float>::max();
	//}

	rSumCheck();
	float  *h_gray, *d_gradient1, *d_gradient2;
	h_gray = (float *)malloc(Nbytes);

	CHECK(cudaMalloc((float**)&d_gradient1, Nbytes));
	CHECK(cudaMalloc((float**)&d_gradient2, Nbytes));

	dim3 blockDim(LDIM);
	dim3 gridDim((h*w + blockDim.x - 1) / blockDim.x);
	
	directGrad << <gridDim, blockDim >> > (d_im1R, d_im1G, d_im1B, d_gradient1, w, h);
	CHECK(cudaDeviceSynchronize());

	// check kernel error
	//CHECK(cudaGetLastError());
	directGrad << <gridDim, blockDim >> > (d_im2R, d_im2G, d_im2B, d_gradient2, w, h);
	CHECK(cudaDeviceSynchronize());

	// check kernel error
	//CHECK(cudaGetLastError());

	costVol << <gridDim, blockDim >> > (d_im1R, d_im1G, d_im1B, d_gradient1, d_im2R, d_im2G, d_im2B, d_gradient2, w, h, dispMin, param.color_threshold, param.gradient_threshold, param.alpha, d_cost);
	CHECK(cudaDeviceSynchronize());

	// check kernel error
	//CHECK(cudaGetLastError());

	CHECK(cudaMemcpy(h_gray, d_cost, Nbytes, cudaMemcpyDeviceToHost));

	CHECK(cudaFree(d_im1R));
	CHECK(cudaFree(d_im1G));
	CHECK(cudaFree(d_im1B));
	CHECK(cudaFree(d_im2R));
	CHECK(cudaFree(d_im2G));
	CHECK(cudaFree(d_im2B));
	CHECK(cudaFree(d_gradient1));
	CHECK(cudaFree(d_gradient2));
	CHECK(cudaFree(d_cost));
	return Image(h_gray,w,h);
}

