
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<opencv2/opencv.hpp>

double cubicInterpolate(double p[4], double x) {
	return p[1] + 0.5 * x*(p[2] - p[0] + x * (2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + x * (3.0*(p[1] - p[2]) + p[3] - p[0])));
}

double bicubicInterpolate(double p[4][4], double x, double y) {
	double arr[4];
	arr[0] = cubicInterpolate(p[0], y);
	arr[1] = cubicInterpolate(p[1], y);
	arr[2] = cubicInterpolate(p[2], y);
	arr[3] = cubicInterpolate(p[3], y);
	return cubicInterpolate(arr, x);
}

template <typename T>
__global__ void resize_nearest_kernel(T* pIn, T* pOut, int widthIn, int heightIn, int widthOut, int heightOut, const int num_channels)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < heightOut && j < widthOut)
	{
		int iIn =  int(i * (float)heightIn / heightOut);
		int jIn = int(j * (float)widthIn / widthOut);
		for (int k = 0; k < num_channels; k++) {
			pOut[(i*widthOut + j) * num_channels + k] = pIn[(iIn*widthIn + jIn) * num_channels + k];
		}
	}
}

// bilinear interpolate
__global__ void resize_linear_kernel(const int n, const float*src, int srcWidth, int srcHeight, \
	float *dst, int dstWidth, int dstHeight) {

	float srcColTidf;
	float srcRowTidf;
	float c, r;
	const float rowScale = srcHeight / (float)(dstHeight);
	const float colScale = srcWidth / (float)(dstWidth);
	//int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
		i < (n); \
		i += blockDim.x * gridDim.x) {
		int tidC = i;
		int tidR = i;// *colScaleExtend;
		float srcColTidf = (float)((tidC % (dstWidth)) * colScale);
		float srcRowTidf = (float)((tidR / (dstWidth)) * rowScale);
		int srcColTid = (int)srcColTidf;
		int srcRowTid = (int)srcRowTidf;
		c = srcColTidf - srcColTid;
		r = srcRowTidf - srcRowTid;

		int dstInd = i;
		int srcInd = srcRowTid * srcWidth + srcColTid;
		dst[dstInd] = 0;
		dst[dstInd] += (1 - c)*(1 - r)*src[srcRowTid * srcWidth + srcColTid];
		dst[dstInd] += (1 - c)*r*src[(srcRowTid + 1)*srcWidth + srcColTid];
		dst[dstInd] += c * (1 - r)*src[srcRowTid*srcWidth + srcColTid + 1];
		dst[dstInd] += c * r*src[(srcRowTid + 1)*srcWidth + srcColTid + 1];
	}
}

// cubic interploation
__global__ void resize_cubic_kernel(const int n, const float*src, int srcWidth, int srcHeight, \
	float *dst, int dstWidth, int dstHeight) {

	float srcColTidf;
	float srcRowTidf;
	float c, r;
	float A = -0.75;
	const float rowScale = srcHeight / (float)(dstHeight);
	const float colScale = srcWidth / (float)(dstWidth);
	//int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
		i < (n); \
		i += blockDim.x * gridDim.x) {
		int tidC = i;
		int tidR = i;// *colScaleExtend;
		float srcColTidf = (float)((tidC % (dstWidth)) * colScale);
		float srcRowTidf = (float)((tidR / (dstWidth)) * rowScale);
		int srcColTid = (int)srcColTidf;
		int srcRowTid = (int)srcRowTidf;
		c = srcColTidf - srcColTid;
		r = srcRowTidf - srcRowTid;

		int dstInd = i;
		int srcInd = srcRowTid * srcWidth + srcColTid;
		dst[dstInd] = 0;

		{
			//
			float coeffsY[4];
			coeffsY[0] = ((A*(r + 1) - 5 * A)*(r + 1) + 8 * A)*(r + 1) - 4 * A;
			coeffsY[1] = ((A + 2)*r - (A + 3))*r*r + 1;
			coeffsY[2] = ((A + 2)*(1 - r) - (A + 3))*(1 - r)*(1 - r) + 1;
			coeffsY[3] = 1.f - coeffsY[0] - coeffsY[1] - coeffsY[2];

			float coeffsX[4];
			coeffsX[0] = ((A*(c + 1) - 5 * A)*(c + 1) + 8 * A)*(c + 1) - 4 * A;
			coeffsX[1] = ((A + 2)*c - (A + 3))*c*c + 1;
			coeffsX[2] = ((A + 2)*(1 - c) - (A + 3))*(1 - c)*(1 - c) + 1;
			coeffsX[3] = 1.f - coeffsX[0] - coeffsX[1] - coeffsX[2];

			dst[dstInd] =
				src[(srcRowTid - 1) * srcWidth + (srcColTid - 1)] * coeffsX[0] * coeffsY[0] \
				+ src[(srcRowTid)* srcWidth + (srcColTid - 1)] * coeffsX[0] * coeffsY[1] \
				+ src[(srcRowTid + 1) * srcWidth + (srcColTid - 1)] * coeffsX[0] * coeffsY[2] \
				+ src[(srcRowTid + 2) * srcWidth + (srcColTid - 1)] * coeffsX[0] * coeffsY[3] \
				+ src[(srcRowTid - 1) * srcWidth + (srcColTid)] * coeffsX[1] * coeffsY[0] \
				+ src[(srcRowTid)* srcWidth + (srcColTid)] * coeffsX[1] * coeffsY[1] \
				+ src[(srcRowTid + 1) * srcWidth + (srcColTid)] * coeffsX[1] * coeffsY[2] \
				+ src[(srcRowTid + 2) * srcWidth + (srcColTid)] * coeffsX[1] * coeffsY[3] \
				+ src[(srcRowTid - 1) * srcWidth + (srcColTid + 1)] * coeffsX[2] * coeffsY[0] \
				+ src[(srcRowTid)* srcWidth + (srcColTid + 1)] * coeffsX[2] * coeffsY[1] \
				+ src[(srcRowTid + 1) * srcWidth + (srcColTid + 1)] * coeffsX[2] * coeffsY[2] \
				+ src[(srcRowTid + 2) * srcWidth + (srcColTid + 1)] * coeffsX[2] * coeffsY[3] \
				+ src[(srcRowTid - 1) * srcWidth + (srcColTid + 2)] * coeffsX[3] * coeffsY[0] \
				+ src[(srcRowTid)* srcWidth + (srcColTid + 2)] * coeffsX[3] * coeffsY[1] \
				+ src[(srcRowTid + 1) * srcWidth + (srcColTid + 2)] * coeffsX[3] * coeffsY[2] \
				+ src[(srcRowTid + 2) * srcWidth + (srcColTid + 2)] * coeffsX[3] * coeffsY[3];
			if (dstInd == 11) {
				printf("%f \n", dst[dstInd]);
			}
		}
	}
}

template <typename T>
void resizeGPU(T* pIn_d, T* pOut_d, int widthIn, int heightIn, int widthOut, int heightOut,
	const int num_channels)
{

	//dim3 block(16, 16);
	//dim3 grid((widthOut + 15) / 16, (heightOut + 15) / 16);
	//resize_nearest_kernel << < grid, block >> > (pIn_d, pOut_d, widthIn, heightIn, widthOut, heightOut, num_channels);
	int n = widthOut * heightOut * num_channels;
	resize_cubic_kernel << <(n + 255)/256, 256 >> > (n, pIn_d, widthIn, heightIn, pOut_d, widthOut, heightOut);
}

int main()
{
	cv::Mat srcImage = cv::imread("C:\\Users\\Administrator\\Pictures\\2007_000799.jpg");
	srcImage.convertTo(srcImage, CV_32F, 1.0 / 225);
	cv::cvtColor(srcImage, srcImage, cv::COLOR_BGR2GRAY);
	const int num_channels = srcImage.channels();
	cv::Mat dstImage = cv::Mat::zeros(cv::Size(800, 800), srcImage.type());
	size_t srcLen = srcImage.rows * srcImage.cols * num_channels * sizeof(float);
	size_t dstLen = dstImage.rows * dstImage.cols * num_channels * sizeof(float);
	float *srcPtr;
	float *dstPtr;
	cudaMalloc(&srcPtr, srcLen);
	cudaMalloc(&dstPtr, dstLen);

	cudaMemcpy(srcPtr, srcImage.data, srcLen, cudaMemcpyHostToDevice);
	//cudaMemcpy(dstPtr, dstImage.data, dstLen, cudaMemcpyHostToDevice);

	resizeGPU(srcPtr, dstPtr, srcImage.cols, srcImage.rows, dstImage.cols, dstImage.rows, num_channels);
	cudaDeviceSynchronize();

	//cudaMemcpy(srcImage.data, srcPtr, srcLen, cudaMemcpyDeviceToHost);
	cudaMemcpy(dstImage.data, dstPtr, dstLen, cudaMemcpyDeviceToHost);
	
	cudaFree(srcPtr);
	cudaFree(dstPtr);

	cv::imshow("Src image", srcImage);
	cv::imshow("Dst image", dstImage);
	cv::waitKey(0);
	cv::destroyAllWindows();
}
