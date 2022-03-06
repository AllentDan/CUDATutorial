
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<opencv2/opencv.hpp>
#include<torch/torch.h>
#include<torch/script.h>

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

// Based on
// https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
template <typename scalar_t>
__device__ __forceinline__ static scalar_t cubic_convolution1(scalar_t x, scalar_t A) {
	return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template <typename scalar_t>
__device__ __forceinline__ static scalar_t cubic_convolution2(scalar_t x, scalar_t A) {
	return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template <typename scalar_t>
__device__ __forceinline__ static void get_cubic_upsample_coefficients(
	scalar_t coeffs[4],
	scalar_t t) {
	scalar_t A = -0.75;

	scalar_t x1 = t;
	coeffs[0] = cubic_convolution2<scalar_t>(x1 + 1.0, A);
	coeffs[1] = cubic_convolution1<scalar_t>(x1, A);

	// opposite coefficients
	scalar_t x2 = 1.0 - t;
	coeffs[2] = cubic_convolution1<scalar_t>(x2, A);
	coeffs[3] = cubic_convolution2<scalar_t>(x2 + 1.0, A);
}

template <typename scalar_t>
__device__ __forceinline__ static scalar_t cubic_interp1d(
	scalar_t x0,
	scalar_t x1,
	scalar_t x2,
	scalar_t x3,
	scalar_t t) {
	scalar_t coeffs[4];
	get_cubic_upsample_coefficients<scalar_t>(coeffs, t);

	return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

/* Used by UpSampleBicubic2d.cu */
template <typename scalar_t>
__device__ __forceinline__ static scalar_t upsample_get_value_bounded(
	const scalar_t *data,
	int batch,
	int channel,
	int batchsize,
	int channels,
	int height,
	int width,
	int y,
	int x) {
	int access_y = max(min(y, height - 1), 0);
	int access_x = max(min(x, width - 1), 0);
	return data[batch*channels *height*width +channel*height*width + access_y*width + access_x];
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t area_pixel_compute_source_index(
	scalar_t scale,
	int64_t dst_index,
	bool align_corners,
	bool cubic) {
	if (align_corners) {
		return scale * dst_index;
	}
	else {
		scalar_t src_idx = scale * (dst_index + 0.5) - 0.5;
		// [Note] Follow Opencv resize logic:
		// We allow negative src_idx here and later will use
		//   dx = src_idx - floorf(src_idx)
		// to compute the "distance"(which affects weights).
		// For linear modes, weight distribution doesn't matter
		// for negative indices as they use 2 pixels to interpolate.
		// For example, [-1, 0], they both use pixel 0 value so it
		// doesn't affect if we bound the src_idx to 0 or not.
		// TODO: Our current linear mode impls use unbound indices
		// where we should and then remove this cubic flag.
		// This matters in cubic mode, as we might need [-1, 0, 1, 2]
		// to interpolate and the weights can be affected.
		return (!cubic && src_idx < 0) ? scalar_t(0) : src_idx;
	}
}

// cubic interploation pytorch
__global__ void resize_cubic_kernel_torch(const int num_elements, const float*src, int srcWidth, int srcHeight, \
	float *dst, int dstWidth, int dstHeight, bool align_corners = true) {
	float height_scale = float(srcHeight) / dstHeight;
	float width_scale = float(srcWidth) / dstWidth;
	if (align_corners && dstWidth>1) {
		height_scale = (float)(srcHeight - 1) / (dstHeight - 1);
		width_scale = (float)(srcWidth - 1) / (dstWidth - 1);
		
	}
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= num_elements) {
		return;
	}
	// Special case: input and output are the same size, just copy
	const int output_x = index % dstWidth;
	const int output_y = index / dstWidth;

	const size_t batchsize = 1;
	const size_t channels = 3;
	if (srcHeight == dstHeight && srcWidth == dstWidth) {
		for (int n = 0; n < batchsize; n++) {
			for (int c = 0; c < channels; c++) {
				const float val = src[n* channels *dstHeight*dstWidth + c*dstHeight*dstWidth + output_y*dstWidth + output_x];
				dst[n* channels *dstHeight*dstWidth + c * dstHeight*dstWidth + output_y * dstWidth + output_x] = val;
			}
		}
		return;
	}
	// Interpolation kernel
	float real_x = area_pixel_compute_source_index(
		width_scale, output_x, align_corners, /*cubic=*/true);
	int in_x = floorf(real_x);
	float t_x = real_x - in_x;

	float real_y = area_pixel_compute_source_index(
		height_scale, output_y, align_corners, /*cubic=*/true);
	int in_y = floorf(real_y);
	float t_y = real_y - in_y;

	for (int n = 0; n < batchsize; n++) {
		for (int c = 0; c < channels; c++) {
			float coefficients[4];

			for (int k = 0; k < 4; k++) {
				coefficients[k] = cubic_interp1d<float>(
					upsample_get_value_bounded(
						src, n, c, batchsize, channels, srcHeight, srcWidth, in_y - 1 + k, in_x - 1),
					upsample_get_value_bounded(
						src, n, c, batchsize, channels, srcHeight, srcWidth, in_y - 1 + k, in_x + 0),
					upsample_get_value_bounded(
						src, n, c, batchsize, channels, srcHeight, srcWidth, in_y - 1 + k, in_x + 1),
					upsample_get_value_bounded(
						src, n, c, batchsize, channels, srcHeight, srcWidth, in_y - 1 + k, in_x + 2),
					t_x);
			}

			dst[n* channels*dstHeight* dstWidth + c*dstHeight* dstWidth +output_y * dstWidth + output_x] = float(cubic_interp1d(
				coefficients[0],
				coefficients[1],
				coefficients[2],
				coefficients[3],
				t_y));
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
	resize_cubic_kernel_torch << <(n + 255)/256, 256 >> > (widthOut * heightOut, pIn_d, widthIn, heightIn, pOut_d, widthOut, heightOut);
}

int main()
{

	torch::Tensor inputs = torch::rand({ 1,3,22,33 }, torch::device(torch::kCPU()));
	auto outputs_torch = torch::upsample_bicubic2d(inputs, { 44,66 }, true);

	float *dstPtr;
	float *srcPtr;
	torch::Tensor outputs_cuda = torch::rand({ 1,3,44,66 }, torch::device(torch::kCPU()));
	int dstLen = 1 * 3 * 44 * 66 * sizeof(float);
	int srcLen = 1 * 3 * 22 * 22 * sizeof(float);
	cudaMalloc(&dstPtr, dstLen);
	cudaMalloc(&srcPtr, srcLen);
	cudaMemcpy(srcPtr, inputs.data_ptr(), srcLen, cudaMemcpyHostToDevice);

	resizeGPU(srcPtr, dstPtr, 33, 22, 66, 44, 3);
	cudaDeviceSynchronize();

	cudaMemcpy(outputs_cuda.data_ptr(), dstPtr, dstLen, cudaMemcpyDeviceToHost);

	cudaFree(srcPtr);
	cudaFree(dstPtr);

	std::cout << outputs_cuda[0][0][42];
	std::cout << outputs_torch[0][0][42];
	return 0;


	//cv::Mat srcImage = cv::imread("C:\\Users\\Administrator\\Pictures\\2007_000799.jpg");
	//srcImage.convertTo(srcImage, CV_32F, 1.0 / 225);
	////cv::cvtColor(srcImage, srcImage, cv::COLOR_BGR2GRAY);
	//const int num_channels = srcImage.channels();
	//cv::Mat dstImage = cv::Mat::zeros(srcImage.size(), srcImage.type());
	//std::cout << "The image shape in opencv is: " << dstImage.size();
	//size_t srcLen = srcImage.rows * srcImage.cols * num_channels * sizeof(float);
	//size_t dstLen = dstImage.rows * dstImage.cols * num_channels * sizeof(float);
	//float *srcPtr;
	//float *dstPtr;
	//cudaMalloc(&srcPtr, srcLen);
	//cudaMalloc(&dstPtr, dstLen);

	//cudaMemcpy(srcPtr, srcImage.data, srcLen, cudaMemcpyHostToDevice);
	////cudaMemcpy(dstPtr, dstImage.data, dstLen, cudaMemcpyHostToDevice);

	//resizeGPU(srcPtr, dstPtr, srcImage.cols, srcImage.rows, dstImage.cols, dstImage.rows, num_channels);
	//cudaDeviceSynchronize();

	////cudaMemcpy(srcImage.data, srcPtr, srcLen, cudaMemcpyDeviceToHost);
	//cudaMemcpy(dstImage.data, dstPtr, dstLen, cudaMemcpyDeviceToHost);
	//
	//cudaFree(srcPtr);
	//cudaFree(dstPtr);

	//cv::imshow("Src image", srcImage);
	//cv::imshow("Dst image", dstImage);
	//cv::waitKey(0);
	//cv::destroyAllWindows();
}
