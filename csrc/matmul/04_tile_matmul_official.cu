#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

// Pull out matrix and shared memory tile size 
const int N = 1 << 10;
const int SHMEM_SIZE = 1 << 10;

__global__ void matrixMul(const float *a, const float *b, float *c) {
  // Block row and column
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  // Thread row and column within Csub
  int rowWithin = threadIdx.y;
  int colWithin = threadIdx.x;

  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Accumulate in temporary variable
  float tmp = 0;

  // Sweep the tile across the Matrix
  for (int i = 0; i < N; i += blockDim.x) {

    // Statically allocated shared memory
    // here SHMEM_SIZE = BLOCK_SIZE * BLOCK_SIZE, it can be imagined as a
    // 2D array, a cache tile here is exactly the same size as a block
    __shared__ float s_a[SHMEM_SIZE];
    __shared__ float s_b[SHMEM_SIZE];

    // Load in elements for this tile
    // rowWithin * BLOCK_SIZE + colWithin
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] =
        b[i * N + threadIdx.y * N + col];

    // Wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // Do matrix multiplication on the small matrix
    for (int j = 0; j < blockDim.x; j++) {
      tmp +=
          s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }

    // Wait for all threads to finish using current tiles before loading in new
    // ones
    __syncthreads();
  }

  // Write back results
  c[row * N + col] = tmp;
}

// Check result on the CPU
void verify_result(vector<float> &a, vector<float> &b, vector<float> &c) {
  // For every row...
  for (float i = 0; i < N; i++) {
    // For every column...
    for (float j = 0; j < N; j++) {
      // For every element in the row-column pair
      float tmp = 0;
      for (float k = 0; k < N; k++) {
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * N + j];
      }

      // Check against the CPU result
      assert(tmp == c[i * N + j]);
    }
  }
}

int main() {
  // Size (in bytes) of matrix
  size_t bytes = N * N * sizeof(float);

  // Host vectors
  vector<float> h_a(N * N);
  vector<float> h_b(N * N);
  vector<float> h_c(N * N);

  // Initialize matrices
  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

  // Allocate device memory
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Copy data to the device
  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  // Threads per CTA dimension
  float THREADS = 32;

  // Blocks per grid dimension (assumes THREADS divides N evenly)
  // Here we can also treat a cache tile THREADS * THREADS = N = the size of block
  float BLOCKS = N / THREADS;

  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  // Launch kernel
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);

  // Copy back to the host
  // cudaMemcpy(h_a.data(), d_a, bytes, cudaMemcpyDeviceToHost);
  // cudaMemcpy(h_b.data(), d_b, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  // Check result
  verify_result(h_a, h_b, h_c);

  cout << "COMPLETED SUCCESSFULLY\n";

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}