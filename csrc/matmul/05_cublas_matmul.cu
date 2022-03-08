
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cublas_v2.h"
#include "curand.h"
#include <assert.h>
#include <time.h>
#include <stdio.h>
#include <math.h>


// Check result on the CPU
void verify_result(float *a, float *b, float *c, int N) {
  float temp;
  float epsilon=0.001;
  for (int i = 0; i < N; i++) {
    // For every column...
    for (int j = 0; j < N; j++) {
      // For every element in the row-column pair
      temp = 0;
      for (int k = 0; k < N; k++) {
        // Accumulate the partial results
        temp += a[k * N + i] * b[j * N + k];
      }

      // Check against the CPU result
      assert(fabs(temp - c[j * N + i] < epsilon));
    }
  }
}

int main(){
    //problem size
    int n = 1<<10;
    size_t bytes = n * n * sizeof(float);

    // Declare pointers to matrices on device and host
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    
    // Allocate memory
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Pseudo random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // set the seed
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

    // Fill the matrix with random numbers on the device
    curandGenerateUniform(prng, d_a, n*n);
    curandGenerateUniform(prng, d_b, n*n);

    cublasHandle_t handle;
    cublasCreate(&handle);
}
