// Challenge 01: Vector Add — starter. The test runner copies this to solution.cu (gitignored).
// A, B, C are DEVICE pointers. Compute C[i] = A[i] + B[i] for i in [0, N).
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    // TODO
}

void solve(const float* A, const float* B, float* C, int N) {
    // TODO: pick threads per block and number of blocks, then launch vector_add
}
