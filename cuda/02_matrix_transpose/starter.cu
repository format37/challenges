// Challenge 02: Matrix Transpose — starter. The test runner copies this to solution.cu (gitignored).
// input is rows x cols, output is cols x rows, both row-major DEVICE pointers.
// Compute output[c][r] = input[r][c].
#include <cuda_runtime.h>

__global__ void transpose(const float* input, float* output, int rows, int cols) {
    // TODO
}

void solve(const float* input, float* output, int rows, int cols) {
    // TODO: choose a 2D block (dim3) and grid, then launch transpose
}
