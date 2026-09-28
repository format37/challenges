// Shared test-harness helpers for the CUDA challenges.
//
// Data file format (see datagen.py): int32 ndim, int32 shape[ndim], float32 data[prod(shape)].
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            printf("FAIL  CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
                   cudaGetErrorString(err_));                                     \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;
    size_t size() const { return data.size(); }
    size_t bytes() const { return data.size() * sizeof(float); }
};

inline Tensor load_tensor(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { printf("FAIL  cannot open %s\n", path.c_str()); exit(2); }
    Tensor t;
    int ndim = 0;
    if (fread(&ndim, sizeof(int), 1, f) != 1 || ndim < 1 || ndim > 8) {
        printf("FAIL  bad header in %s\n", path.c_str()); exit(2);
    }
    t.shape.resize(ndim);
    if (fread(t.shape.data(), sizeof(int), ndim, f) != (size_t)ndim) {
        printf("FAIL  bad header in %s\n", path.c_str()); exit(2);
    }
    size_t n = 1;
    for (int d : t.shape) n *= d;
    t.data.resize(n);
    if (fread(t.data.data(), sizeof(float), n, f) != n) {
        printf("FAIL  short read in %s\n", path.c_str()); exit(2);
    }
    fclose(f);
    return t;
}

inline float* to_device(const Tensor& t) {
    float* d;
    CUDA_CHECK(cudaMalloc(&d, t.bytes()));
    CUDA_CHECK(cudaMemcpy(d, t.data.data(), t.bytes(), cudaMemcpyHostToDevice));
    return d;
}

// Output buffer pre-filled with NaN (0xFFFFFFFF), so any element the solution skips fails.
inline float* device_nan(size_t n) {
    float* d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d, 0xFF, n * sizeof(float)));
    return d;
}

inline std::vector<float> from_device(const float* d, size_t n) {
    std::vector<float> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

// Launch + runtime error check after the solution's first call.
inline void sync_check() {
    cudaError_t err = cudaGetLastError();
    const char* stage = "kernel launch";
    if (err == cudaSuccess) { err = cudaDeviceSynchronize(); stage = "kernel execution"; }
    if (err != cudaSuccess) {
        printf("FAIL  CUDA error in solve() (%s): %s\n", stage, cudaGetErrorString(err));
        exit(1);
    }
}

// Fails the test if a read-only input was written to.
inline void check_unchanged(const float* d, const Tensor& t, const char* name, const char* desc) {
    if (from_device(d, t.size()) != t.data) {
        printf("FAIL  %-14s input %s was modified\n", desc, name);
        exit(1);
    }
}

// |got - exp| <= rtol * max(1, |exp|). rtol = 0 means exact match.
inline void check_close(const std::vector<float>& got, const Tensor& exp, float rtol, const char* desc) {
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        float tol = rtol * fmaxf(1.0f, fabsf(exp.data[i]));
        if (!(fabsf(got[i] - exp.data[i]) <= tol)) { if (!bad) first = i; ++bad; }
    }
    if (!bad) return;
    printf("FAIL  %-14s %zu wrong element(s); first at i=%zu", desc, bad, first);
    if (exp.shape.size() == 2) printf(" [row %zu, col %zu]", first / exp.shape[1], first % exp.shape[1]);
    printf(": got %g, expected %g\n", got[first], exp.data[first]);
    exit(1);
}

// Average ms per call of fn(). L2 is flushed before each timed call so we measure
// DRAM bandwidth, not cache (RTX 4090 has 72 MB of L2).
template <class F>
float bench_ms(F fn, int reps = 50) {
    const size_t flush_bytes = 256u << 20;
    char* flush;
    CUDA_CHECK(cudaMalloc(&flush, flush_bytes));
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    fn();  // warm-up
    float total = 0;
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaMemset(flush, r, flush_bytes));
        CUDA_CHECK(cudaEventRecord(t0));
        fn();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        total += ms;
    }
    CUDA_CHECK(cudaGetLastError());
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(flush);
    return total / reps;
}

// bytes_moved = total bytes the kernel must read + write.
inline void report_pass(const char* desc, float ms, double bytes_moved) {
    printf("PASS  %-14s %9.4f ms  %7.1f GB/s\n", desc, ms, bytes_moved / (ms * 1e-3) / 1e9);
}
