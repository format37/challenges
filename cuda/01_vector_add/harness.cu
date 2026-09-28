// Test harness — do not edit.  Usage: ./build/test <case_dir>
#include "harness.cuh"

void solve(const float* A, const float* B, float* C, int N);

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <case_dir>\n", argv[0]); return 2; }
    std::string dir = argv[1];
    Tensor a = load_tensor(dir + "/a.bin"), b = load_tensor(dir + "/b.bin");
    Tensor exp = load_tensor(dir + "/expected.bin");
    int n = (int)a.size();
    char desc[32];
    snprintf(desc, sizeof desc, "N=%d", n);

    float *dA = to_device(a), *dB = to_device(b), *dC = device_nan(n);

    solve(dA, dB, dC, n);
    sync_check();
    check_unchanged(dA, a, "A", desc);
    check_unchanged(dB, b, "B", desc);
    check_close(from_device(dC, n), exp, 1e-5f, desc);

    float ms = bench_ms([&] { solve(dA, dB, dC, n); });
    report_pass(desc, ms, 3.0 * a.bytes());  // read A, read B, write C

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
