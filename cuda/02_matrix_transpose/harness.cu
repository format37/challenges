// Test harness — do not edit.  Usage: ./build/test <case_dir>
#include "harness.cuh"

void solve(const float* input, float* output, int rows, int cols);

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <case_dir>\n", argv[0]); return 2; }
    std::string dir = argv[1];
    Tensor in = load_tensor(dir + "/input.bin"), exp = load_tensor(dir + "/expected.bin");
    int rows = in.shape[0], cols = in.shape[1];
    char desc[32];
    snprintf(desc, sizeof desc, "%dx%d", rows, cols);

    float *dIn = to_device(in), *dOut = device_nan(in.size());

    solve(dIn, dOut, rows, cols);
    sync_check();
    check_unchanged(dIn, in, "input", desc);
    check_close(from_device(dOut, in.size()), exp, 0.0f, desc);  // transpose must be exact

    float ms = bench_ms([&] { solve(dIn, dOut, rows, cols); });
    report_pass(desc, ms, 2.0 * in.bytes());  // read input, write output

    cudaFree(dIn); cudaFree(dOut);
    return 0;
}
