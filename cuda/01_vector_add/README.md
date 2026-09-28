# Challenge 01: Vector Add

**Difficulty:** easy · **Topics:** thread indexing, grid sizing, bounds checks

## Task

Given two float arrays `A` and `B` of length `N`, compute

```
C[i] = A[i] + B[i]      for every i in [0, N)
```

on the GPU, using **many threads in parallel**, one element per thread.

## What you write

Edit **`solution.cu`**. The first `./test.sh` run creates it from `starter.cu`. Implement:

```cpp
void solve(const float* A, const float* B, float* C, int N);
```

- `A`, `B`, `C` are **device pointers**. They are already allocated and `A`, `B`
  are already filled in, so you don't need `cudaMalloc` or `cudaMemcpy`.
- Inside `solve`, launch your kernel. The harness synchronizes and checks for errors afterwards.
- Don't change the signature. Don't write to `A` or `B`.

## Rules / constraints

- `1 <= N <= 2,097,152`
- Your code must be correct for **any** `N`, including ones that are not a multiple of your block size
  (e.g. `N = 1`, `N = 7`, `N = 1025`, `N = 1,000,003`).
- The harness fills `C` with NaN before calling you, so a skipped element fails the test.
- Tolerance: `|C[i] - expected[i]| <= 1e-5 * max(1, |expected[i]|)`.

## Hints

1. Global thread index: `int i = blockIdx.x * blockDim.x + threadIdx.x;`
2. Number of blocks is a ceiling division: `(N + threads - 1) / threads`.
3. The last block usually has more threads than elements left. What should those extra threads do?

## Run the tests

```bash
./test.sh                    # tests solution.cu
./test.sh solution_float4.cu # tests any other file, handy for trying variants
```

The output shows PASS/FAIL per case, and for passing cases the kernel time and effective memory
bandwidth. For reference, an RTX 4090 peaks at about 1000 GB/s. Only the largest cases mean much
for performance. Small cases always look slow because each run has a fixed kernel-launch cost of
about 2–3 µs, however little work it does.

## Bonus (optional)

- **Grid-stride loop:** launch a *fixed* number of blocks (e.g. `4 * number_of_SMs`) and have each
  thread loop over several elements. Does it get faster or slower?
- **Vectorized loads:** process 4 elements per thread using `float4`. Be careful with the tail when
  `N % 4 != 0`!
- Try block sizes 32, 128, 256, 1024 and compare the bandwidth numbers.

## Evaluation

When you're done, ask Claude to evaluate. It will look at:
1. **Correctness:** all test cases pass.
2. **Robustness:** bounds check and correct grid size for any `N`.
3. **Code quality:** readable, error checking where it makes sense.
4. **Performance:** how close you get to the GPU's memory bandwidth.

## Files

| File | Purpose |
|---|---|
| `starter.cu` | template, copied to `solution.cu` on first run |
| `solution*.cu` | **your code** (gitignored) |
| `harness.cu` | loads data, calls `solve`, checks results, measures time (don't edit) |
| `test.sh` | compiles and runs every case (shared logic in `../common/run_tests.sh`) |
| `gen_data.py` | generates `data/` (fixed seed; gitignored, created on first run) |
