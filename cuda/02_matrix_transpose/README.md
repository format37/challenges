# Challenge 02: Matrix Transpose

**Difficulty:** easy (correctness) → medium (speed) · **Topics:** 2D grids, `dim3`, memory coalescing, shared memory

## Task

Given a `rows × cols` matrix `input` (row-major), write its transpose into `output`
(`cols × rows`, row-major):

```
output[c][r] = input[r][c]        i.e.   output[c * rows + r] = input[r * cols + c]
```

```
input (2×3)        output (3×2)
1 2 3              1 4
4 5 6      →       2 5
                   3 6
```

## What you write

Edit **`solution.cu`**. The first `./test.sh` run creates it from `starter.cu`. Implement:

```cpp
void solve(const float* input, float* output, int rows, int cols);
```

- Both are **device pointers**, already allocated. `input` is already filled in.
- Launch your kernel inside `solve`. The harness synchronizes and checks for errors.
- Don't change the signature. Don't write to `input`.

## Rules / constraints

- `1 <= rows, cols <= 100,000` and `rows * cols <= 16,777,216`
- Results must match **exactly** (it's just data movement, so no rounding is involved).
- Matrices can be tiny, very tall (`100000 × 3`), very wide (`3 × 100000`), or not a multiple of
  any nice block size (`33 × 65`, `2000 × 3001`).

## Hints

1. Use a 2D block and grid: `dim3 block(X, Y); dim3 grid(..., ...); kernel<<<grid, block>>>(...)`.
   Map `x` to columns and `y` to rows.
2. You need a bounds check on **both** dimensions.
3. Grid dimensions have hardware limits: `gridDim.x` up to 2³¹−1, but **`gridDim.y` and
   `gridDim.z` only up to 65,535**. Look at the tall test case.
4. Lesson from Challenge 01: blocks should have **many** threads (e.g. 256), and the block width
   should be a multiple of 32 (a *warp*).

## Performance levels

Each level is roughly the bandwidth you'd see on the `4096x4096` case. Transpose reads and writes
every element once, so the ideal is a plain memory copy: about 900 GB/s on an RTX 4090.

| Level | ~GB/s | What it takes |
|---|---|---|
| 🥉 Works | < 100 | correct for every case |
| 🥈 Naive done right | ~300 | sensible 2D blocks, one element per thread |
| 🥇 Coalesced | ~800 | shared-memory tile, so **both** reads and writes are coalesced |

**Why is naive stuck at ~300?** A warp is 32 threads that access memory together. When those 32
threads touch 32 *neighbouring* floats, that's one 128-byte transaction (*coalesced*). In a
transpose, if the reads along a row are coalesced, the writes go down a column: 32 addresses
`rows * 4` bytes apart, which means 32 separate transactions.

**The fix (🥇):** each block loads a 32×32 tile into `__shared__` memory with coalesced reads,
calls `__syncthreads()`, then writes the tile out transposed with coalesced writes, so the
"transpose" happens in fast on-chip memory. Watch for **shared-memory bank conflicts**. Declaring
the tile as `[32][33]` instead of `[32][32]` is a classic trick; try to work out why it helps.

## Run the tests

```bash
./test.sh                  # tests solution.cu
./test.sh solution_naive.cu   # keep several versions and compare them
```

## Bonus (optional)

- Have each thread handle several elements (e.g. a `32×8` block covering a `32×32` tile). Faster?
- Compare `[32][32]` and `[32][33]` tiles. How big is the bank-conflict cost?
- Profile it: `ncu --set full ./build/test data/case_11_4096x4096/` and look at
  "Memory Workload Analysis" and sectors per request.

## Evaluation

Ask Claude to evaluate. It will look at correctness on every case, how grid and block sizes are
chosen, the bounds checks, and which performance level you reached.

## Files

| File | Purpose |
|---|---|
| `starter.cu` | template, copied to `solution.cu` on first run |
| `solution*.cu` | **your code** (gitignored) |
| `harness.cu` | loads data, calls `solve`, checks results, measures time (don't edit) |
| `test.sh` | compiles and runs every case (shared logic in `../common/run_tests.sh`) |
| `gen_data.py` | generates `data/` (~190 MB; fixed seed; gitignored, created on first run) |
