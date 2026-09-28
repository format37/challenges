# CUDA challenges

| # | Challenge | Difficulty | Topics |
|---|---|---|---|
| 01 | [Vector Add](01_vector_add/) | easy | thread indexing, grid sizing, bounds checks |
| 02 | [Matrix Transpose](02_matrix_transpose/) | easy → medium | 2D grids, coalescing, shared memory |

## Requirements

- NVIDIA GPU + CUDA toolkit (`nvcc`); tested with CUDA 12.5 on an RTX 4090
- Python 3 + NumPy (for test-data generation)

Code is compiled with `-arch=native`, so it targets whatever GPU is installed.

## Layout

```
cuda/
  common/
    harness.cuh     # shared C++ helpers: data loading, checks, L2-flushed benchmark
    run_tests.sh    # shared runner behind every challenge's test.sh
    datagen.py      # shared Python helpers for gen_data.py
  NN_name/
    README.md       # the task
    starter.cu      # template → copied to solution.cu (gitignored)
    harness.cu      # calls solve(), checks, benchmarks
    gen_data.py     # writes data/case_*/*.bin
    test.sh
```

Data files use this format: `int32 ndim`, `int32 shape[ndim]`, then `float32 data[prod(shape)]`.
