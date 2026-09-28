# Challenges

Small programming challenges for practice. Each one comes with a starter template, a test
harness and reproducible test data.

| Track | Challenges |
|---|---|
| [CUDA](cuda/) | GPU programming: kernels, memory hierarchy, performance |

## How it works

```bash
cd cuda/01_vector_add
./test.sh          # first run: creates solution.cu from starter.cu, generates data/, runs tests
$EDITOR solution.cu
./test.sh          # compile + run every test case
```

- `starter.*` is the committed template.
- `solution*` files are **your** work and are gitignored, so the repo keeps only the challenges and
  you can practice without spoiling them. Keep as many variants as you like (`solution_naive.cu`,
  `solution_tiled.cu`, …) and test one with `./test.sh <file>`.
- `data/` is generated from a fixed seed on first run and isn't committed.
