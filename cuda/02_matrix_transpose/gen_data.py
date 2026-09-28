#!/usr/bin/env python3
"""Generate test cases for Challenge 02 (fixed seed, reproducible)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from datagen import write_cases  # noqa: E402

rng = np.random.default_rng(2)


def case(rows, cols):
    x = rng.uniform(-1000, 1000, (rows, cols)).astype(np.float32)
    return {"input": x, "expected": x.T}


if __name__ == "__main__":
    write_cases(os.path.join(HERE, "data"), {
        f"case_{i:02d}_{r}x{c}": case(r, c)
        for i, (r, c) in enumerate([
            (1, 1),
            (1, 7),
            (7, 1),
            (32, 32),
            (33, 65),
            (100, 257),
            (100_000, 3),
            (3, 100_000),
            (1024, 1024),
            (2000, 3001),
            (4096, 4096),
        ], start=1)
    })
