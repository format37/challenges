#!/usr/bin/env python3
"""Generate test cases for Challenge 01 (fixed seed, reproducible)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from datagen import write_cases  # noqa: E402

rng = np.random.default_rng(42)


def case(a, b):
    return {"a": a, "b": b, "expected": a + b}


def random_case(n):
    return case(rng.uniform(-100, 100, n).astype(np.float32), rng.uniform(-100, 100, n).astype(np.float32))


def special_case():
    vals = np.array([0.0, -0.0, 1.0, -1.0, 1e-30, -1e-30, 3.4e38, -3.4e38, 1e7, 0.5, 123.456, -7.0],
                    dtype=np.float32)
    a = np.tile(vals, 11)[:129]
    b = np.roll(np.tile(vals[::-1], 11), 3)[:129].copy()
    b[np.abs(a) == np.float32(3.4e38)] = 0.0  # avoid inf from max + max
    return case(a, b)


if __name__ == "__main__":
    write_cases(os.path.join(HERE, "data"), {
        "case_01_n1": random_case(1),
        "case_02_n7": random_case(7),
        "case_03_n256": random_case(256),
        "case_04_n1000": random_case(1000),
        "case_05_n1025": random_case(1025),
        "case_06_special": special_case(),
        "case_07_n65537": random_case(65537),
        "case_08_n1000003": random_case(1_000_003),
        "case_09_n2097152": random_case(2_097_152),
    })
