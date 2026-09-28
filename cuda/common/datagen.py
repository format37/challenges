"""Shared helpers for gen_data.py scripts.

Data file format: int32 ndim, int32 shape[ndim], float32 data[prod(shape)].
"""
import os
import shutil

import numpy as np


def write_tensor(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with open(path, "wb") as f:
        np.array([arr.ndim, *arr.shape], dtype=np.int32).tofile(f)
        arr.tofile(f)


def write_cases(data_dir, cases):
    """cases: {case_name: {file_stem: array}} -> data_dir/case_name/file_stem.bin"""
    shutil.rmtree(data_dir, ignore_errors=True)
    for name, files in cases.items():
        d = os.path.join(data_dir, name)
        os.makedirs(d)
        for stem, arr in files.items():
            write_tensor(os.path.join(d, stem + ".bin"), arr)
        print(f"{name}: " + ", ".join(f"{k}{list(v.shape)}" for k, v in files.items()))
