#!/usr/bin/env bash
# Shared test runner. Called by each challenge's test.sh:
#   run_tests.sh <challenge_dir> [solution_file]
#
# - If solution.cu doesn't exist yet, it's created from starter.cu.
# - data/ is generated on first run (it's gitignored).
# - Any file can be tested, e.g. ./test.sh solution_float4.cu
set -uo pipefail

CH=$(cd "$1" && pwd)
COMMON=$(cd "$(dirname "$0")" && pwd)
SOL=$(realpath -m "${2:-$CH/solution.cu}")
export PATH=/usr/local/cuda/bin:$PATH
cd "$CH"

if [ ! -f "$SOL" ]; then
    if [ "$SOL" = "$CH/solution.cu" ]; then
        cp starter.cu solution.cu
        echo ">> Created solution.cu from starter.cu. Write your solution there."
    else
        echo "No such file: $SOL"; exit 1
    fi
fi

if [ ! -d data ]; then
    echo "== generating data =="
    python3 gen_data.py || exit 1
fi

mkdir -p build
case "$SOL" in "$CH"/*) SHOW=${SOL#"$CH"/} ;; *) SHOW=$SOL ;; esac
echo "== compiling $SHOW =="
nvcc -O2 -arch=native -std=c++17 -I "$COMMON" -o build/test harness.cu "$SOL" || { echo "COMPILE FAILED"; exit 1; }

echo "== running =="
pass=0; total=0
for d in data/case_*/; do
    total=$((total + 1))
    printf "%-22s " "$(basename "$d")"
    timeout 30 ./build/test "$d"
    rc=$?
    if [ $rc -eq 0 ]; then pass=$((pass + 1))
    elif [ $rc -eq 124 ]; then echo "FAIL  timeout (30 s)"
    elif [ $rc -gt 128 ]; then echo "FAIL  crashed (signal $((rc - 128)))"
    fi
done

echo "== $pass / $total passed =="
[ $pass -eq $total ]
