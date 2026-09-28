#!/usr/bin/env bash
# Usage: ./test.sh [solution_file]   (default: solution.cu, created from starter.cu if missing)
exec "$(dirname "$0")/../common/run_tests.sh" "$(dirname "$0")" "$@"
