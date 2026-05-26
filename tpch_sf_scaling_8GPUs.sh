#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../.."

QUERY="${1:-all}"

if [[ "$QUERY" == "-h" || "$QUERY" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash tpch_sf_scaling_8GPUs.sh [QUERY-ID]

If QUERY-ID is omitted, runs all queries.
Queries: 3 4 5 10 12 14 17
Set STORAGE_SPECIAL_PATH to the common prefix, e.g., /home/users/tpch, if you have directories such as tpch-100/ and tpch-200/
Example: export STORAGE_SPECIAL_PATH="/home/users/tpch"
EOF
  exit 0
fi

STORAGE_THREADS=8
NUM_REPEATS=20
STORAGE_BLOCKING=False

run_query() {
case "$1" in
  3)
    for sf in 100 200 300 400 500 600 700 800 900 1000; do
      echo "Q3 SF=$sf"
      STORAGE_BLOCKING="$STORAGE_BLOCKING" NCCL_P2P_DISABLE=0 SHUFFLE_BUFFER_RIGHT=8 TPCH_SCALE_FACTOR="$sf" TPCH_QUERY=3 STORAGE_THREADS=$STORAGE_THREADS NUM_REPEATS=$NUM_REPEATS python multiproc.py run_tpch.py 8
    done
    ;;

  4)
    for sf in 100 200 300 400 500 600 700 800 900 1000; do
      echo "Q4 SF=$sf"
      STORAGE_BLOCKING="$STORAGE_BLOCKING" NCCL_P2P_DISABLE=0 SHUFFLE_BUFFER_LEFT=200 SHUFFLE_BUFFER_RIGHT=80 TPCH_SCALE_FACTOR="$sf" TPCH_QUERY=4 STORAGE_THREADS=$STORAGE_THREADS NUM_REPEATS=$NUM_REPEATS python multiproc.py run_tpch.py 8
    done
    ;;

  5)
    for sf in 100 200 300 400 500 600 700 800 900 1000; do
      echo "Q5 SF=$sf"
      STORAGE_BLOCKING="$STORAGE_BLOCKING" NCCL_P2P_DISABLE=0 SHUFFLE_BUFFER_LEFT=24 SHUFFLE_BUFFER_RIGHT=8 TPCH_SCALE_FACTOR="$sf" TPCH_QUERY=5 STORAGE_THREADS=$STORAGE_THREADS NUM_REPEATS=$NUM_REPEATS python multiproc.py run_tpch.py 8
    done
    ;;

  10)
    for sf in 100 200 300 400 500 600 700 800 900 1000; do
      echo "Q10 SF=$sf"
      STORAGE_BLOCKING="$STORAGE_BLOCKING" NCCL_P2P_DISABLE=0 SHUFFLE_BUFFER_LEFT=32 SHUFFLE_BUFFER_RIGHT=16 TPCH_SCALE_FACTOR="$sf" TPCH_QUERY=10 STORAGE_THREADS=$STORAGE_THREADS NUM_REPEATS=$NUM_REPEATS python multiproc.py run_tpch.py 8
    done
    ;;

  12)
    for sf in 100 200 300 400 500 600 700 800 900 1000; do
      echo "Q12 SF=$sf"
      STORAGE_BLOCKING="$STORAGE_BLOCKING" NCCL_P2P_DISABLE=0 SHUFFLE_RECV_TYPE=COMB_RECV SHUFFLE_BUFFER_RIGHT=12 TPCH_SCALE_FACTOR="$sf" TPCH_QUERY=12 STORAGE_THREADS=$STORAGE_THREADS NUM_REPEATS=$NUM_REPEATS python multiproc.py run_tpch.py 8
    done
    ;;

  14)
    for sf in 100 200 300 400 500 600 700 800 900 1000; do
      echo "Q14 SF=$sf"
      STORAGE_BLOCKING="$STORAGE_BLOCKING" NCCL_P2P_DISABLE=0 SHUFFLE_BUFFER_LEFT=40 TPCH_SCALE_FACTOR="$sf" TPCH_QUERY=14 STORAGE_THREADS=$STORAGE_THREADS NUM_REPEATS=$NUM_REPEATS python multiproc.py run_tpch.py 8
    done
    ;;

  17)
    for sf in 100 200 300 400 500 600 700 800 900 1000; do
      echo "Q17 SF=$sf"
      STORAGE_BLOCKING="$STORAGE_BLOCKING" NCCL_P2P_DISABLE=0 TPCH_SCALE_FACTOR="$sf" SHUFFLE_BUFFER_RIGHT=8 TPCH_QUERY=17 STORAGE_THREADS=$STORAGE_THREADS NUM_REPEATS=$NUM_REPEATS python multiproc.py run_tpch.py 8
    done
    ;;

  *)
    echo "Unsupported query: $1" >&2
    echo "Supported queries: 3 4 5 10 12 14 17" >&2
    exit 1
    ;;
esac
}

if [[ "$QUERY" == "all" ]]; then
  run_query 3
  run_query 4
  run_query 5
  run_query 10
  run_query 12
  run_query 14
  run_query 17
else
  run_query "$QUERY"
fi
