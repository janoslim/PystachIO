#!/bin/bash

cd "$(dirname $(readlink -f "$0"))/../"

# conda activate pytorchio-env


STDOUT_TEMPLATE="results/tpch_{}.txt"
nodes=`nvidia-smi -L | wc -l`
TPCH_SCALE_FACTOR=200
truefalse="True False"
shuffle_types="BLOCKING OVERLAPPED"
# tpch_queries="1 3 4 5 6 12 14"
tpch_queries="14"
# tpch_queries="line_order_shuffle"
# storage_threadss="1 2 4 8"
storage_threadss="8"


# tpch
STDOUT_TEMPLATE="results/tpch_queries_{}.txt"
for STORAGE_BLOCKING in $truefalse; do
    for SHUFFLE_TYPE in $shuffle_types; do
        for TPCH_QUERY in $tpch_queries; do
            for STORAGE_THREADS in $storage_threadss; do
                if [[ ("$STORAGE_BLOCKING" == "True") && ("$SHUFFLE_TYPE" == "BLOCKING") ]]; then
                    continue
                fi
                # echo "STORAGE_BLOCKING=$STORAGE_BLOCKING SHUFFLE_TYPE=$SHUFFLE_TYPE"
                STDOUT_TEMPLATE=$STDOUT_TEMPLATE SHUFFLE_TYPE=$SHUFFLE_TYPE TPCH_SCALE_FACTOR=$TPCH_SCALE_FACTOR TPCH_QUERY=$TPCH_QUERY STORAGE_THREADS=$STORAGE_THREADS NUM_REPEATS=3 STORAGE_BLOCKING=$STORAGE_BLOCKING STORAGE_LOGGING_LEVEL=WARN PRUNING_FLAG=False python multiproc.py run_tpch.py $nodes
            done
        done
    done
done

# combining recv buffers
# STDOUT_TEMPLATE="results/tpch_recvbuf_htsize_{}.txt"
# shuffle_recv_types="RECV_PER_RANK COMB_RECV"
# for TPCH_QUERY in $tpch_queries; do
#     for HASH_SIZE_NAIVE in $truefalse; do
#         for SHUFFLE_RECV_TYPE in $shuffle_recv_types; do
#             for STORAGE_THREADS in $storage_threadss; do
#                 # echo "STORAGE_BLOCKING=$STORAGE_BLOCKING SHUFFLE_TYPE=$SHUFFLE_TYPE"
#                 STDOUT_TEMPLATE=$STDOUT_TEMPLATE TPCH_SCALE_FACTOR=$TPCH_SCALE_FACTOR TPCH_QUERY=$TPCH_QUERY STORAGE_THREADS=$STORAGE_THREADS SHUFFLE_RECV_TYPE=$SHUFFLE_RECV_TYPE HASH_SIZE_NAIVE=$HASH_SIZE_NAIVE NUM_REPEATS=3 STORAGE_LOGGING_LEVEL=WARN PRUNING_FLAG=False python multiproc.py run_tpch.py $nodes
#             done
#         done
#     done
# done


# varying row group size
# STDOUT_TEMPLATE="results/rg_size_{}.txt"
# rg_sizes="100000 1000000 10000000 20000000 500000 5000000"
# for rg_size in $rg_sizes; do
#     for STORAGE_BLOCKING in $truefalse; do
#         for SHUFFLE_TYPE in $shuffle_types; do
#             for TPCH_QUERY in $tpch_queries; do
#                 for STORAGE_THREADS in $storage_threadss; do
#                     if [[ ("$STORAGE_BLOCKING" == "True") && ("$SHUFFLE_TYPE" == "BLOCKING") ]]; then
#                         continue
#                     fi
#                     # echo "STORAGE_BLOCKING=$STORAGE_BLOCKING SHUFFLE_TYPE=$SHUFFLE_TYPE"
#                     STDOUT_TEMPLATE=$STDOUT_TEMPLATE SHUFFLE_TYPE=$SHUFFLE_TYPE TPCH_SCALE_FACTOR=$TPCH_SCALE_FACTOR TPCH_QUERY=$TPCH_QUERY STORAGE_THREADS=$STORAGE_THREADS NUM_REPEATS=3 STORAGE_SPECIAL_PATH="/RGs/$rg_size/" STORAGE_BLOCKING=$STORAGE_BLOCKING STORAGE_LOGGING_LEVEL=WARN PRUNING_FLAG=False python multiproc.py run_tpch.py $nodes
#                 done
#             done
#         done
#     done
# done


STDOUT_FILE=`echo "$STDOUT_TEMPLATE" | sed "s/{}/0/g"`
echo "### END OF BENCH SCRIPT" >> $STDOUT_FILE