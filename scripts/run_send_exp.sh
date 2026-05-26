#!/bin/bash

cd "$(dirname $(readlink -f "$0"))/../"

# conda activate pytorchio-env

# nodes=`nvidia-smi -L | wc -l`
nodes="2"
truefalse="True False"
# shuffle_types="BLOCKING OVERLAPPED OVERLAPPED_GROUPS"
num_colss="1 2 4 6"

# const send
STDOUT_TEMPLATE="results/const_send_exp{}.txt"
# const send
for node in $nodes; do
    for USE_RMM in $truefalse; do
        for USE_NET_OPT_FLAGS in $truefalse; do
            SEND_BENCH="CONST_SEND" STDOUT_TEMPLATE=$STDOUT_TEMPLATE USE_RMM=$USE_RMM USE_NET_OPT_FLAGS=$USE_NET_OPT_FLAGS python multiproc.py benchmarks/send_bench.py $node
        done
    done
done


# comm size
STDOUT_TEMPLATE="results/comm_size_exp{}.txt"
for node in $nodes; do
    for num_cols in $num_colss; do
        # for USE_RMM in $truefalse; do
        #     for USE_NET_OPT_FLAGS in $truefalse; do
                SEND_BENCH="COMM_SIZE" STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_COLS=$num_cols python multiproc.py benchmarks/send_bench.py $node
        #     done
        # done
    done
done



STDOUT_FILE=`echo "$STDOUT_TEMPLATE" | sed "s/{}/0/g"`
echo "### END OF BENCH SCRIPT" >> $STDOUT_FILE
