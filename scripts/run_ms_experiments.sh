#!/bin/bash

# nodes=`nvidia-smi -L | wc -l`
nodes="2"
if [[ $# -ge 1 ]] ; then
    nodes="$1"
fi

cd "$(dirname $(readlink -f "$0"))/../"
mkdir -p "results"
# conda activate pytorchio-env

num_colss="2 4 6"
truefalse="True False"
# shuffle_types="BLOCKING OVERLAPPED"
probe_sched="STANDARD INVERTED"
p2p_disable=0

# ----------------------------------------------------------------------------------------
# send bench
STDOUT_TEMPLATE="results/send_ms_{}.txt"

for node in $nodes; do
    STDOUT_TEMPLATE=$STDOUT_TEMPLATE NCCL_P2P_DISABLE=$p2p_disable MORSEL_MB_LO="4" SEND_BENCH="CONST_SEND" python multiproc.py benchmarks/send_bench.py $node
    # for num_cols in $num_colss; do
    #     STDOUT_TEMPLATE=$STDOUT_TEMPLATE NCCL_P2P_DISABLE=$p2p_disable MORSEL_MB_LO="4" SEND_BENCH="COMM_SIZE" NUM_COLS=$num_cols python multiproc.py benchmarks/send_bench.py $node
    # done
done

# ----------------------------------------------------------------------------------------
# shuffle bench
STDOUT_TEMPLATE="results/shuffle_ms_{}.txt"
SHUFFLE_LEFT_TUPLES=60000000
SHUFFLE_RIGHT_TUPLES=200000000

for node in $nodes; do
    for num_cols in $num_colss; do
        for USE_OPT in $truefalse; do
            for SHUFFLE_PROBE_SCHED in $probe_sched; do
                # overlapped
                STDOUT_TEMPLATE=$STDOUT_TEMPLATE NCCL_P2P_DISABLE=$p2p_disable MORSEL_MB_LO=128 MORSEL_MB_HI=2048 SHUFFLE_LEFT_TUPLES=$SHUFFLE_LEFT_TUPLES SHUFFLE_RIGHT_TUPLES=$SHUFFLE_RIGHT_TUPLES NUM_COLS=$num_cols SHUFFLE_TYPE=OVERLAPPED USE_RMM=$USE_OPT USE_NET_OPT_FLAGS=$USE_OPT SHUFFLE_PROBE_SCHED=$SHUFFLE_PROBE_SCHED python multiproc.py run_shuffle.py $node
                # blocking
                STDOUT_TEMPLATE=$STDOUT_TEMPLATE NCCL_P2P_DISABLE=$p2p_disable MORSEL_MB_LO=-1 MORSEL_MB_HI=-1 SHUFFLE_LEFT_TUPLES=$SHUFFLE_LEFT_TUPLES SHUFFLE_RIGHT_TUPLES=$SHUFFLE_RIGHT_TUPLES NUM_COLS=$num_cols SHUFFLE_TYPE=BLOCKING USE_RMM=$USE_OPT USE_NET_OPT_FLAGS=$USE_OPT python multiproc.py run_shuffle.py $node
            done
        done
    done
done

# ----------------------------------------------------------------------------------------

STDOUT_FILE=`echo "$STDOUT_TEMPLATE" | sed "s/{}/0/g"`
echo "### END OF BENCH SCRIPT" >> $STDOUT_FILE