#!/bin/bash

cd "$(dirname $(readlink -f "$0"))/../"

# conda activate pytorchio-env

# nodes=`nvidia-smi -L | wc -l`
# nodes="2"
nodes="2 3 4 5 6 7 8"
truefalse="True False"
# shuffle_types="BLOCKING OVERLAPPED OVERLAPPED_GROUPS"
shuffle_types="BLOCKING OVERLAPPED"
send_types="SEND_PER_RANK COMB_SEND"
recv_types="RECV_PER_RANK COMB_RECV"
probe_sched="STANDARD INVERTED"
num_colss="4"
shuffle_probe_types="FULL NOOP"

# nccl bench
# USE_NET_OPT_FLAGS=True python multiproc.py benchmarks/nccl_tests.py $nodes

# ----------------------------------------------------------------------------------------
# shuffle bench naive
# STDOUT_TEMPLATE="results/shuffle_nodes_{}.txt"
# STDOUT_TEMPLATE="results/shuffle_all_{}.txt"
# nodes="3 4 5"
# for node in $nodes; do
#     for num_cols in $num_colss; do
#         for USE_RMM in $truefalse; do
#             for USE_NET_OPT_FLAGS in $truefalse; do
#                 for SHUFFLE_PROBE_TYPE in $shuffle_probe_types; do
#                     # blocking chunked
#                     STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=1 MORSEL_MB_HI=2048 NUM_COLS=$num_cols SHUFFLE_TYPE=BLOCKING USE_RMM=$USE_RMM USE_NET_OPT_FLAGS=$USE_NET_OPT_FLAGS SHUFFLE_PROBE_TYPE=$SHUFFLE_PROBE_TYPE python multiproc.py run_shuffle.py $node
#                     # blocking full
#                     STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=-1 MORSEL_MB_HI=-1 NUM_COLS=$num_cols SHUFFLE_TYPE=BLOCKING USE_RMM=$USE_RMM USE_NET_OPT_FLAGS=$USE_NET_OPT_FLAGS SHUFFLE_PROBE_TYPE=$SHUFFLE_PROBE_TYPE python multiproc.py run_shuffle.py $node
#                 done
#             done
#         done
#     done
# done
# # # shuffle bench overlapped
# for node in $nodes; do
#     for num_cols in $num_colss; do
#         for USE_RMM in $truefalse; do
#             for USE_NET_OPT_FLAGS in $truefalse; do
#                 for SHUFFLE_PROBE_TYPE in $shuffle_probe_types; do
#                     for SHUFFLE_PROBE_SCHED in $probe_sched; do
#                         STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=1 MORSEL_MB_HI=2048 NUM_COLS=$num_cols SHUFFLE_TYPE=OVERLAPPED USE_RMM=$USE_RMM USE_NET_OPT_FLAGS=$USE_NET_OPT_FLAGS SHUFFLE_PROBE_TYPE=$SHUFFLE_PROBE_TYPE SHUFFLE_PROBE_SCHED=$SHUFFLE_PROBE_SCHED python multiproc.py run_shuffle.py $node
#                         # STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=512 MORSEL_MB_HI=2048 NUM_COLS=$num_cols SHUFFLE_TYPE=OVERLAPPED USE_RMM=$USE_RMM USE_NET_OPT_FLAGS=$USE_NET_OPT_FLAGS SHUFFLE_PROBE_TYPE=$SHUFFLE_PROBE_TYPE SHUFFLE_PROBE_SCHED=$SHUFFLE_PROBE_SCHED python multiproc.py run_shuffle.py $node
#                     done
#                 done
#             done
#         done
#     done
# done

# ----------------------------------------------------------------------------------------
# for send/recv types:
# STDOUT_TEMPLATE="results/send_recv_types_{}.txt"
# shuffle_types="OVERLAPPED OVERLAPPED_GROUPS"
# for node in $nodes; do
#     for SHUFFLE_TYPE in $shuffle_types; do
#         for send_type in $send_types; do
#             for recv_type in $recv_types; do
#                     STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=1 MORSEL_MB_HI=256 NUM_COLS=6 SHUFFLE_TYPE=$SHUFFLE_TYPE SHUFFLE_SEND_TYPE=$send_type SHUFFLE_RECV_TYPE=$recv_type python multiproc.py run_shuffle.py $node
#                     # STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=512 MORSEL_MB_HI=2048 NUM_COLS=6 SHUFFLE_TYPE=$SHUFFLE_TYPE SHUFFLE_SEND_TYPE=$send_type SHUFFLE_RECV_TYPE=$recv_type python multiproc.py run_shuffle.py $node
#             done
#         done
#     done
# done

# ----------------------------------------------------------------------------------------
# FIXED TUPLE COUNTS:
# SHUFFLE_LEFT_TUPLES=80000000
# SHUFFLE_RIGHT_TUPLES=200000000
# STDOUT_TEMPLATE="results/fixed_tuples_{}.txt"
# for node in $nodes; do
#     for num_cols in $num_colss; do
#         for USE_RMM in $truefalse; do
#             for USE_NET_OPT_FLAGS in $truefalse; do
#                 for SHUFFLE_PROBE_TYPE in $shuffle_probe_types; do
#                     STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=2048 MORSEL_MB_HI=2048 NUM_COLS=$num_cols SHUFFLE_TYPE=BLOCKING USE_RMM=$USE_RMM USE_NET_OPT_FLAGS=$USE_NET_OPT_FLAGS SHUFFLE_PROBE_TYPE=$SHUFFLE_PROBE_TYPE SHUFFLE_LEFT_TUPLES=$SHUFFLE_LEFT_TUPLES SHUFFLE_RIGHT_TUPLES=$SHUFFLE_RIGHT_TUPLES python multiproc.py run_shuffle.py $node
#                 done
#             done
#         done
#     done
# done
# for node in $nodes; do
#     for num_cols in $num_colss; do
#         for USE_RMM in $truefalse; do
#             for USE_NET_OPT_FLAGS in $truefalse; do
#                 for SHUFFLE_PROBE_TYPE in $shuffle_probe_types; do
#                     STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=512 MORSEL_MB_HI=2048 NUM_COLS=$num_cols SHUFFLE_TYPE=OVERLAPPED USE_RMM=$USE_RMM USE_NET_OPT_FLAGS=$USE_NET_OPT_FLAGS SHUFFLE_PROBE_TYPE=$SHUFFLE_PROBE_TYPE SHUFFLE_LEFT_TUPLES=$SHUFFLE_LEFT_TUPLES SHUFFLE_RIGHT_TUPLES=$SHUFFLE_RIGHT_TUPLES python multiproc.py run_shuffle.py $node
#                 done
#             done
#         done
#     done
# done

# ----------------------------------------------------------------------------------------
# NUM_COLS=6 MORSEL_MB_LO=3 MORSEL_MB_HIGH=3072 SHUFFLE_AGG_GROUPS=10 SHUFFLE_LEFT_TUPLES=60000000 SHUFFLE_RIGHT_TUPLES=200000000 SHUFFLE_RIGHT_REPEAT=1 SHUFFLE_JOIN_RATIO=0.5
# Computation Tradeoff
# SHUFFLE_LEFT_TUPLES=60000000
# SHUFFLE_RIGHT_TUPLES=200000000
# STDOUT_TEMPLATE="results/shuffle_computation_{}.txt"

# for SHUFFLE_PROBE_TYPE in $shuffle_probe_types; do
#     for SHUFFLE_PROBE_SCHED in $probe_sched; do
#         for USE_RMM in $truefalse; do
#             for USE_NET_OPT_FLAGS in $truefalse; do
#                 # without aggregation
#                 STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_REPEATS=20 MORSEL_MB_LO=3 MORSEL_MB_HI=3072 NUM_COLS=6 USE_RMM=$USE_RMM USE_NET_OPT_FLAGS=$USE_NET_OPT_FLAGS SHUFFLE_LEFT_TUPLES=$SHUFFLE_LEFT_TUPLES SHUFFLE_RIGHT_TUPLES=$SHUFFLE_RIGHT_TUPLES SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_PROBE_TYPE=$SHUFFLE_PROBE_TYPE SHUFFLE_PROBE_SCHED=$SHUFFLE_PROBE_SCHED SHUFFLE_RIGHT_REPEAT=1 python multiproc.py run_shuffle.py 2

#                 # with aggregation
#                 SHUFFLE_AGG_GROUPS=10 NUM_REPEATS=20 STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=3 MORSEL_MB_HI=3072 USE_RMM=$USE_RMM USE_NET_OPT_FLAGS=$USE_NET_OPT_FLAGS NUM_COLS=6 SHUFFLE_LEFT_TUPLES=$SHUFFLE_LEFT_TUPLES SHUFFLE_RIGHT_TUPLES=$SHUFFLE_RIGHT_TUPLES SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_PROBE_TYPE=$SHUFFLE_PROBE_TYPE SHUFFLE_PROBE_SCHED=$SHUFFLE_PROBE_SCHED SHUFFLE_RIGHT_REPEAT=1 python multiproc.py run_shuffle.py 2
#             done
#         done
#     done
# done
# simpler tradeoff:
# SHUFFLE_LEFT_TUPLES=60000000
# SHUFFLE_RIGHT_TUPLES=200000000
# STDOUT_TEMPLATE="results/shuffle_computation2_{}.txt"

# USE_NET_OPT_FLAGS=True SHUFFLE_PROBE_TYPE=NOOP NUM_REPEATS=20 STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=3 MORSEL_MB_HI=3072 NUM_COLS=6 SHUFFLE_LEFT_TUPLES=$SHUFFLE_LEFT_TUPLES SHUFFLE_RIGHT_TUPLES=$SHUFFLE_RIGHT_TUPLES SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_RIGHT_REPEAT=1 python multiproc.py run_shuffle.py 2
# USE_NET_OPT_FLAGS=False SHUFFLE_PROBE_TYPE=NOOP NUM_REPEATS=20 STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=3 MORSEL_MB_HI=3072 NUM_COLS=6 SHUFFLE_LEFT_TUPLES=$SHUFFLE_LEFT_TUPLES SHUFFLE_RIGHT_TUPLES=$SHUFFLE_RIGHT_TUPLES SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_RIGHT_REPEAT=1 python multiproc.py run_shuffle.py 2


# ----------------------------------------------------------------------------------------
# scalability2_fixed_total
# SHUFFLE_LEFT_TUPLES=60000000
# SHUFFLE_RIGHT_TUPLES=160000000
# left_total=120000000
# right_total=320000000
# STDOUT_TEMPLATE="results/shuffle_scalability2_fixed_total_{}.txt"

# for node in $nodes; do
#     for num_cols in $num_colss; do
#         for SHUFFLE_PROBE_TYPE in $shuffle_probe_types; do
#             # default blocking
#             STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_REPEATS=20 USE_RMM=False USE_NET_OPT_FLAGS=False SHUFFLE_TYPE=BLOCKING MORSEL_MB_LO=-1 NUM_COLS=$num_cols SHUFFLE_LEFT_TUPLES=$(($left_total/$node)) SHUFFLE_RIGHT_TUPLES=$(($right_total/$node)) SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_RIGHT_REPEAT=1 SHUFFLE_PROBE_TYPE=$SHUFFLE_PROBE_TYPE python multiproc.py run_shuffle.py $node
#             # + rmm + net opt flags
#             STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_REPEATS=20 SHUFFLE_TYPE=BLOCKING MORSEL_MB_LO=-1 NUM_COLS=$num_cols SHUFFLE_LEFT_TUPLES=$(($left_total/$node)) SHUFFLE_RIGHT_TUPLES=$(($right_total/$node)) SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_RIGHT_REPEAT=1 SHUFFLE_PROBE_TYPE=$SHUFFLE_PROBE_TYPE python multiproc.py run_shuffle.py $node
#             # + chunking
#             STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_REPEATS=20 MORSEL_MB_LO=256 MORSEL_MB_HI=1024 NUM_COLS=$num_cols SHUFFLE_LEFT_TUPLES=$(($left_total/$node)) SHUFFLE_RIGHT_TUPLES=$(($right_total/$node)) SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_RIGHT_REPEAT=1 SHUFFLE_PROBE_TYPE=$SHUFFLE_PROBE_TYPE SHUFFLE_PROBE_SCHED=STANDARD python multiproc.py run_shuffle.py $node
#             # + late probing
#             STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_REPEATS=20 MORSEL_MB_LO=256 MORSEL_MB_HI=1024 NUM_COLS=$num_cols SHUFFLE_LEFT_TUPLES=$(($left_total/$node)) SHUFFLE_RIGHT_TUPLES=$(($right_total/$node)) SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_RIGHT_REPEAT=1 SHUFFLE_PROBE_TYPE=$SHUFFLE_PROBE_TYPE python multiproc.py run_shuffle.py $node
#         done
#     done
# done

# ----------------------------------------------------------------------------------------
# skewed distribution
# left_total=120000000
# right_total=320000000
# zipfs="1.0 0.75 0.5 0.0"
# STDOUT_TEMPLATE="results/shuffle_skewed_{}.txt"

# for node in $nodes; do
#     for num_cols in $num_colss; do
#         for right_zipf in $zipfs; do
#             # default blocking
#             STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_REPEATS=5 USE_RMM=False USE_NET_OPT_FLAGS=False SHUFFLE_TYPE=BLOCKING MORSEL_MB_LO=-1 NUM_COLS=$num_cols SHUFFLE_LEFT_TUPLES=$(($left_total/$node)) SHUFFLE_RIGHT_TUPLES=$(($right_total/$node)) SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_ZIPF_RIGHT=$right_zipf SHUFFLE_RIGHT_REPEAT=1 python multiproc.py run_shuffle.py $node
#             # + rmm + net opt flags
#             STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_REPEATS=5 SHUFFLE_TYPE=BLOCKING MORSEL_MB_LO=-1 NUM_COLS=$num_cols SHUFFLE_LEFT_TUPLES=$(($left_total/$node)) SHUFFLE_RIGHT_TUPLES=$(($right_total/$node)) SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_ZIPF_RIGHT=$right_zipf SHUFFLE_RIGHT_REPEAT=1 python multiproc.py run_shuffle.py $node
#             # + chunking
#             STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_REPEATS=5 MORSEL_MB_LO=256 MORSEL_MB_HI=1024 NUM_COLS=$num_cols SHUFFLE_LEFT_TUPLES=$(($left_total/$node)) SHUFFLE_RIGHT_TUPLES=$(($right_total/$node)) SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_ZIPF_RIGHT=$right_zipf SHUFFLE_RIGHT_REPEAT=1 SHUFFLE_PROBE_SCHED=STANDARD python multiproc.py run_shuffle.py $node
#             # + late probing
#             STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_REPEATS=5 MORSEL_MB_LO=256 MORSEL_MB_HI=1024 NUM_COLS=$num_cols SHUFFLE_LEFT_TUPLES=$(($left_total/$node)) SHUFFLE_RIGHT_TUPLES=$(($right_total/$node)) SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_ZIPF_RIGHT=$right_zipf SHUFFLE_RIGHT_REPEAT=1 python multiproc.py run_shuffle.py $node
#         done
#     done
# done


# ----------------------------------------------------------------------------------------
# increasing nodes
# left_total="240000000"
# right_total="840000000"
# STDOUT_TEMPLATE="results/shuffle_increasing_nodes_{}.txt"

# for node in $nodes; do
#     for num_cols in $num_colss; do
#         # default blocking
#         STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_REPEATS=10 USE_RMM=False USE_NET_OPT_FLAGS=False SHUFFLE_TYPE=BLOCKING MORSEL_MB_LO=-1 NUM_COLS=$num_cols SHUFFLE_LEFT_TUPLES=$(($left_total/$node)) SHUFFLE_RIGHT_TUPLES=$(($right_total/$node)) SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_RIGHT_REPEAT=1 python multiproc.py run_shuffle.py $node
#         # + rmm + net opt flags
#         STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_REPEATS=10 SHUFFLE_TYPE=BLOCKING MORSEL_MB_LO=-1 NUM_COLS=$num_cols SHUFFLE_LEFT_TUPLES=$(($left_total/$node)) SHUFFLE_RIGHT_TUPLES=$(($right_total/$node)) SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_RIGHT_REPEAT=1 python multiproc.py run_shuffle.py $node
#         # + chunking
#         STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_REPEATS=10 MORSEL_MB_LO=512 MORSEL_MB_HI=2048 NUM_COLS=$num_cols SHUFFLE_LEFT_TUPLES=$(($left_total/$node)) SHUFFLE_RIGHT_TUPLES=$(($right_total/$node)) SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_RIGHT_REPEAT=1 SHUFFLE_PROBE_SCHED=STANDARD python multiproc.py run_shuffle.py $node
#         # + late probing
#         STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_REPEATS=10 MORSEL_MB_LO=512 MORSEL_MB_HI=2048 NUM_COLS=$num_cols SHUFFLE_LEFT_TUPLES=$(($left_total/$node)) SHUFFLE_RIGHT_TUPLES=$(($right_total/$node)) SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_RIGHT_REPEAT=1 python multiproc.py run_shuffle.py $node
#     done
# done


# ----------------------------------------------------------------------------------------
# increasing_tuples probe side scaling
# left_total=120000000
# right_totals="1200000000 1080000000 960000000 840000000 720000000 600000000 480000000 360000000 240000000"
# STDOUT_TEMPLATE="results/shuffle_increasing_tuples_{}.txt"

# for node in $nodes; do
#     for right_total in $right_totals; do
#         # default blocking
#         STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_REPEATS=5 USE_RMM=False USE_NET_OPT_FLAGS=False SHUFFLE_TYPE=BLOCKING MORSEL_MB_LO=-1 NUM_COLS=4 SHUFFLE_LEFT_TUPLES=$(($left_total/$node)) SHUFFLE_RIGHT_TUPLES=$(($right_total/$node)) SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_RIGHT_REPEAT=1 python multiproc.py run_shuffle.py $node
#         # + rmm + net opt flags
#         STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_REPEATS=5 SHUFFLE_TYPE=BLOCKING MORSEL_MB_LO=-1 NUM_COLS=4 SHUFFLE_LEFT_TUPLES=$(($left_total/$node)) SHUFFLE_RIGHT_TUPLES=$(($right_total/$node)) SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_RIGHT_REPEAT=1 python multiproc.py run_shuffle.py $node
#         # + chunking
#         STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_REPEATS=5 MORSEL_MB_LO=256 MORSEL_MB_HI=1024 NUM_COLS=4 SHUFFLE_LEFT_TUPLES=$(($left_total/$node)) SHUFFLE_RIGHT_TUPLES=$(($right_total/$node)) SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_RIGHT_REPEAT=1 SHUFFLE_PROBE_SCHED=STANDARD python multiproc.py run_shuffle.py $node
#         # + late probing
#         STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_REPEATS=5 MORSEL_MB_LO=256 MORSEL_MB_HI=1024 NUM_COLS=4 SHUFFLE_LEFT_TUPLES=$(($left_total/$node)) SHUFFLE_RIGHT_TUPLES=$(($right_total/$node)) SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_RIGHT_REPEAT=1 python multiproc.py run_shuffle.py $node
#     done
# done
# ----------------------------------------------------------------------------------------
# for representative bmon runs:
# STDOUT_TEMPLATE="results/shuffle_bmon_{}.txt"
# for node in $nodes; do
#     for num_cols in $num_colss; do
#         for USE_RMM in $truefalse; do
#             for USE_NET_OPT_FLAGS in $truefalse; do
#                 for SHUFFLE_PROBE_TYPE in $shuffle_probe_types; do
#                     USE_BMON=True BMON_TIME=100 BMON_FILE="results/beemon/beemon_shuffle.csv" STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=512 MORSEL_MB_HI=2048 NUM_COLS=$num_cols SHUFFLE_TYPE=BLOCKING USE_RMM=$USE_RMM USE_NET_OPT_FLAGS=$USE_NET_OPT_FLAGS SHUFFLE_PROBE_TYPE=$SHUFFLE_PROBE_TYPE python multiproc.py run_shuffle.py $node
#                     USE_BMON=True BMON_TIME=100 BMON_FILE="results/beemon/beemon_shuffle.csv" STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=-1 MORSEL_MB_HI=-1 NUM_COLS=$num_cols SHUFFLE_TYPE=BLOCKING USE_RMM=$USE_RMM USE_NET_OPT_FLAGS=$USE_NET_OPT_FLAGS SHUFFLE_PROBE_TYPE=$SHUFFLE_PROBE_TYPE python multiproc.py run_shuffle.py $node
#                 done
#             done
#         done
#     done
# done
# # # shuffle bench overlapped
# for node in $nodes; do
#     for num_cols in $num_colss; do
#         for USE_RMM in $truefalse; do
#             for USE_NET_OPT_FLAGS in $truefalse; do
#                 for SHUFFLE_PROBE_TYPE in $shuffle_probe_types; do
#                     for SHUFFLE_PROBE_SCHED in $probe_sched; do
#                         USE_BMON=True BMON_TIME=100 BMON_FILE="results/beemon/beemon_shuffle.csv" STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=512 MORSEL_MB_HI=2048 NUM_COLS=$num_cols SHUFFLE_TYPE=OVERLAPPED USE_RMM=$USE_RMM USE_NET_OPT_FLAGS=$USE_NET_OPT_FLAGS SHUFFLE_PROBE_TYPE=$SHUFFLE_PROBE_TYPE SHUFFLE_PROBE_SCHED=$SHUFFLE_PROBE_SCHED python multiproc.py run_shuffle.py $node
#                     done
#                 done
#             done
#         done
#     done
# done
# ----------------------------------------------------------------------------------------
# aggregation
# STDOUT_TEMPLATE="results/shuffle_aggregation_{}.txt"
# shuffle_agg_groupss="1 10 50 500 1000 5000"
# for node in $nodes; do
#     for num_cols in $num_colss; do
#         # overlapped, without aggregation:
#         STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=1 MORSEL_MB_HI=2048 NUM_COLS=$num_cols python multiproc.py run_shuffle.py $node
#         # blocking chunked
#         STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=1 MORSEL_MB_HI=2048 NUM_COLS=$num_cols SHUFFLE_TYPE=BLOCKING USE_RMM=False USE_NET_OPT_FLAGS=False python multiproc.py run_shuffle.py $node
#         # blocking full
#         STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=-1 NUM_COLS=$num_cols SHUFFLE_TYPE=BLOCKING USE_RMM=False USE_NET_OPT_FLAGS=False python multiproc.py run_shuffle.py $node

#         # with different number of aggregation groups:
#         for SHUFFLE_AGG_GROUPS in $shuffle_agg_groupss; do
#             # overlapped
#             STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=1 MORSEL_MB_HI=2048 NUM_COLS=$num_cols SHUFFLE_AGG_GROUPS=$SHUFFLE_AGG_GROUPS python multiproc.py run_shuffle.py $node
#             # blocking chunked
#             STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=1 MORSEL_MB_HI=2048 NUM_COLS=$num_cols SHUFFLE_TYPE=BLOCKING USE_RMM=False SHUFFLE_AGG_GROUPS=$SHUFFLE_AGG_GROUPS USE_NET_OPT_FLAGS=False python multiproc.py run_shuffle.py $node
#             # blocking full
#             STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=-1 NUM_COLS=$num_cols SHUFFLE_TYPE=BLOCKING USE_RMM=False SHUFFLE_AGG_GROUPS=$SHUFFLE_AGG_GROUPS USE_NET_OPT_FLAGS=False python multiproc.py run_shuffle.py $node
#         done
#     done
# done
# # AGG_USE_CONS_COUNT=True
# for node in $nodes; do
#     for num_cols in $num_colss; do
#         # with different number of aggregation groups:
#         for SHUFFLE_AGG_GROUPS in $shuffle_agg_groupss; do
#             # overlapped
#             AGG_USE_CONS_COUNT=True STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=1 MORSEL_MB_HI=2048 NUM_COLS=$num_cols SHUFFLE_AGG_GROUPS=$SHUFFLE_AGG_GROUPS python multiproc.py run_shuffle.py $node
#             # blocking chunked
#             AGG_USE_CONS_COUNT=True STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=1 MORSEL_MB_HI=2048 NUM_COLS=$num_cols SHUFFLE_TYPE=BLOCKING USE_RMM=False SHUFFLE_AGG_GROUPS=$SHUFFLE_AGG_GROUPS USE_NET_OPT_FLAGS=False python multiproc.py run_shuffle.py $node
#             # blocking full
#             AGG_USE_CONS_COUNT=True STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=-1 NUM_COLS=$num_cols SHUFFLE_TYPE=BLOCKING USE_RMM=False SHUFFLE_AGG_GROUPS=$SHUFFLE_AGG_GROUPS USE_NET_OPT_FLAGS=False python multiproc.py run_shuffle.py $node
#         done
#     done
# done
# ----------------------------------------------------------------------------------------
# simpler chunk sizes
STDOUT_TEMPLATE="results/shuffle_chunk_sizes_{}.txt"
SHUFFLE_LEFT_TUPLES=60000000
SHUFFLE_RIGHT_TUPLES=200000000
node=2
num_colss="6"
for num_cols in $num_colss; do
    
    NUM_REPEATS=20 STDOUT_TEMPLATE=$STDOUT_TEMPLATE MORSEL_MB_LO=3 MORSEL_MB_HI=3072 SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_RIGHT_REPEAT=1 NUM_COLS=$num_cols python multiproc.py run_shuffle.py $node
done

for num_cols in $num_colss; do
    for USE_OPT in $truefalse; do
        # blocking chunked
        # STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_REPEATS=20 MORSEL_MB_LO=3 MORSEL_MB_HI=3072 NUM_COLS=$num_cols SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_RIGHT_REPEAT=1 SHUFFLE_TYPE=BLOCKING USE_RMM=$USE_OPT USE_NET_OPT_FLAGS=$USE_OPT SHUFFLE_PROBE_TYPE=$SHUFFLE_PROBE_TYPE python multiproc.py run_shuffle.py $node

        # blocking full
        STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_REPEATS=10 SHUFFLE_LEFT_TUPLES=$SHUFFLE_LEFT_TUPLES SHUFFLE_RIGHT_TUPLES=$SHUFFLE_RIGHT_TUPLES MORSEL_MB_LO=-1 MORSEL_MB_HI=-1 NUM_COLS=$num_cols SHUFFLE_TYPE=BLOCKING SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_RIGHT_REPEAT=1 USE_OPT=$USE_RMM USE_NET_OPT_FLAGS=$USE_OPT python multiproc.py run_shuffle.py $node
        # chunked overlapped \wo deferred sync
        STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_REPEATS=10 SHUFFLE_LEFT_TUPLES=$SHUFFLE_LEFT_TUPLES SHUFFLE_RIGHT_TUPLES=$SHUFFLE_RIGHT_TUPLES MORSEL_MB_LO=3 MORSEL_MB_HI=3072 NUM_COLS=$num_cols SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_RIGHT_REPEAT=1 USE_RMM=$USE_OPT USE_NET_OPT_FLAGS=$USE_OPT SHUFFLE_PROBE_SCHED="STANDARD" python multiproc.py run_shuffle.py $node
        # chunked overlapped
        STDOUT_TEMPLATE=$STDOUT_TEMPLATE NUM_REPEATS=10 SHUFFLE_LEFT_TUPLES=$SHUFFLE_LEFT_TUPLES SHUFFLE_RIGHT_TUPLES=$SHUFFLE_RIGHT_TUPLES MORSEL_MB_LO=3 MORSEL_MB_HI=3072 NUM_COLS=$num_cols SHUFFLE_JOIN_RATIO=0.5 SHUFFLE_RIGHT_REPEAT=1 USE_RMM=$USE_OPT USE_NET_OPT_FLAGS=$USE_OPT python multiproc.py run_shuffle.py $node
    done
done

# ----------------------------------------------------------------------------------------

STDOUT_FILE=`echo "$STDOUT_TEMPLATE" | sed "s/{}/0/g"`
echo "### END OF BENCH SCRIPT" >> $STDOUT_FILE