#!/usr/bin/env bash

if [[ $# -ne 1 ]] ; then
    echo "Usage: <scriptname> <path_to_csv_file_to_clean>"
    exit 1
fi
# set -x

# remove the line "[Gloo] Rank 1 is connected to 1 peer ranks."
sed -i '/\[Gloo\] Rank [0-9]\+ is connected to [0-9]\+ peer ranks\./d' "$1"

# find a line that contains "algo,"
csv_header=`grep -m 1 -h "algo," "$1"`
# echo $csv_header
# remove all occurences of that line,
sed -i '/algo,/d' "$1"
# and put it in the first line once
sed -i "1i $csv_header" "$1"

# errors: remove all lines between a line containing: "Traceback (most recent call last):"
# and the next "####### started"
sed -i '/Traceback (most recent call last):/,/####### started/d' "$1"
