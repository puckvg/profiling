#!/usr/bin/env bash 
file=$1
proffile="${file%.*}.cprof"

python -m cProfile -o $proffile $file
pyprof2calltree -k -i $proffile 
