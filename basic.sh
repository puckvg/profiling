#!/usr/bin/env bash
file=$1

python -m cProfile -s tottime $file
