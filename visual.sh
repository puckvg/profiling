#!/usr/bin/env bash 

python -m cProfile -o dummy.cprof dummy.py 
pyprof2calltree -k -i dummy.cprof
