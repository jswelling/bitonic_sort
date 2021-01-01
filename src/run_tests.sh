#!/bin/bash

# Test the compiled version
make
mpirun -n 4 ./sort_tester -c 1000000 -s 1234

# Test the python version
python setup.py build_ext --inplace
mpirun -n 4 python ./pysort_tester.py --count 1000000 --seed 1234
