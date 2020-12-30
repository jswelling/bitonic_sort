from mpi4py import MPI
import jpsort_if
import numpy as np

arr = np.random.random(100)
print('calling')
jpsort_if.pyjpSort(MPI.COMM_WORLD, arr)
print('done')