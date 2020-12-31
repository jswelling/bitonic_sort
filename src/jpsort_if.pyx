import numpy as np
cimport numpy as cnp
from mpi4py.MPI cimport Comm
from mpi4py.libmpi cimport MPI_Comm, MPI_COMM_WORLD

"""
This module docstring does not seem to end up in the python object
"""

cdef extern from *:
    struct Comm:
        pass

cdef extern from "jpsort.h":
    void jpSort(MPI_Comm comm,
                void* base,
                size_t nmemb,
                size_t size,
                int (*compar)(const void*, const void*))

cdef int compare_float64(const void* p1, const void* p2):
  cdef cnp.npy_float64 v1= (<cnp.npy_float64*>p1)[0];
  cdef cnp.npy_float64 v2= (<cnp.npy_float64*>p2)[0];
  if (v1<v2):
      return -1;
  elif (v1>v2):
      return 1;
  else:
      return 0;

def pyjpSort(Comm comm,
             cnp.ndarray array
             ):
    """
    Will this come out in the docstring?
    """
    assert array.flags['C_CONTIGUOUS'], 'Array must be contiguous and in C order'
    assert array.flags['WRITEABLE'], 'Array is not writeable and so cannot be sorted'
    assert array.dtype == np.float64, 'Only float64 data is supported so far'
    cdef size_t nmemb = array.shape[0]
    cdef size_t size = array.itemsize
    cdef MPI_Comm mpi_comm = comm.ob_mpi
    return jpSort(mpi_comm, <void *>array.data, nmemb, size, compare_float64)

