import numpy as np
cimport numpy as cnp

from mpi4py.MPI cimport Comm
from mpi4py.libmpi cimport MPI_Comm, MPI_COMM_WORLD

#DTYPE = np.float64

cdef extern from *:
    struct Comm:
        pass

cdef extern from "jpsort.h":
    void jpSort(MPI_Comm comm,
                void* base,
                size_t nmemb,
                size_t size,
                int (*compar)(const void*, const void*))

cdef int compar(const void* p1, const void* p2):
  cdef double v1= (<double*>p1)[0];
  cdef double v2= (<double*>p2)[0];
  if (v1<v2):
      return -1;
  elif (v1>v2):
      return 1;
  else:
      return 0;

def pyjpSort(Comm comm,
             object array  # a numpy array- how to specify that?
             ):
    print(array)
    print(array.dtype)
    #cdef double[::1] array_view = np.ascontiguousarray(array)
    #cdef array_view = np.ascontiguousarray(array)
    cdef cnp.ndarray[double, ndim=1, mode='c'] np_buf = np.ascontiguousarray(array)
    cdef size_t nmemb = np_buf.shape[0]
    cdef size_t size = np_buf.itemsize
    cdef MPI_Comm mpi_comm = comm.ob_mpi
    return jpSort(mpi_comm, <void *>np_buf.data, nmemb, size, compar)

