#!/usr/bin/env python
from mpi4py import MPI
import jpsort_if
import numpy as np
import argparse
import sys

DEFAULT_COUNT = 1000000


def mpiabort_excepthook(type, value, traceback):
    """
    It is necessary to modify the exception handler to force the other MPI
    workers to exit, or an exception in one thread will cause a deadlock.
    See https://stackoverflow.com/questions/49868333/fail-fast-with-mpi4py
    """
    MPI.COMM_WORLD.Abort()
    sys.__excepthook__(type, value, traceback)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--count',
                        help=('Number of values in each local buffer'
                              ' (default {})'.format(DEFAULT_COUNT)),
                        type=int,
                        default=DEFAULT_COUNT)
    parser.add_argument('--seed', help='set random seed', type=int)
    try:
        args = parser.parse_args()
    except SystemExit as e:
        raise RuntimeError()  # which will trigger the exception handler
    return {'count': args.count, 'seed': args.seed}


is_sorted = lambda a: bool(np.all(a[:-1] <= a[1:]))  # Thanks stackoverflow!


def main():
    # Set up, parse args, distribute the results
    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank
    if rank == 0:
        args = parse_args()
    else:
        args = None
    args = comm.bcast(args)

    # Generate random test data

    if args['seed']:
        np.random.seed(seed=args['seed'] + rank)
    arr = np.random.random((args['count'],1))
    if rank == 0:
        time_start = MPI.Wtime()
    else:
        time_start = None
    jpsort_if.pyjpSort(MPI.COMM_WORLD, arr)
    if rank == 0:
        time_sort = MPI.Wtime() - time_start
        print(f'rank {rank}: total sort time: {time_sort}')
    else:
        time_sort = None

    # Verify the arrays were sorted globally
    success = is_sorted(arr)
    if rank < size - 1:
        comm.send(arr[-1], dest=rank+1)
    if rank > 0:
        low_nbr = comm.recv()
        success = success and bool(low_nbr <= arr[0])
    all_success = comm.gather(success, root=0)
    if rank == 0:
        assert all(all_success), "At least one worker contains unsorted data"
        print('done')


if __name__ == "__main__":
    sys.excepthook = mpiabort_excepthook
    main()
    sys.excepthook = sys.__excepthook__
