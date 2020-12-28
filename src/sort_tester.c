#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include <jpsort.h>

/* Notes-
 */

static int compar(const void* p1, const void* p2)
{
  double v1= *(double*)p1;
  double v2= *(double*)p2;
  if (v1<v2) return -1;
  else if (v1>v2) return 1;
  else return 0;
}

int main(int argc, char* argv[])
{
  int commSize;
  int myRank;
  int rc;
  int i;
  int test;
  int allTest;
  double leftMax= 0.0;
  MPI_Status status;
  double t1= 0.0, t2= 0.0;
  int     c, errflg = 0;
  double* sortMe= NULL;
  long size= 0;
  long seed= 0;
  
  MPI_Init(&argc, &argv);
  if ((rc=MPI_Comm_size(MPI_COMM_WORLD,&commSize)) != MPI_SUCCESS)
    MPI_Abort(MPI_COMM_WORLD,rc);
  if ((rc=MPI_Comm_rank(MPI_COMM_WORLD,&myRank)) != MPI_SUCCESS)
    MPI_Abort(MPI_COMM_WORLD,rc);

  optarg = NULL;
  while (!errflg && ((c = getopt(argc, argv, "c:s:")) != -1))
    switch (c) {
    case 'c'        :
      size = atoi(optarg);
      break;
    case 's'        :
      seed = atoi(optarg);
      break;
    default :
      errflg++;
    }
  if (errflg || !seed || !size) {
    if (myRank==0)
      fprintf(stderr,"usage: %s -c nValsInLocalBuf -s randSeed\n",
	      argv[0]);
    MPI_Finalize();
    exit(0);
  }

  if (!(sortMe=(double*)malloc(size*sizeof(double)))) {
    fprintf(stderr,"Rank %d: Unable to allocate %ld doubles!\n",myRank,size);
    MPI_Finalize();
    exit(0);
  }

  srand48( seed+myRank );

  for (i=0; i<size; i++) sortMe[i]= drand48();

#ifdef never
  if (myRank==0) {
    fprintf(stderr,"Unsorted:\n");
    for (i=0; i<size; i++) fprintf(stderr,"%d: %lg\n",i,sortMe[i]);
  }
#endif
  if (myRank==0) t1= MPI_Wtime();
  jpSort(MPI_COMM_WORLD, sortMe, size, sizeof(double), compar);
  if (myRank==0) t2= MPI_Wtime();
#ifdef never
  if (myRank==1) {
    fprintf(stderr,"Sorted:\n");
    for (i=0; i<size; i++) fprintf(stderr,"%d: %g\n",i,sortMe[i]);
  }
#endif

  test= 1;
  for (i=1; i<size; i++)
    if (sortMe[i]<sortMe[i-1]) {
      fprintf(stderr,"Failure on rank %d at i= %d!\n",myRank,i);
      test= 0;
      break;
    }
  /* Send global consistency check info right */
  if (myRank<(commSize-1)) {
    rc= MPI_Send(&(sortMe[size-1]),1,MPI_DOUBLE,myRank+1,0,MPI_COMM_WORLD);
  }
  if (myRank>0) {
    rc= MPI_Recv(&leftMax,1,MPI_DOUBLE,myRank-1,0,MPI_COMM_WORLD,&status);
  }
  if (leftMax>sortMe[0]) {
    fprintf(stderr,"Failure on rank %d comparing with left neighbor!\n",
	    myRank);
    test= 0;
  }

  /* Did everyone pass the test? */
  rc= MPI_Reduce(&test,&allTest,1,MPI_INT,MPI_MIN,0,MPI_COMM_WORLD);

  if (myRank==0) {
    if (allTest) fprintf(stdout,"%d %ld %ld %g\n",commSize,size,seed,t2-t1);
    else fprintf(stderr,"%d %ld %ld FAILED\n",commSize,seed,size);
  }

  MPI_Finalize();
  return 0;
}
