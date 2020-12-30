#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SIZE 1000000
#define SEED 123456

typedef enum { OP_SORT, OP_MERGE } Op;
typedef enum { ASCENDING, DESCENDING } Dir;

typedef struct StackElement_struct {
  int lvl;
  Op op;
  Dir dir;
  struct StackElement_struct* next;
} StackElement;

static int (*_current_compar)(const void*, const void*)= NULL;

static int _reverse_compar(const void* p1, const void* p2)
{
  return _current_compar(p2,p1);
}

static void se_push( int lvl, Op op, Dir dir, StackElement** stack )
{
  StackElement* newSE= NULL;
  if (!(newSE=(StackElement*)malloc(sizeof(StackElement)))) {
    fprintf(stderr,"Unable to allocate %d bytes!\n",sizeof(StackElement));
    exit(-1);
  }
  newSE->lvl= lvl;
  newSE->op= op;
  newSE->dir= dir;
  newSE->next= *stack;
  *stack= newSE;
}

static void se_pop( StackElement** stack )
{
  StackElement* target= *stack;
  if (target) {
    *stack= target->next;
    free(target);
  }
}

/*
  -Case of commSize=1: just qsort once ascending.
  -Case of commSize=2:
  1) qsort rank 0 ascending; qsort rank 1 descending
  2) merge( rank1, rank2 ) ascending
  3) qsort rank1, rank2 ascending
  4) done
  -Case of commSize=4:
  1) qsort ranks 0,2 ascending; qsort rank 1,3 descending
  2) merge( rank0, rank1 ) ascending; merge( rank2, rank3 ) ascending
  3) qsort rank0, rank1 ascending; qsort rank2, rank3 descending
  4) merge( rank0, rank2 ); merge( rank1, rank3 )
  5) qsort ranks 0,2 ascending; qsort ranks 1,3 descending
  6) merge( rank0,rank1); merge( rank2, rank3)
  7) qsort 0,1,2,3 ascending
  
*/

static void merge_ascending_uphill(MPI_Comm comm, void* base, 
				   size_t nmemb,size_t size,
				   int (*compar)(const void*, const void*),
				   int myRank, int partnerRank,
				   void* scratch, void* tmp)
{
  int rc;
  MPI_Status status;
  int i;
  void* topHalf= (void*)((char*)base + ((nmemb+1)/2)*size);
  int count;

  /* I will shuffle the top half of the buffer */
  /* Catch top half of partner buf */
  rc= MPI_Recv(scratch, size*(nmemb/2), MPI_UNSIGNED_CHAR,
	       partnerRank, 0, comm, &status);
  rc = MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &count);
#ifdef never
  fprintf(stderr,"Task %d: Received %d char(s) from task %d with tag %d \n",
       myRank, count, status.MPI_SOURCE, status.MPI_TAG);
#endif

  /* Send bottom half of my buf */
  rc= MPI_Send(base, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR,
	       partnerRank, 0, comm);
  /* Shuffle top half of buf, ascending from above */
  for (i=0; i<nmemb/2; i++) {
    void* p1= (void*)((char*)topHalf + i*size);
    void* p2= (void*)((char*)scratch + i*size);
    if (compar(p1,p2)<0) {
      bcopy(p1,tmp,size);
      bcopy(p2,p1,size);
      bcopy(tmp,p2,size);
    }
  }
  /* Catch bottom half of my buf */
  rc= MPI_Recv(base, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR,
	       partnerRank, 0, comm, &status);
  /* Send top half of partner buf */
  rc= MPI_Send(scratch, size*(nmemb/2), MPI_UNSIGNED_CHAR,
	       partnerRank, 0, comm);
}

static void merge_ascending_downhill(MPI_Comm comm, void* base, 
				     size_t nmemb,size_t size,
				     int (*compar)(const void*, const void*),
				     int myRank, int partnerRank,
				     void* scratch, void* tmp)
{
  int rc;
  MPI_Status status;
  int i;
  void* topHalf= (void*)((char*)base + ((nmemb+1)/2)*size);

  /* I will shuffle the bottom half of the buffer */
  /* Send top half of my buf */
  rc= MPI_Send(topHalf, size*(nmemb/2), MPI_UNSIGNED_CHAR, 
	       partnerRank, 0, comm);
  /* catch bottom half of partner buf */
  rc= MPI_Recv(scratch, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR, 
	       partnerRank, 0, comm, &status);
  /* Shuffle bottom half of buf, ascending from below */
  for (i=0; i<(nmemb+1)/2; i++) {
    void* p1= (void*)((char*)scratch + i*size);
    void* p2= (void*)((char*)base + i*size);
    if (compar(p1,p2)<0) {
      bcopy(p1,tmp,size);
      bcopy(p2,p1,size);
      bcopy(tmp,p2,size);
    }
  }
  /* send bottom half of partner buf */
  rc= MPI_Send(scratch, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR,
	       partnerRank, 0, comm);
  /* catch top half of my buf */
  rc= MPI_Recv(topHalf, size*(nmemb/2), MPI_UNSIGNED_CHAR, 
	       partnerRank, 0, comm, &status);
}

static void merge_descending_uphill(MPI_Comm comm, void* base, 
				    size_t nmemb,size_t size,
				    int (*compar)(const void*, const void*),
				    int myRank, int partnerRank, 
				    void* scratch, void* tmp)
{
  int rc;
  MPI_Status status;
  int i;
  void* topHalf= (void*)((char*)base + ((nmemb+1)/2)*size);

  /* I will shuffle the top half of the buffer */
  /* Catch top half of partner buf */
  rc= MPI_Recv(scratch, size*(nmemb/2), MPI_UNSIGNED_CHAR,
	       partnerRank, 0, comm, &status);
  /* Send bottom half of my buf */
  rc= MPI_Send(base, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR,
	       partnerRank, 0, comm);
  /* Shuffle top half of buf, descending from above */
  for (i=0; i<nmemb/2; i++) {
    void* p1= (void*)((char*)topHalf + i*size);
    void* p2= (void*)((char*)scratch + i*size);
    if (compar(p1,p2)>0) {
      bcopy(p1,tmp,size);
      bcopy(p2,p1,size);
      bcopy(tmp,p2,size);
    }
  }
  /* Catch bottom half of my buf */
  rc= MPI_Recv(base, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR,
	       partnerRank, 0, comm, &status);
  /* Send top half of partner buf */
  rc= MPI_Send(scratch, size*(nmemb/2), MPI_UNSIGNED_CHAR,
	       partnerRank, 0, comm);
}

static void merge_descending_downhill(MPI_Comm comm, void* base, 
				      size_t nmemb,size_t size,
				      int (*compar)(const void*, const void*),
				      int myRank, int partnerRank,
				      void* scratch, void* tmp)
{
  int rc;
  MPI_Status status;
  int i;
  void* topHalf= (void*)((char*)base + ((nmemb+1)/2)*size);

  /* I will shuffle the bottom half of the buffer */
  /* Send top half of my buf */
  rc= MPI_Send(topHalf, size*(nmemb/2), MPI_UNSIGNED_CHAR, 
	       partnerRank, 0, comm);
  /* catch bottom half of partner buf */
  rc= MPI_Recv(scratch, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR, 
	       partnerRank, 0, comm, &status);
  /* Shuffle bottom half of buf, descending from below */
  for (i=0; i<(nmemb+1)/2; i++) {
    void* p1= (void*)((char*)scratch + i*size);
    void* p2= (void*)((char*)base + i*size);
    if (compar(p1,p2)>0) {
      bcopy(p1,tmp,size);
      bcopy(p2,p1,size);
      bcopy(tmp,p2,size);
    }
  }
  /* send bottom half of partner buf */
  rc= MPI_Send(scratch, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR,
	       partnerRank, 0, comm);
  /* catch top half of my buf */
  rc= MPI_Recv(topHalf, size*(nmemb/2), MPI_UNSIGNED_CHAR, 
	       partnerRank, 0, comm, &status);
}


void jpSort(MPI_Comm comm,
	    void* base,
	    size_t nmemb,
	    size_t size,
	    int (*compar)(const void*, const void*)) 
{
  int err;
  int commSize;
  int myRank;
  int nLevels;
  int i;
  StackElement* stack= NULL;
  void* scratch= NULL;
  void* tmp= NULL;

  if (!(scratch=(void*)malloc(((nmemb+1)/2)*size))) {
    fprintf(stderr,"unable to allocate %d bytes!\n",((nmemb+1)/2)*size);
    exit(-1);
  }
  if (!(tmp=(void*)malloc(size))) {
    fprintf(stderr,"unable to allocate %d bytes!\n",size);
    exit(-1);
  }

  if ((err=MPI_Comm_size(comm,&commSize)) != MPI_SUCCESS)
    MPI_Abort(comm,err);
  if ((err=MPI_Comm_rank(comm,&myRank)) != MPI_SUCCESS)
    MPI_Abort(comm,err);

  nLevels= 1;
  while (nLevels<commSize) nLevels *= 2;
  if (nLevels>commSize) {
    fprintf(stderr,"Comm size is not a power of two!\n");
    exit(-1);
  }

  se_push(nLevels, OP_SORT, ASCENDING, &stack);
  while (stack) {
    int lvl= stack->lvl;
    Op op= stack->op;
    Dir dir= stack->dir;
    Dir otherDir= (dir==ASCENDING)?DESCENDING:ASCENDING;
    int group= myRank/lvl;
    int placeInGroup= myRank-(group*lvl);
    int partnerRank= (group*lvl) + ((myRank+lvl/2)%lvl);
    int uphillFlag= (partnerRank<myRank);
    se_pop(&stack);
#ifdef never
    switch (op) {
    case OP_SORT: 
      fprintf(stderr,"rank %d: sort %d %s\n",myRank,lvl,
	      ((dir==ASCENDING)?"ascending":"descending"));
      break;
    case OP_MERGE:
      fprintf(stderr,"rank %d: merge %d %s\n",myRank,lvl,
	      ((dir==ASCENDING)?"ascending":"descending"));
      break;
    }
#endif
    switch (op) {
    case OP_SORT: 
      if (lvl==1) {
	if (dir==ASCENDING) {
	  /* sort in ascending order */
	  qsort(base, nmemb, size, compar);
	}
	else {
	  /* sort in descending order.  I hate this part; qsort is not
	   * reentrant so we cannot do this in a reentrant fashion.
	   */
	  _current_compar= compar;
	  qsort(base, nmemb, size, _reverse_compar);
	  _current_compar= NULL;
	}
      }
      else {
	se_push(lvl/2,OP_SORT,dir,&stack);
	se_push(lvl,OP_MERGE,dir,&stack);
	se_push(lvl/2,OP_SORT,(uphillFlag ? otherDir : dir),&stack);
      }
      break;
    case OP_MERGE:
#ifdef never
      fprintf(stderr,
	      "Rank %d merge %s at level %d: partner rank is %d; I am %s\n",
	      myRank,(dir==ASCENDING)?"ascending":"descending",
	      lvl,partnerRank,(uphillFlag ? "uphill":"downhill"));
#endif
      if (dir==ASCENDING) {
	if (uphillFlag) {
	  merge_ascending_uphill(comm,base,nmemb,size,compar,
				 myRank,partnerRank,scratch,tmp);
	}
	else {
	  merge_ascending_downhill(comm,base,nmemb,size,compar,
				   myRank,partnerRank,scratch,tmp);
	}
      }
      else {
	if (uphillFlag) {
	  merge_descending_uphill(comm,base,nmemb,size,compar,
				  myRank,partnerRank,scratch,tmp);
	}
	else {
	  merge_descending_downhill(comm,base,nmemb,size,compar,
				    myRank,partnerRank,scratch,tmp);
	}
      }
      break;
    }
  }

  free(scratch);
  free(tmp);
}

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
  int err;
  double sortMe[SIZE];
  int i;
  int test;
  
  MPI_Init(&argc, &argv);
  if ((err=MPI_Comm_size(MPI_COMM_WORLD,&commSize)) != MPI_SUCCESS)
    MPI_Abort(MPI_COMM_WORLD,err);
  if ((err=MPI_Comm_rank(MPI_COMM_WORLD,&myRank)) != MPI_SUCCESS)
    MPI_Abort(MPI_COMM_WORLD,err);

  srand48( SEED+myRank );

  for (i=0; i<SIZE; i++) sortMe[i]= drand48();

#ifdef never
  if (myRank==0) {
    fprintf(stderr,"Unsorted:\n");
    for (i=0; i<SIZE; i++) fprintf(stderr,"%d: %g\n",i,sortMe[i]);
  }
#endif
  jpSort(MPI_COMM_WORLD, sortMe, SIZE, sizeof(double), compar);
#ifdef never
  if (myRank==3) {
    fprintf(stderr,"Sorted:\n");
    for (i=0; i<SIZE; i++) fprintf(stderr,"%d: %g\n",i,sortMe[i]);
  }
#endif

  test= 1;
  for (i=1; i<SIZE; i++)
    if (sortMe[i]<sortMe[i-1]) {
      fprintf(stderr,"Failure on rank %d at i= %d!\n",myRank,i);
      test= 0;
    }
  if (test) fprintf(stderr,"rank %d OK, %g to %g\n",
		    myRank,sortMe[0],sortMe[i-1]);

  MPI_Finalize();
  return 0;
}
