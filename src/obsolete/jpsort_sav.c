#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* Notes-
 * -test the SIZE==1 and SIZE==2 cases!
 * -eliminate gblRank; it's not needed.
 */

#define SIZE 113
#define SEED 123456

typedef enum { OP_SORT, OP_MERGE } Op;
typedef enum { ASCENDING, DESCENDING } Dir;

typedef struct StackElement_struct {
  int lvl;
  Op op;
  Dir dir;
  struct StackElement_struct* next;
} StackElement;

#define MPITEST(code, funcall) \
{if ((code=funcall)!=MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD,code);}

static int gblRank;

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
  MPITEST(rc, MPI_Recv(scratch, size*(nmemb/2), MPI_UNSIGNED_CHAR,
		       partnerRank, 0, comm, &status));
#ifdef never
  MPITEST(rc, MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &count));
  fprintf(stderr,"Task %d: Received %d char(s) from task %d with tag %d \n",
       myRank, count, status.MPI_SOURCE, status.MPI_TAG);
#endif

  /* Send bottom half of my buf */
  MPITEST(rc, MPI_Send(base, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR,
		       partnerRank, 0, comm));
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
  MPITEST(rc, MPI_Recv(base, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR,
		       partnerRank, 0, comm, &status));
  /* Send top half of partner buf */
  MPITEST(rc, MPI_Send(scratch, size*(nmemb/2), MPI_UNSIGNED_CHAR,
		       partnerRank, 0, comm));
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
  MPITEST(rc, MPI_Send(topHalf, size*(nmemb/2), MPI_UNSIGNED_CHAR, 
		       partnerRank, 0, comm));
  /* catch bottom half of partner buf */
  MPITEST(rc, MPI_Recv(scratch, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR, 
		       partnerRank, 0, comm, &status));
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
  MPITEST(rc, MPI_Send(scratch, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR,
		       partnerRank, 0, comm));
  /* catch top half of my buf */
  MPITEST(rc, MPI_Recv(topHalf, size*(nmemb/2), MPI_UNSIGNED_CHAR, 
		       partnerRank, 0, comm, &status));
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
  MPITEST(rc, MPI_Recv(scratch, size*(nmemb/2), MPI_UNSIGNED_CHAR,
		       partnerRank, 0, comm, &status));
  /* Send bottom half of my buf */
  MPITEST(rc, MPI_Send(base, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR,
		       partnerRank, 0, comm));
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
  MPITEST(rc, MPI_Recv(base, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR,
		       partnerRank, 0, comm, &status));
  /* Send top half of partner buf */
  MPITEST(rc, MPI_Send(scratch, size*(nmemb/2), MPI_UNSIGNED_CHAR,
		       partnerRank, 0, comm));
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
  MPITEST(rc, MPI_Send(topHalf, size*(nmemb/2), MPI_UNSIGNED_CHAR, 
		       partnerRank, 0, comm));
  /* catch bottom half of partner buf */
  MPITEST(rc, MPI_Recv(scratch, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR, 
		       partnerRank, 0, comm, &status));
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
  MPITEST(rc, MPI_Send(scratch, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR,
		       partnerRank, 0, comm));
  /* catch top half of my buf */
  MPITEST(rc, MPI_Recv(topHalf, size*(nmemb/2), MPI_UNSIGNED_CHAR, 
		       partnerRank, 0, comm, &status));
}

static int findIndexOfMax(void* base, size_t nmemb, size_t size,
			  int (*compar)(const void*, const void*))
{
  int ip0= 0;
  int ip1= nmemb/2;
  int ip2= nmemb-1;
  int done= 0;
  while (!done) {
    void* p0= (void*)((char*)base + ip0*size);
    void* p1= (void*)((char*)base + ip1*size);
    void* p2= (void*)((char*)base + ip2*size);

#ifdef never
    fprintf(stderr,"Rank %d: searching: %d %d %d -> %g %g %g\n",
	    gblRank,ip0,ip1,ip2,
	    *(double*)p0,*(double*)p1,*(double*)p2);
#endif

    if (compar(p0,p1)<=0) {
      /* p0<=p1 */
      if (compar(p1,p2)<0) {
	/* p0<=p1<=p2; p2 is local max.  Step to the right. */
	ip0= ip1;
	ip1= ip2;
	ip2= (ip2+nmemb)/2;
      }
      else {
	/* p1 is local max; shrink */
	ip0= (ip0+ip1+1)/2; /* round up, and thus inward */
	ip2= (ip1+ip2)/2; /* round down, and thus inward */
      }
    }
    else {
      /* p0>p1 */
      if (compar(p1,p2)>=0) {
	/* p0 is local max; shift left */
	ip2= ip1;
	ip1= ip0;
	ip0= (ip0+1)/2;
      }
      else {
	/* This case violates the looking-for-max constraint */
	fprintf(stderr,"rank %d: ordering constraint 1 violated!\n",
		gblRank);
	MPI_Abort(MPI_COMM_WORLD,-1);
      }
    }
    done= (ip0==ip1)&&(ip1==ip2);
  }
  return ip1;
}

static int findIndexOfMin(void* base, size_t nmemb, size_t size,
			  int (*compar)(const void*, const void*))
{
  int ip0= 0;
  int ip1= nmemb/2;
  int ip2= nmemb-1;
  int done= 0;
  while (!done) {
    void* p0= (void*)((char*)base + ip0*size);
    void* p1= (void*)((char*)base + ip1*size);
    void* p2= (void*)((char*)base + ip2*size);

#ifdef never
    fprintf(stderr,"Rank %d: searching: %d %d %d -> %g %g %g\n",
	    gblRank,ip0,ip1,ip2,
	    *(double*)p0,*(double*)p1,*(double*)p2);
#endif
    if (compar(p0,p1)>=0) {
      /* p0>=p1 */
      if (compar(p1,p2)>0) {
	/* p0>=p1>=p2; p2 is local min.  Step to the right. */
	ip0= ip1;
	ip1= ip2;
	ip2= (ip2+nmemb)/2;
      }
      else {
	/* p1 is local min; shrink */
	ip0= (ip0+ip1+1)/2; /* round up, and thus inward */
	ip2= (ip1+ip2)/2; /* round down, and thus inward */
      }
    }
    else {
      /* p0<p1 */
      if (compar(p1,p2)<=0) {
	/* p0 is local min; shift left */
	ip2= ip1;
	ip1= ip0;
	ip0= (ip0+1)/2;
      }
      else {
	/* This case violates the looking-for-max constraint */
	fprintf(stderr,"rank %d: ordering constraint 1 violated!\n",
		gblRank);
	MPI_Abort(MPI_COMM_WORLD,-1);
      }
    }
    done= (ip0==ip1)&&(ip1==ip2);
  }
  return ip1;
}

#define COPYBLK(i,j) \
{ bcopy((void*)((char*)base+(i)*size),(void*)((char*)scratch+(j)*size),size); }

static void bitonicMergeSortAscending(void* base, size_t nmemb, size_t size,
				      int (*compar)(const void*, const void*),
				      void* scratch, void* tmp)
{
  void* next= (void*)(((char*)base)+size);
  void* last= (void*)(((char*)base)+(nmemb-1)*size);
  fprintf(stderr,"rank %d bitonicMergeSortAscending\n",gblRank);
  if (nmemb==1) return;
  else if (nmemb==2) {
    if (compar(base, next)<0) {
      bcopy(base,tmp,size);
      bcopy(next,base,size);
      bcopy(tmp,next,size);
    }
    return;
  }
  else {
    /* We will arbitrarily classify the monotonic-up case as bitonic-up
     * and rely on the merge sort to deal with it.
     */

    if (compar(base,next)<0) {
      /* bitonic-up */
      /* Work outwards from middle, which is high */
      int max= findIndexOfMax(base, nmemb, size, compar);
      int low= max-1;
      int hi= max+1;
      int offset= nmemb-1;

      fprintf(stderr, "rank %d merge clause 1, max= %d\n",gblRank,max);
      COPYBLK(max, offset--);
      while (low>=0 && hi<nmemb) {
	if (compar((void*)((char*)base+low*size),
		   (void*)((char*)base+hi*size))<=0) {
	  COPYBLK(hi++, offset--);
	}
	else {
	  COPYBLK(low--, offset--);
	}
      }
      while (low>=0) COPYBLK(low--, offset--);
      while (hi<nmemb) COPYBLK(hi++, offset--);
    }
    else {
      /* bitonic-down */
      /* Work outwards from the middle */
      int min= findIndexOfMin(base, nmemb, size, compar);
      int low= min-1;
      int hi= min+1;
      int offset= 0;

      fprintf(stderr, "rank %d merge clause 2, min= %d\n",gblRank,min);
      COPYBLK(min, offset++);
      while (low>=0 && hi<nmemb) {
	if (compar((void*)((char*)base+low*size),
		   (void*)((char*)base+hi*size))<=0) {
	  COPYBLK(low--, offset++);
	}
	else {
	  COPYBLK(hi++, offset++);
	}
      }
      while (low>=0) COPYBLK(low--, offset++);
      while (hi<nmemb) COPYBLK(hi++, offset++);
    }

  }

  bcopy(scratch, base, size*nmemb);
}

static void bitonicMergeSortDescending(void* base, size_t nmemb, size_t size,
				       int (*compar)(const void*, const void*),
				       void* scratch, void* tmp)
{
  _current_compar= compar;
  qsort(base, nmemb, size, _reverse_compar);
  _current_compar= NULL;
}

#undef COPYBLK

void jpSort(MPI_Comm comm,
	    void* base,
	    size_t nmemb,
	    size_t size,
	    int (*compar)(const void*, const void*)) 
{
  int rc;
  int commSize;
  int myRank;
  int nLevels;
  int nGhosts;
  int i;
  int firstSort= 1;
  StackElement* stack= NULL;
  void* scratch= NULL;
  void* tmp= NULL;

  if (!(scratch=(void*)malloc(nmemb*size))) {
    fprintf(stderr,"unable to allocate %d bytes!\n",((nmemb+1)/2)*size);
    exit(-1);
  }
  if (!(tmp=(void*)malloc(size))) {
    fprintf(stderr,"unable to allocate %d bytes!\n",size);
    exit(-1);
  }

  MPITEST(rc, MPI_Comm_size(comm,&commSize));
  MPITEST(rc, MPI_Comm_rank(comm,&myRank));

  nLevels= 1;
  while (nLevels<commSize) nLevels *= 2;
  nGhosts= nLevels - commSize;

  se_push(nLevels, OP_SORT, ASCENDING, &stack);
  while (stack) {
    int lvl= stack->lvl;
    Op op= stack->op;
    Dir dir= stack->dir;
    Dir otherDir= (dir==ASCENDING)?DESCENDING:ASCENDING;
    int myGhostedRank= myRank+nGhosts;
    int group= myGhostedRank/lvl;
    int placeInGroup= myGhostedRank-(group*lvl);
    int partnerGhostedRank= (group*lvl) + ((myGhostedRank+lvl/2)%lvl);
    int partnerRank= partnerGhostedRank-nGhosts;
    int uphillFlag= (partnerGhostedRank<myGhostedRank);
    se_pop(&stack);
#ifdef never
    fprintf(stderr,"rank %d plus %d ghosts: lvl %d, group is %d, place %d, partnerGhosted %d, uphill %d\n",
	    myRank,nGhosts,lvl,group,placeInGroup,
	    partnerGhostedRank,uphillFlag);
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
	if (firstSort) {
	  /* A full sort of this random sequence */
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
	  firstSort= 0;
	}
	else {
	  /* Sequence is bitonic.  Use a merge sort, which is much faster. */
	  gblRank= myRank;
	  if (dir==ASCENDING)
	    bitonicMergeSortAscending(base, nmemb, size, compar,
				      scratch, tmp);
	  else
	    bitonicMergeSortDescending(base, nmemb, size, compar, 
				       scratch, tmp);
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
	      "Rank %d plus %d ghosts; merge %s at level %d: partner rank is %d; I am %s\n",
	      myRank,nGhosts,(dir==ASCENDING)?"ascending":"descending",
	      lvl,partnerRank,(uphillFlag ? "uphill":"downhill"));
#endif
      /* If the partner is a ghost, we just don't merge! */
      if (partnerRank>=0) {
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
  int rc;
  double sortMe[SIZE];
  int i;
  int test;
  double leftMax= 0.0;
  MPI_Status status;
  
  
  MPI_Init(&argc, &argv);
  if ((rc=MPI_Comm_size(MPI_COMM_WORLD,&commSize)) != MPI_SUCCESS)
    MPI_Abort(MPI_COMM_WORLD,rc);
  if ((rc=MPI_Comm_rank(MPI_COMM_WORLD,&myRank)) != MPI_SUCCESS)
    MPI_Abort(MPI_COMM_WORLD,rc);

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
  /* Send global consistency check info right */
  if (myRank<(commSize-1)) {
    rc= MPI_Send(&(sortMe[SIZE-1]),1,MPI_DOUBLE,myRank+1,0,MPI_COMM_WORLD);
  }
  if (myRank>0) {
    rc= MPI_Recv(&leftMax,1,MPI_DOUBLE,myRank-1,0,MPI_COMM_WORLD,&status);
  }
  if (leftMax>sortMe[0]) {
    fprintf(stderr,"Failure on rank %d comparing with left neighbor!\n",
	    myRank);
    test= 0;
  }
  if (test) fprintf(stderr,"rank %d OK, %g to %g; left neighbor %g\n",
		    myRank,sortMe[0],sortMe[i-1],leftMax);

  MPI_Finalize();
  return 0;
}
