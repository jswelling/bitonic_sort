#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <mpi.h>

/*
*/

typedef enum { OP_SORT, OP_MERGE } Op;
typedef enum { ASCENDING, DESCENDING } Dir;

typedef struct StackElement_struct {
  int partner;
  Op op;
  Dir dir;
  struct StackElement_struct* next;
} StackElement;

#define DEBUG 1
#define DEBUG_RANK_EXPR (_gblRank==0)
#define STATS 1
#define USE_MERGESORT 1

#if DEBUG
int _gblRank;
#endif

#if STATS
static int nSwaps= 0;
static double swapTime= 0.0;
static int nSwapMsgs= 0;
static double swapMsgTime= 0.0;
static int nQSorts= 0;
static double qsortTime= 0.0;
static int nMergeSorts= 0;
static double mergesortTime= 0.0;
static int nDummy= 0;
static double dummyTime= 0.0;
static double baseTime= 0.0;
#define TIMECOUNT( counter, bin ) \
  { double t=MPI_Wtime(); counter++; bin += t-baseTime; baseTime= t; }
#else
#define TIMECOUNT( counter, bin ) /* nothing */
#endif

#define MPITEST(code, funcall) \
{if ((code=funcall)!=MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD,code);}

static int (*_current_compar)(const void*, const void*)= NULL;

static int _reverse_compar(const void* p1, const void* p2)
{
  return _current_compar(p2,p1);
}

static void se_push( int partner, Op op, Dir dir, StackElement** stack )
{
  StackElement* newSE= NULL;
  if (!(newSE=(StackElement*)malloc(sizeof(StackElement)))) {
    fprintf(stderr,"Unable to allocate %ld bytes!\n",sizeof(StackElement));
    MPI_Abort(MPI_COMM_WORLD,-1);
  }
  newSE->partner= partner;
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

#if DEBUG
static void bitonicTest(void* base, size_t nmemb, size_t size,
			int (*compar)(const void*, const void*))
{
  int nChanges= 0;
  int prev= 0;
  int first= 1;
  int dir= 0;
  int i;
  void* v1;
  void* v2;
  for (i=1; i<nmemb; i++) {
    v1= (void*)((char*)base + (i-1)*size);
    v2= (void*)((char*)base + i*size);
    int result= compar(v1,v2);
    if (first) {
      if (result != 0) {
	first= 0;
	dir= result;
      }
      prev= result;
    }
    else {
      if (prev*result<0) nChanges++;
    }
    prev= result;
  }
  if (nChanges>2) fprintf(stderr,"rank %d: not bitonic! nChanges= %d\n",
			  _gblRank, nChanges);
}
#endif

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
  TIMECOUNT(nDummy, dummyTime);
  MPITEST(rc, MPI_Recv(scratch, size*(nmemb/2), MPI_UNSIGNED_CHAR,
		       partnerRank, 0, comm, &status));
  /* Send bottom half of my buf */
  MPITEST(rc, MPI_Send(base, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR,
		       partnerRank, 0, comm));
  TIMECOUNT(nSwapMsgs, swapMsgTime);
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
  TIMECOUNT(nSwaps, swapTime);
  /* Catch bottom half of my buf */
  MPITEST(rc, MPI_Recv(base, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR,
		       partnerRank, 0, comm, &status));
  /* Send top half of partner buf */
  MPITEST(rc, MPI_Send(scratch, size*(nmemb/2), MPI_UNSIGNED_CHAR,
		       partnerRank, 0, comm));
  TIMECOUNT(nSwapMsgs, swapMsgTime);
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
  TIMECOUNT(nDummy, dummyTime);
  MPITEST(rc, MPI_Send(topHalf, size*(nmemb/2), MPI_UNSIGNED_CHAR, 
		       partnerRank, 0, comm));
  /* catch bottom half of partner buf */
  MPITEST(rc, MPI_Recv(scratch, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR, 
		       partnerRank, 0, comm, &status));
  TIMECOUNT(nSwapMsgs, swapMsgTime);
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
  TIMECOUNT(nSwaps, swapTime);
  /* send bottom half of partner buf */
  MPITEST(rc, MPI_Send(scratch, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR,
		       partnerRank, 0, comm));
  /* catch top half of my buf */
  MPITEST(rc, MPI_Recv(topHalf, size*(nmemb/2), MPI_UNSIGNED_CHAR, 
		       partnerRank, 0, comm, &status));
  TIMECOUNT(nSwapMsgs, swapMsgTime);
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
  TIMECOUNT(nDummy, dummyTime);
  MPITEST(rc, MPI_Recv(scratch, size*(nmemb/2), MPI_UNSIGNED_CHAR,
		       partnerRank, 0, comm, &status));
  /* Send bottom half of my buf */
  MPITEST(rc, MPI_Send(base, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR,
		       partnerRank, 0, comm));
  TIMECOUNT(nSwapMsgs, swapMsgTime);
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
  TIMECOUNT(nSwaps, swapTime);
  /* Catch bottom half of my buf */
  MPITEST(rc, MPI_Recv(base, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR,
		       partnerRank, 0, comm, &status));
  /* Send top half of partner buf */
  MPITEST(rc, MPI_Send(scratch, size*(nmemb/2), MPI_UNSIGNED_CHAR,
		       partnerRank, 0, comm));
  TIMECOUNT(nSwapMsgs, swapMsgTime);
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
  TIMECOUNT(nDummy, dummyTime);
  MPITEST(rc, MPI_Send(topHalf, size*(nmemb/2), MPI_UNSIGNED_CHAR, 
		       partnerRank, 0, comm));
  /* catch bottom half of partner buf */
  MPITEST(rc, MPI_Recv(scratch, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR, 
		       partnerRank, 0, comm, &status));
  TIMECOUNT(nSwapMsgs, swapMsgTime);
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
  TIMECOUNT(nSwaps, swapTime);
  /* send bottom half of partner buf */
  MPITEST(rc, MPI_Send(scratch, size*((nmemb+1)/2), MPI_UNSIGNED_CHAR,
		       partnerRank, 0, comm));
  /* catch top half of my buf */
  MPITEST(rc, MPI_Recv(topHalf, size*(nmemb/2), MPI_UNSIGNED_CHAR, 
		       partnerRank, 0, comm, &status));
  TIMECOUNT(nSwapMsgs, swapMsgTime);
}

static void findExtrema(void* base, size_t nmemb, size_t size,
			int (*compar)(const void*, const void*),
			long* min, long* max) 
{
  int c;
  int changes= 0;
  int increasing= 0; /* reset below */
  void* p1;
  void* p2;
  void* end= (void*)((char*)base+(nmemb-1)*size);
  
  /* Check bitonic, finding max and min as we go */
  p1= base;
  do {
    p2= (void*)((char*)p1 + size);
    c= compar(p1,p2);
    if (c<0) {
      if (p1==base) {
	increasing= 1;
      }
      else {
	if (increasing) {
	  /* do nothing */
	}
	else {
	  *min= ((char*)p1 - (char*)base)/size;
	  increasing= 1;
	  changes++;
	  if (changes>2) {
	    fprintf(stderr,"Internal error; sequence not bitonic!\n");
	    MPI_Abort(MPI_COMM_WORLD,-1);
	  }
	}
      }
    }
    else if (c>0) {
      if (p1==base) {
	increasing= 0;
      }
      else {	
	if (!increasing) {
	  /* do nothing */
	}
	else {
	  *max= ((char*)p1 - (char*)base)/size;
	  increasing= 0;
	  changes++;
	  if (changes>2) {
	    fprintf(stderr,"Internal error; sequence not bitonic!\n");
	    MPI_Abort(MPI_COMM_WORLD,-1);
	  }
	}
      }
    }
    else {
      /* vals same- do nothing */
    }
    p1= p2;
  } while (p2<end);
  /* Wrap around */
  p2= base;
  c= compar(p1,p2);
  if (c<0) {
    if (increasing) {
      /* do nothing */
    }
    else {
      *min= ((char*)p1 - (char*)base)/size;
      changes++;
      if (changes>2) {
	fprintf(stderr,"Internal error; sequence not bitonic!\n");
	MPI_Abort(MPI_COMM_WORLD,-1);
      }
    }
  }
  else if (c>0) {
    if (!increasing) {
      /* do nothing */
    }
    else {
      *max= ((char*)p1 - (char*)base)/size;
      changes++;
      if (changes>2) {
	fprintf(stderr,"Internal error; sequence not bitonic!\n");
	MPI_Abort(MPI_COMM_WORLD,-1);
      }
    }
  }
  else {
    /* vals equal- do nothing */
  }
}

#define COPYBLK(i,j) \
{ bcopy((void*)((char*)base+(i)*size),(void*)((char*)scratch+(j)*size),size); }

static void bitonicMergeSortAscending(void* base, size_t nmemb, size_t size,
				      int (*compar)(const void*, const void*),
				      void* scratch, void* tmp)
{
  if (nmemb==1) return;
  else {
    long max= 0;
    long min= 0;
    long i= 0;
    long low;
    long hi;

    findExtrema(base, nmemb, size, compar, &min, &max);
#if 0
    if (DEBUG_RANK_EXPR) {
      fprintf(stderr,"Rank %d: min at %ld, max at %ld, sort ascending\n",
	      _gblRank,min,max);
    }
#endif

    COPYBLK(min,i++);
    low= min-1;
    if (low<0) low += nmemb;
    hi= (min+1)%nmemb;
    while (low!=max && hi!=max) {
      if (compar((void*)((char*)base+(low*size)),
		 (void*)((char*)base+(hi*size)))<=0) {
	COPYBLK(low,i++);
	low= low-1;
	if (low<0) low += nmemb;
      }
      else {
	COPYBLK(hi,i++);
	hi= (hi+1)%nmemb;
      }
    }
    while (low!=max) {
      COPYBLK(low,i++);
      low--;
      if (low>=0) low %= nmemb;
      else low += nmemb;
    }
    while (hi!=max) {
      COPYBLK(hi,i++);
      hi= (hi+1)%nmemb;
    }
    COPYBLK(max,i);

    bcopy(scratch, base, size*nmemb);
  }

}

static void bitonicMergeSortDescending(void* base, size_t nmemb, size_t size,
				       int (*compar)(const void*, const void*),
				       void* scratch, void* tmp)
{
  if (nmemb==1) return;
  else {
    long max= 0;;
    long min= 0;
    long i= 0;
    long low;
    long hi;

    findExtrema(base, nmemb, size, compar, &min, &max);
#if 0
    if (DEBUG_RANK_EXPR) {
      fprintf(stderr,"Rank %d: min at %ld, max at %ld; sort descending\n",
	      _gblRank,min,max);
    }
#endif

    COPYBLK(max,i++);
    low= max-1;
    if (low<0) low += nmemb;
    hi= (max+1)%nmemb;
    while (low!=min && hi!=min) {
      if (compar((void*)((char*)base+(low*size)),
		 (void*)((char*)base+(hi*size)))<=0) {
	COPYBLK(hi,i++);
	hi= (hi+1)%nmemb;
      }
      else {
	COPYBLK(low,i++);
	low= low-1;
	if (low<0) low += nmemb;
      }
    }
    while (low!=min) {
      COPYBLK(low,i++);
      low--;
      if (low>=0) low %= nmemb;
      else low += nmemb;
    }
    while (hi!=min) {
      COPYBLK(hi,i++);
      hi= (hi+1)%nmemb;
    }
    COPYBLK(min,i);

    bcopy(scratch, base, size*nmemb);
  }

}

#undef COPYBLK

static void generateLocalSort( int myRank, int nGhosts, int commSize,
			       Dir dir, StackElement** stack )
{
  se_push( myRank, OP_SORT, dir, stack );
}

static void generateBitonicMerge( int myRank, int nGhosts, int commSize,
				  int place, int field, int fieldBase,
				  Dir dir, StackElement** stack )
{
  if (place>=field/2) { 
    se_push( (place-(field/2))+(fieldBase-nGhosts), OP_MERGE, dir, stack );
    if (field/2 > 1)
      generateBitonicMerge( myRank, nGhosts, commSize,
			    place-(field/2), field/2, fieldBase+(field/2),
			    dir, stack );
  }
  else {
    se_push( (place+(field/2))+(fieldBase-nGhosts), OP_MERGE, dir, stack );
    if (field/2 > 1)
      generateBitonicMerge( myRank, nGhosts, commSize,
			    place, field/2, fieldBase,
			    dir, stack );
  }
}

static void generateBitonicSortPattern( int myRank, int nGhosts, int commSize,
					int place, int field, int fieldBase,
					Dir dir, StackElement** stack )
{
  Dir otherDir= (dir==ASCENDING)?DESCENDING:ASCENDING;
  if (field>1) {
    if (place<field/2) {
      generateBitonicSortPattern( myRank, nGhosts, commSize,
				  place, field/2, fieldBase, dir, stack );
    }
    else {
      generateBitonicSortPattern( myRank, nGhosts, commSize,
				  place-(field/2), field/2, 
				  fieldBase+(field/2),
				  otherDir, stack );
    }
    generateBitonicMerge( myRank, nGhosts, commSize,
			  place, field, fieldBase, dir, stack );
  }
  generateLocalSort( myRank, nGhosts, commSize, dir, stack );
}

static void generateFullPattern(int commSize, int nLevels, int nGhosts, 
				int myRank, StackElement** pRealStack)
{
  int myGhostedRank= myRank+nGhosts;
  StackElement* stack= NULL;
  int i;
  int groupSize;

  /* This routine generates a full Bitonic sort pattern (ascending
   * order).
   */
  /* It's easier to generate these in order of occurence and then
   * reverse the order than to generate them in reverse, as the final
   * stack order requires.
   */

  /* All the clever merging is only necessary if there is more than one
   * participant.
   */
  if (commSize>1) {

    if (myRank+nGhosts < (commSize+nGhosts)/2) {
      generateBitonicSortPattern( myRank, nGhosts, commSize,
				  myRank+nGhosts, 
				  (commSize+nGhosts)/2, 
				  0,
				  ASCENDING, &stack );
    }
    else {
      int fieldBase= (commSize+nGhosts)/2;
      generateBitonicSortPattern( myRank, nGhosts, commSize,
				  (myRank+nGhosts)-fieldBase, 
				  (commSize+nGhosts)/2,
				  fieldBase,
				  DESCENDING, &stack );
    }
    generateBitonicMerge( myRank, nGhosts, commSize, 
			  myRank+nGhosts, commSize+nGhosts, 0, 
			  ASCENDING, &stack );
  }
  generateLocalSort( myRank, nGhosts, commSize, ASCENDING, &stack );
    
  while (stack) {
    se_push(stack->partner, stack->op, stack->dir, pRealStack);
    se_pop(&stack);
  }
}

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
    fprintf(stderr,"unable to allocate %ld bytes!\n",((nmemb+1)/2)*size);
    MPI_Abort(MPI_COMM_WORLD,-1);
  }
  if (!(tmp=(void*)malloc(size))) {
    fprintf(stderr,"unable to allocate %ld bytes!\n",size);
    MPI_Abort(MPI_COMM_WORLD,-1);
  }

  MPITEST(rc, MPI_Comm_size(comm,&commSize));
  MPITEST(rc, MPI_Comm_rank(comm,&myRank));

#if DEBUG
  _gblRank= myRank;
#endif

  nLevels= 1;
  while (nLevels<commSize) nLevels *= 2;
  nGhosts= nLevels - commSize;

  generateFullPattern(commSize,nLevels,nGhosts,myRank,&stack);
  TIMECOUNT(nDummy, dummyTime);
  while (stack) {
    int partner= stack->partner;
    Op op= stack->op;
    Dir dir= stack->dir;
    Dir otherDir= (dir==ASCENDING)?DESCENDING:ASCENDING;
    int uphillFlag= (partner<myRank);
    se_pop(&stack);
#if DEBUG
    if (DEBUG_RANK_EXPR) {
      switch (op) {
      case OP_SORT: 
	fprintf(stderr,"rank %d plus %d ghosts: sort partner %d %s\n",
		myRank,nGhosts,partner,
		((dir==ASCENDING)?"ascending":"descending"));
	break;
      case OP_MERGE:
	fprintf(stderr,"rank %d plus %d ghosts: merge partner %d %s\n",
		myRank,nGhosts,partner,
		((dir==ASCENDING)?"ascending":"descending"));
	break;
      }
    }
#endif
    switch (op) {
    case OP_SORT: 
      if (partner==myRank) {
	if (!USE_MERGESORT || firstSort) {
	  TIMECOUNT(nDummy, dummyTime);
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
	  TIMECOUNT(nQSorts, qsortTime);
	  firstSort= 0;
	}
	else {
	  /* Sequence is bitonic.  Use a merge sort, which is much faster. */
	  TIMECOUNT(nDummy, dummyTime);
	  if (dir==ASCENDING)
	    bitonicMergeSortAscending(base, nmemb, size, compar,
				      scratch, tmp);
	  else
	    bitonicMergeSortDescending(base, nmemb, size, compar, 
				       scratch, tmp);
	  TIMECOUNT(nMergeSorts, mergesortTime);
	}
      }
      else {
	fprintf(stderr,"### sort pattern error: %d != %d!\n",
		myRank,partner);
	MPI_Abort(MPI_COMM_WORLD,-1);
      }
      break;
    case OP_MERGE:
      /* If the partner is a ghost, we just don't merge! */
      if (partner>=0) {
	if (dir==ASCENDING) {
	  if (uphillFlag) {
	    merge_ascending_uphill(comm,base,nmemb,size,compar,
				   myRank,partner,scratch,tmp);
	  }
	  else {
	    merge_ascending_downhill(comm,base,nmemb,size,compar,
				     myRank,partner,scratch,tmp);
	  }
	}
	else {
	  if (uphillFlag) {
	    merge_descending_uphill(comm,base,nmemb,size,compar,
				    myRank,partner,scratch,tmp);
	  }
	  else {
	    merge_descending_downhill(comm,base,nmemb,size,compar,
				      myRank,partner,scratch,tmp);
	  }
	}
      }
      break;
    }
#if DEBUG
    if (DEBUG_RANK_EXPR) bitonicTest(base,nmemb,size,compar);
#endif
  }

#if STATS
  if (myRank==0) {
    fprintf(stderr,
	    "rank %d: nSwaps %d, nSwapMsgs %d, nQSorts %d, nMergeSorts %d\n",
	    myRank,nSwaps,nSwapMsgs,nQSorts,nMergeSorts);
    if (nSwaps>0) fprintf(stderr,"rank %d: time per swap %g\n",
			  myRank, swapTime/nSwaps);
    if (nSwapMsgs>0) fprintf(stderr,"rank %d: time per swap msg exchange %g\n",
			     myRank, swapMsgTime/nSwapMsgs);
    if (nQSorts>0) fprintf(stderr,"rank %d: time per qsort %g\n",
			   myRank, qsortTime/nQSorts);
    if (nMergeSorts>0) fprintf(stderr,"rank %d: time per mergesort %g\n",
			       myRank, mergesortTime/nMergeSorts);
  }
#endif

  free(scratch);
  free(tmp);
}

