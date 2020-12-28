/*
 * -All callers must give the same nmemb, size, and comparator.
 * 
 * -There are assumed to be 2^n members in the communicator.
 *
 * -The routine does not start with a barrier; the procs
 * are assumed to call jpsort after a barrier in the calling code.
 */
void jpSort(MPI_Comm comm,
	    void* base,
	    size_t nmemb,
	    size_t size,
	    int (*compar)(const void*, const void*));
