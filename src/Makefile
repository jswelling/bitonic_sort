
EXECUTABLES = sort_tester

CFLAGS = -I. -I/usr/include/mpich -O2
#CFLAGS = -I. -g

LIBS = -lmpi -lutil -lm

sort_tester: sort_tester.o jpsort.o
	mpicc -o $@ sort_tester.o jpsort.o $(LIBS)

clean:
	rm *.o $(EXECUTABLES)

test: sort_tester
	mpirun -n 4 ./sort_tester -c 1000000 -s 1234
