
EXECUTABLES = sort_tester

CFLAGS = -I. -O2
#CFLAGS = -I. -g

LIBS = -lmpi -llam -lutil -lm

sort_tester: sort_tester.o jpsort.o
	cc -o $@ sort_tester.o jpsort.o $(LIBS)

clean:
	rm *.o $(EXECUTABLES)

