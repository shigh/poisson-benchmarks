
CC        = mpicc
CXX       = mpicxx
HYPRE_DIR = /home/user01/src/hypre-2.9.0b-babel/src/hypre

COPTS     = -g -Wall
CINCLUDES = -I$(HYPRE_DIR)/include
CDEFS     = -DHAVE_CONFIG_H -DHYPRE_TIMING
CFLAGS    = $(COPTS) $(CINCLUDES) $(CDEFS)

CXXOPTS   = $(COPTS) -Wno-deprecated
CXXINCLUDES = $(CINCLUDES) -I..
CXXDEFS   = $(CDEFS)
#IFLAGS_BXX = -I../babel-runtime/sidl
CXXFLAGS  = $(CXXOPTS) $(CXXINCLUDES) $(CXXDEFS) $(IFLAGS_BXX)

LINKOPTS  = $(COPTS)
LIBS      = -L$(HYPRE_DIR)/lib -lHYPRE -lm
LFLAGS    = $(LINKOPTS) $(LIBS) -lstdc++

.SUFFIXES: .c .cxx

.c.o:
	$(CC) $(CFLAGS) -c $<
.cxx.o:
	$(CXX) $(CXXFLAGS) -c $<

ALLPROGS = ex3 ex5

all: $(ALLPROGS)

ex3: ex3.o
	$(CC) -o $@ $^ $(LFLAGS)

run-ex3: ex3
	mpirun -np 16 ex3 -n 33 -solver 0 -v 1 1

ex5: ex5.o
	$(CC) -o $@ $^ $(LFLAGS)

run-ex5: ex5
	 mpirun -np 4 ex5

clean:
	rm -vf *.o $(ALLPROGS)

