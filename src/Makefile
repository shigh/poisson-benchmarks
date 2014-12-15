
CXX         = mpic++
CXXFLAGS    = -g -fPIC
CXXINCLUDES = -I$(HYPRE_DIR)/include -I$(FFTW_DIR)/include
CXXDEFS     = -DHAVE_CONFIG_H -DHYPRE_TIMING

LDLIBS  = -L$(FFTW_DIR)/lib  -lfftw3_mpi -lfftw3 -lm
LDLIBS += -L$(HYPRE_DIR)/lib -lHYPRE 

DEPS = problem.hpp fftwlocal.hpp fftwmpi.hpp mpi_stats.hpp hypre.hpp
OBJS = problem.o fftwlocal.o fftwmpi.o mpi_stats.o hypre.o

testsolvers: testsolvers.pyx $(OBJS) $(DEPS)
	python setup.py build_ext --inplace

test: test.o $(OBJS)
	$(CXX)  $(CXXFLAGS) -o test test.o $(OBJS) $(LDLIBS)

run-test: test
	mpirun -n 2 ./test

test.o : test.cpp
	$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -o $@ -c $<

%.o : %.cpp %.hpp
	$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -o $@ -c $<

clean:
	rm -vf *.o *.so testsolvers.cpp
	rm -vrf build
	rm -vf test tmp.py
