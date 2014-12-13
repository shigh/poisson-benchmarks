
CXX      = mpic++
CXXFLAGS = -g -fPIC
LDLIBS = -lfftw3_mpi -lfftw3 -lm

DEPS = problem.hpp fftwlocal.hpp fftwmpi.hpp mpi_stats.hpp
OBJS = problem.o fftwlocal.o fftwmpi.o mpi_stats.o

testsolvers: testsolvers.pyx $(OBJS) $(DEPS)
	python setup.py build_ext --inplace

test: test.o $(OBJS)
	$(CXX)  $(CXXFLAGS) -o test test.o $(OBJS) $(LDLIBS)

run-test: test
	mpirun -n 2 ./test

%.o : %.cpp %.hpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

clean:
	rm -vf *.o *.so testsolvers.cpp
	rm -vrf build
	rm -vf test
