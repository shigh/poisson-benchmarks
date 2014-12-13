
CXX      = g++
CXXFLAGS = -g -fPIC
LDLIBS = -lfftw3 -lm

DEPS = problem.hpp fftwlocal.hpp
OBJS = problem.o fftwlocal.o

testsolvers: testsolvers.pyx $(OBJS) $(DEPS)
	python setup.py build_ext --inplace

%.o : %.cpp %.hpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

clean:
	rm -vf *.o *.so testsolvers.cpp
	rm -vrf build
