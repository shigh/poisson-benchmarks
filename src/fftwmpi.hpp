#include <fftw3.h>
#include <fftw3-mpi.h>
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <vector>

void setup_fftw_mpi();

void cleanup_fftw_mpi();

void solve_1d_mpi(int N, double *x, double L);
