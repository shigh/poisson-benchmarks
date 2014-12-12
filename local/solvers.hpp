#include <fftw3.h>
#include <cmath>
#include <iostream>
#include <vector>

void solve_1d(int N, double *x, double L);

void solve_2d(int ny, int nx, double *x, double Ly, double Lx);

void solve_3d(int nz, int ny, int nx, double *x, double Lz, double Ly, double Lx);
