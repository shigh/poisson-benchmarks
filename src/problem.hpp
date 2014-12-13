
#include <vector>
#include <math.h>


// 1D test problem setup
void build_problem(double *x,
				   int xstart, int nx, double dx,
				   double k);

void build_solution(double *x,
					int xstart, int nx, double dx,
					double k);

// 2D test problem setup
void build_problem(double *x,
				   int ystart, int ny, double dy,
				   int xstart, int nx, double dx,
				   double k);

void build_solution(double *x,
					int ystart, int ny, double dy,
					int xstart, int nx, double dx,
					double k);

