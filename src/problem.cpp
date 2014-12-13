
#include "problem.hpp"

void build_problem(double *x,
				   int ystart, int ny, double dy,
				   int xstart, int nx, double dx,
				   double k)
{

	const double kdx = k*dx;
	const double kdy = k*dy;

	for(int j=0; j<ny; j++)
		for(int i=0; i<nx; i++)
			x[j*nx+i] = sin(kdx*(xstart+i))*sin(kdy*(ystart+j));

}

void build_solution(double *x,
					int ystart, int ny, double dy,
					int xstart, int nx, double dx,
					double k)
{

	const double kdx = k*dx;
	const double kdy = k*dy;
	//const double s = 1./(kdx*kdx + kdy*kdy);
	const double s = 1./(2*k*k);

	for(int j=0; j<ny; j++)
		for(int i=0; i<nx; i++)
			x[j*nx+i] = s*sin(kdx*(xstart+i))*sin(kdy*(ystart+j));

}
