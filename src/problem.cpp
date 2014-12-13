
#include "problem.hpp"

TestProblem2D::TestProblem2D(int ystart_, int yend_, double dy_,
							 int xstart_, int xend_, double dx_,
							 double k_):
	ystart(ystart_), yend(yend_), dy(dy_),
	xstart(xstart_), xend(xend_), dx(dx_),
	k(k_)
{


}

void TestProblem2D::build_problem(std::vector<double>& x)
{

	const int lnx = xend-xstart;
	const int lny = yend-ystart;

	const double kdx = k*dx;
	const double kdy = k*dy;

	x.resize(lnx*lny);
	for(int j=0; j<lny; j++)
		for(int i=0; i<lnx; i++)
			x[j*lnx+i] = sin(kdx*(xstart+i))*sin(kdy*(ystart+j));

}

void TestProblem2D::build_solution(std::vector<double>& x)
{

	const int lnx = xend-xstart;
	const int lny = yend-ystart;

	const double kdx = k*dx;
	const double kdy = k*dy;
	const double s = 1./(kdx*kdx + kdy*kdy);

	x.resize(lnx*lny);
	for(int j=0; j<lny; j++)
		for(int i=0; i<lnx; i++)
			x[j*lnx+i] = s*sin(kdx*(xstart+i))*sin(kdy*(ystart+j));

}
