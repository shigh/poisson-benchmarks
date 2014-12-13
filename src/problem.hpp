
#include <vector>
#include <math.h>

class TestProblem2D
{

private:

	double dy, dx;
	double k;

	int xstart, xend;
	int ystart, yend;

public:

	TestProblem2D(int ystart, int yend, double dy,
				  int xstart, int xend, double dx,
				  double k); 


	void build_problem(std::vector<double>& x);

	void build_solution(std::vector<double>& x);

};
