
#include <math.h>
#include <vector>
#include "_hypre_utilities.h"
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"

class HypreSolver2D
{

private:

	int myid, num_procs;
	int N, N0, N1;
	double Ly, Lx, dy, dx;

	int ilower, iupper, y0, ny, nx;
	int local_size, extra;
	int num_iterations;
	double final_res_norm;

	double h2;

	HYPRE_IJMatrix A;
	HYPRE_ParCSRMatrix parcsr_A;
	HYPRE_IJVector hv_b;
	HYPRE_ParVector par_b;
	HYPRE_IJVector hv_x;
	HYPRE_ParVector par_x;

	HYPRE_Solver solver, precond;

	std::vector<int> rows;

	void set_rhs(double *rhs);	

	void build_A();	

public:

	HypreSolver2D(ptrdiff_t N0, double Ly, ptrdiff_t N1, double Lx);

	void set_x0(double *x0);
	
	void build_solver();

	void solve(double *x);

	int get_num_iterations();

	double get_final_res_norm();

	int get_local_size();
	int get_y0();
	int get_ny();
	int get_nx();

	~HypreSolver2D();

};
