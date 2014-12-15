
#include "hypre.hpp"


HypreSolver2D::HypreSolver2D(ptrdiff_t N0_, double Ly_, ptrdiff_t N1_, double Lx_):
	N0(N0_), Ly(Ly_), N1(N1_), Lx(Lx_)
{

	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	N = N0*N1;
	dy = Ly/(N0+1);
	dx = Lx/(N1+1);
	h2 = dy*dy;
	nx = N1;

	int yp    = (int)(N0/num_procs);
	int extra = N0-yp*num_procs;
	if(myid<extra)
	{
		y0 = myid*yp + myid;
		ny = yp+1;
	}
	else
	{
		y0 = myid*yp + extra;
		ny = yp;
	}

	local_size = ny*nx;
	ilower     = y0*nx;
	iupper     = ilower+local_size-1;

	// Build rhs and init x
	HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &hv_b);
	HYPRE_IJVectorSetObjectType(hv_b, HYPRE_PARCSR);
	HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &hv_x);
	HYPRE_IJVectorSetObjectType(hv_x, HYPRE_PARCSR);

	std::vector<double> rhs_values(local_size, 1.);
	std::vector<double> x_values(local_size, 0.);
	rows.resize(local_size);

	for (int i=0; i<local_size; i++)
		rows[i] = ilower + i;

	set_x0(&x_values[0]);
	set_rhs(&rhs_values[0]);

	build_A();
	build_solver();

}

void HypreSolver2D::set_rhs(double *rhs)
{

	for(int i=0; i<local_size; i++)
		rhs[i] = rhs[i]*h2;

	HYPRE_IJVectorInitialize(hv_b);
	HYPRE_IJVectorSetValues(hv_b, local_size, &rows[0], rhs);
	HYPRE_IJVectorAssemble(hv_b);
	HYPRE_IJVectorGetObject(hv_b, (void **) &par_b);

}

void HypreSolver2D::set_x0(double *x0)
{

	HYPRE_IJVectorInitialize(hv_x);
	HYPRE_IJVectorSetValues(hv_x, local_size, &rows[0], x0);
	HYPRE_IJVectorAssemble(hv_x);
	HYPRE_IJVectorGetObject(hv_x, (void **) &par_x);

}
	
void HypreSolver2D::build_solver()
{

	HYPRE_BoomerAMGCreate(&solver);
	//HYPRE_BoomerAMGSetPrintLevel(solver, 3);  /* print solve info + parameters */
	HYPRE_BoomerAMGSetCoarsenType(solver, 6); /* Falgout coarsening */
	HYPRE_BoomerAMGSetRelaxType(solver, 3);   /* G-S/Jacobi hybrid relaxation */
	HYPRE_BoomerAMGSetNumSweeps(solver, 1);   /* Sweeeps on each level */
	HYPRE_BoomerAMGSetMaxLevels(solver, 20);  /* maximum number of levels */
	HYPRE_BoomerAMGSetTol(solver, 1e-7);      /* conv. tolerance */

	HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x);
	
}

void HypreSolver2D::solve(double *x)
{

	set_rhs(x);

	HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);

	HYPRE_IJVectorGetValues(hv_x, local_size, &rows[0], x);

	HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
	HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);

}

void HypreSolver2D::build_A()
{

	HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);
	HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
	HYPRE_IJMatrixInitialize(A);

	int nnz;
	double values[5];
	int cols[5];

	for(int i=ilower; i<=iupper; i++)
	{
		nnz = 0;

		// North
		if ((i-N1)>=0)
		{
			cols[nnz] = i-N1;
			values[nnz] = -1.0;
			nnz++;
		}

		// Left
		if (i%N1)
		{
			cols[nnz] = i-1;
			values[nnz] = -1.0;
			nnz++;
		}

		// Center
		cols[nnz] = i;
		values[nnz] = 4.0;
		nnz++;

		// Right
		if ((i+1)%N1)
		{
			cols[nnz] = i+1;
			values[nnz] = -1.0;
			nnz++;
		}

		// South
		if ((i+N1)<N)
		{
			cols[nnz] = i+N1;
			values[nnz] = -1.0;
			nnz++;
		}

		HYPRE_IJMatrixSetValues(A, 1, &nnz, &i, cols, values);
	}

	HYPRE_IJMatrixAssemble(A);
	HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);

}	

int HypreSolver2D::get_num_iterations()
{
	return num_iterations;
}

double HypreSolver2D::get_final_res_norm()
{
	return final_res_norm;
}

int HypreSolver2D::get_local_size()
{
	return local_size;
}

int HypreSolver2D::get_y0()
{
	return y0;
}

int HypreSolver2D::get_ny()
{
	return ny;
}

int HypreSolver2D::get_nx()
{
	return nx;
}

HypreSolver2D::~HypreSolver2D()
{
	HYPRE_IJMatrixDestroy(A);
	HYPRE_IJVectorDestroy(hv_b);
	HYPRE_IJVectorDestroy(hv_x);
	HYPRE_BoomerAMGDestroy(solver);
}

HypreSolver3D::HypreSolver3D(ptrdiff_t N0_, double Lz_,
							 ptrdiff_t N1_, double Ly_,
							 ptrdiff_t N2_, double Lx_):
	N0(N0_), Lz(Lz_), N1(N1_), Ly(Ly_), N2(N2_), Lx(Lx_)
{

	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	N = N0*N1*N2;
	dz = Lz/(N0+1);
	dy = Ly/(N1+1);
	dx = Lx/(N2+1);
	h2 = dy*dy;
	nx = N2;
	ny = N1;

	int zp    = (int)(N0/num_procs);
	int extra = N0-zp*num_procs;
	if(myid<extra)
	{
		z0 = myid*zp + myid;
		nz = zp+1;
	}
	else
	{
		z0 = myid*zp + extra;
		nz = zp;
	}

	local_size = ny*nx*nz;
	ilower     = z0*nx*ny;
	iupper     = ilower+local_size-1;

	// Build rhs and init x
	HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &hv_b);
	HYPRE_IJVectorSetObjectType(hv_b, HYPRE_PARCSR);
	HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &hv_x);
	HYPRE_IJVectorSetObjectType(hv_x, HYPRE_PARCSR);

	std::vector<double> rhs_values(local_size, 1.);
	std::vector<double> x_values(local_size, 0.);
	rows.resize(local_size);

	for (int i=0; i<local_size; i++)
		rows[i] = ilower + i;

	set_x0(&x_values[0]);
	set_rhs(&rhs_values[0]);

	build_A();
	build_solver();

}

void HypreSolver3D::set_rhs(double *rhs)
{

	for(int i=0; i<local_size; i++)
		rhs[i] = rhs[i]*h2;

	HYPRE_IJVectorInitialize(hv_b);
	HYPRE_IJVectorSetValues(hv_b, local_size, &rows[0], rhs);
	HYPRE_IJVectorAssemble(hv_b);
	HYPRE_IJVectorGetObject(hv_b, (void **) &par_b);

}

void HypreSolver3D::set_x0(double *x0)
{

	HYPRE_IJVectorInitialize(hv_x);
	HYPRE_IJVectorSetValues(hv_x, local_size, &rows[0], x0);
	HYPRE_IJVectorAssemble(hv_x);
	HYPRE_IJVectorGetObject(hv_x, (void **) &par_x);

}
	
void HypreSolver3D::build_solver()
{

	HYPRE_BoomerAMGCreate(&solver);
	//HYPRE_BoomerAMGSetPrintLevel(solver, 3);  /* print solve info + parameters */
	HYPRE_BoomerAMGSetCoarsenType(solver, 6); /* Falgout coarsening */
	HYPRE_BoomerAMGSetRelaxType(solver, 3);   /* G-S/Jacobi hybrid relaxation */
	HYPRE_BoomerAMGSetNumSweeps(solver, 1);   /* Sweeeps on each level */
	HYPRE_BoomerAMGSetMaxLevels(solver, 20);  /* maximum number of levels */
	HYPRE_BoomerAMGSetTol(solver, 1e-7);      /* conv. tolerance */

	HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x);
	
}

void HypreSolver3D::solve(double *x)
{

	set_rhs(x);

	HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);

	HYPRE_IJVectorGetValues(hv_x, local_size, &rows[0], x);

	HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
	HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);

}

void HypreSolver3D::build_A()
{

	HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);
	HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
	HYPRE_IJMatrixInitialize(A);

	const int N12  = N1*N2;
	const int N012 = N0*N1*N2;

	int nnz;
	double values[7];
	int cols[7];

	int i;
	for(int j=ilower; j<=iupper; j++)
	{
		nnz = 0;
		i = j%N12;//*((int)(j/(N12)));

		// North
		if ((i-N2)>=0)
		{
			cols[nnz] = j-N2;
			values[nnz] = -1.0;
			nnz++;
		}

		// Left
		if (i%N2)
		{
			cols[nnz] = j-1;
			values[nnz] = -1.0;
			nnz++;
		}

		// Center
		cols[nnz] = j;
		values[nnz] = 6.0;
		nnz++;

		// Right
		if ((i+1)%N2)
		{
			cols[nnz] = j+1;
			values[nnz] = -1.0;
			nnz++;
		}

		// South
		if ((i+N2)<N12)
		{
			cols[nnz] = j+N2;
			values[nnz] = -1.0;
			nnz++;
		}

		// Top
		if ((j+N12)<N012)
		{
			cols[nnz] = j+N12;
			values[nnz] = -1.0;
			nnz++;
		}

		// Bottom
		if ((j-N12)>=0)
		{
			cols[nnz] = j-N12;
			values[nnz] = -1.0;
			nnz++;
		}

		HYPRE_IJMatrixSetValues(A, 1, &nnz, &j, cols, values);
	}

	HYPRE_IJMatrixAssemble(A);
	HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);

}	

int HypreSolver3D::get_num_iterations()
{
	return num_iterations;
}

double HypreSolver3D::get_final_res_norm()
{
	return final_res_norm;
}

int HypreSolver3D::get_local_size()
{
	return local_size;
}

int HypreSolver3D::get_z0()
{
	return z0;
}

int HypreSolver3D::get_nz()
{
	return nz;
}

int HypreSolver3D::get_ny()
{
	return ny;
}

int HypreSolver3D::get_nx()
{
	return nx;
}

HypreSolver3D::~HypreSolver3D()
{
	HYPRE_IJMatrixDestroy(A);
	HYPRE_IJVectorDestroy(hv_b);
	HYPRE_IJVectorDestroy(hv_x);
	HYPRE_BoomerAMGDestroy(solver);
}
