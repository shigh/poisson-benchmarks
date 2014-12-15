
#include "hypre.hpp"

class HypreSolver2D
{

private:

	int i;
	int myid, num_procs;
	int N, n;

	int ilower, iupper;
	int local_size, extra;

	int print_system;

	double h, h2;

	HYPRE_IJMatrix A;
	HYPRE_ParCSRMatrix parcsr_A;
	HYPRE_IJVector b;
	HYPRE_ParVector par_b;
	HYPRE_IJVector x;
	HYPRE_ParVector par_x;

	HYPRE_Solver solver, precond;

	std::vector<double> rhs_values, x_values;
	std::vector<int> rows;
	//double *rhs_values, *x_values;
	//int    *rows;


public:

	HypreSolver2D()
	{
		MPI_Comm_rank(MPI_COMM_WORLD, &myid);
		MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

		/* Default problem parameters */
		n = 33;
		print_system = 0;

		/* Preliminaries: want at least one processor per row */
		if (n*n < num_procs) n = sqrt(num_procs) + 1;
		N = n*n; /* global number of rows */
		h = 1.0/(n+1); /* mesh size*/
		h2 = h*h;

		local_size = N/num_procs;
		extra = N - local_size*num_procs;

		ilower = local_size*myid;
		ilower += hypre_min(myid, extra);

		iupper = local_size*(myid+1);
		iupper += hypre_min(myid+1, extra);
		iupper = iupper - 1;
		local_size = iupper - ilower + 1;

		// Build rhs and init x
		HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b);
		HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
		HYPRE_IJVectorInitialize(b);

		HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &x);
		HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
		HYPRE_IJVectorInitialize(x);

		rhs_values.resize(local_size);
		x_values.resize(local_size);
		rows.resize(local_size);

		for (i=0; i<local_size; i++)
		{
			rhs_values[i] = h2;
			x_values[i] = 0.0;
			rows[i] = ilower + i;
		}

		HYPRE_IJVectorSetValues(b, local_size, &rows[0], &rhs_values[0]);
		HYPRE_IJVectorSetValues(x, local_size, &rows[0], &x_values[0]);


		HYPRE_IJVectorAssemble(b);
		HYPRE_IJVectorGetObject(b, (void **) &par_b);

		HYPRE_IJVectorAssemble(x);
		HYPRE_IJVectorGetObject(x, (void **) &par_x);

		build_A();
		build_solver();

	}

	void build_solver()
	{

		int num_iterations;
		double final_res_norm;

		HYPRE_BoomerAMGCreate(&solver);

		//HYPRE_BoomerAMGSetPrintLevel(solver, 3);  /* print solve info + parameters */
		HYPRE_BoomerAMGSetCoarsenType(solver, 6); /* Falgout coarsening */
		HYPRE_BoomerAMGSetRelaxType(solver, 3);   /* G-S/Jacobi hybrid relaxation */
		HYPRE_BoomerAMGSetNumSweeps(solver, 1);   /* Sweeeps on each level */
		HYPRE_BoomerAMGSetMaxLevels(solver, 20);  /* maximum number of levels */
		HYPRE_BoomerAMGSetTol(solver, 1e-7);      /* conv. tolerance */

		HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x);
		HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);

		/* Run info - needed logging turned on */
		HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
		HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
		if (myid == 0)
		{
			printf("\n");
			printf("Iterations = %d\n", num_iterations);
			printf("Final Relative Residual Norm = %e\n", final_res_norm);
			printf("\n");
		}


	}

	void build_A()
	{

		HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);
		HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
		HYPRE_IJMatrixInitialize(A);

		int nnz;
		double values[5];
		int cols[5];

		for (i = ilower; i <= iupper; i++)
		{
			nnz = 0;

			// North
			if ((i-n)>=0)
			{
				cols[nnz] = i-n;
				values[nnz] = -1.0;
				nnz++;
			}

			// Left
			if (i%n)
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
			if ((i+1)%n)
			{
				cols[nnz] = i+1;
				values[nnz] = -1.0;
				nnz++;
			}

			// South
			if ((i+n)< N)
			{
				cols[nnz] = i+n;
				values[nnz] = -1.0;
				nnz++;
			}

			HYPRE_IJMatrixSetValues(A, 1, &nnz, &i, cols, values);
		}

		HYPRE_IJMatrixAssemble(A);
		HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);

	}	

	~HypreSolver2D()
	{
		HYPRE_IJMatrixDestroy(A);
		HYPRE_IJVectorDestroy(b);
		HYPRE_IJVectorDestroy(x);
		HYPRE_BoomerAMGDestroy(solver);
	}


};

int hypre_solve ()
{

	HypreSolver2D solver;

   return(0);
}

