
#include "hypre.hpp"

int hypre_solve (int argc, char *argv[])
{
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

   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /* Default problem parameters */
   n = 33;
   print_system = 0;


   /* Parse command line */
   {
      int arg_index = 0;
      int print_usage = 0;

      while (arg_index < argc)
      {
         if ( strcmp(argv[arg_index], "-n") == 0 )
         {
            arg_index++;
            n = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-print_system") == 0 )
         {
            arg_index++;
            print_system = 1;
         }
         else if ( strcmp(argv[arg_index], "-help") == 0 )
         {
            print_usage = 1;
            break;
         }
         else
         {
            arg_index++;
         }
      }

      if ((print_usage) && (myid == 0))
      {
         printf("\n");
         printf("Usage: %s [<options>]\n", argv[0]);
         printf("\n");
         printf("  -n <n>              : problem size in each direction (default: 33)\n");
         printf("  -print_system       : print the matrix and rhs\n");
         printf("\n");
      }

   }

   /* Preliminaries: want at least one processor per row */
   if (n*n < num_procs) n = sqrt(num_procs) + 1;
   N = n*n; /* global number of rows */
   h = 1.0/(n+1); /* mesh size*/
   h2 = h*h;

   /* Each processor knows only of its own rows - the range is denoted by ilower
      and upper.  Here we partition the rows. We account for the fact that
      N may not divide evenly by the number of processors. */
   local_size = N/num_procs;
   extra = N - local_size*num_procs;

   ilower = local_size*myid;
   ilower += hypre_min(myid, extra);

   iupper = local_size*(myid+1);
   iupper += hypre_min(myid+1, extra);
   iupper = iupper - 1;

   /* How many rows do I have? */
   local_size = iupper - ilower + 1;

   /* Create the matrix.
      Note that this is a square matrix, so we indicate the row partition
      size twice (since number of rows = number of cols) */
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);

   /* Choose a parallel csr format storage (see the User's Manual) */
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);

   /* Initialize before setting coefficients */
   HYPRE_IJMatrixInitialize(A);

   /* Now go through my local rows and set the matrix entries.
      Each row has at most 5 entries.
   */
   {
      int nnz;
      double values[5];
      int cols[5];

      for (i = ilower; i <= iupper; i++)
      {
         nnz = 0;

         /* The left identity block:position i-n */
         if ((i-n)>=0)
         {
            cols[nnz] = i-n;
            values[nnz] = -1.0;
            nnz++;
         }

         /* The left -1: position i-1 */
         if (i%n)
         {
            cols[nnz] = i-1;
            values[nnz] = -1.0;
            nnz++;
         }

         /* Set the diagonal: position i */
         cols[nnz] = i;
         values[nnz] = 4.0;
         nnz++;

         /* The right -1: position i+1 */
         if ((i+1)%n)
         {
            cols[nnz] = i+1;
            values[nnz] = -1.0;
            nnz++;
         }

         /* The right identity block:position i+n */
         if ((i+n)< N)
         {
            cols[nnz] = i+n;
            values[nnz] = -1.0;
            nnz++;
         }

         /* Set the values for row i */
         HYPRE_IJMatrixSetValues(A, 1, &nnz, &i, cols, values);
      }
   }

   /* Assemble after setting the coefficients */
   HYPRE_IJMatrixAssemble(A);


   /* Get the parcsr matrix object to use */
   HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);


   /* Create the rhs and solution */
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,&b);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(b);

   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,&x);
   HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(x);

   /* Set the rhs values to h^2 and the solution to zero */
   {
      double *rhs_values, *x_values;
      int    *rows;

      rhs_values = (double*)calloc(local_size, sizeof(double));
      x_values = (double*)calloc(local_size, sizeof(double));
      rows = (int*)calloc(local_size, sizeof(int));

      for (i=0; i<local_size; i++)
      {
         rhs_values[i] = h2;
         x_values[i] = 0.0;
         rows[i] = ilower + i;
      }

      HYPRE_IJVectorSetValues(b, local_size, rows, rhs_values);
      HYPRE_IJVectorSetValues(x, local_size, rows, x_values);

      free(x_values);
      free(rhs_values);
      free(rows);
   }


   HYPRE_IJVectorAssemble(b);
   HYPRE_IJVectorGetObject(b, (void **) &par_b);

   HYPRE_IJVectorAssemble(x);
   HYPRE_IJVectorGetObject(x, (void **) &par_x);

   if (print_system)
   {
      HYPRE_IJMatrixPrint(A, "IJ.out.A");
      HYPRE_IJVectorPrint(b, "IJ.out.b");
   }


   /* AMG */
   {
      int num_iterations;
      double final_res_norm;

      /* Create solver */
      HYPRE_BoomerAMGCreate(&solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      //HYPRE_BoomerAMGSetPrintLevel(solver, 3);  /* print solve info + parameters */
      HYPRE_BoomerAMGSetCoarsenType(solver, 6); /* Falgout coarsening */
      HYPRE_BoomerAMGSetRelaxType(solver, 3);   /* G-S/Jacobi hybrid relaxation */
      HYPRE_BoomerAMGSetNumSweeps(solver, 1);   /* Sweeeps on each level */
      HYPRE_BoomerAMGSetMaxLevels(solver, 20);  /* maximum number of levels */
      HYPRE_BoomerAMGSetTol(solver, 1e-7);      /* conv. tolerance */

      /* Now setup and solve! */
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

      /* Destroy solver */
      HYPRE_BoomerAMGDestroy(solver);
   }

   /* Clean up */
   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJVectorDestroy(x);

   return(0);
}

