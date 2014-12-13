#include "mpi.h"
#include "mpi_stats.hpp"

// You could call these from mpi4py...
// but what would be the fun in that!

bool check_mpi()
{
  int flag;
  MPI_Initialized(&flag);
  return flag == 1;
}

int get_rank()
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

int get_comm_size()
{
  int numtasks;
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  return numtasks;
}

int thread_level()
{
  int provided = -1;
  if(check_mpi()) 
    MPI_Query_thread(&provided);

  return provided;
}

bool has_thread_multiple()
{
  return thread_level() == MPI_THREAD_MULTIPLE;
}
