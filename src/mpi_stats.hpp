#ifndef __MPI_STATS_H
#define __MPI_STATS_H

#include "mpi.h"

bool check_mpi();

int get_rank();

int get_comm_size();

int thread_level();

bool has_thread_multiple();

#endif
