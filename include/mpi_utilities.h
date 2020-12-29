/**
 * File:   mpi_utilities.h
 * Author: akirby
 *
 * Created on April 15, 2020, 11:35 PM
 */

#ifndef MPI_UTILITIES_H
#define MPI_UTILITIES_H

/* header files */
#include "mpi_var.h"
#include "dlmg_types.h"

/* system header files */
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

int mpi_initialize(int argc, char **argv, compute_ctx_t *global, partition_info_t *partition);
void mpi_finalize();
void mpi_set_info(compute_ctx_t *ctx);
void mpi_range_group(MPI_Comm orig_comm, MPI_Group orig_group, compute_ctx_t *group, std::string base);
Real mpi_timer();

#ifdef __cplusplus
}
#endif
#endif /* MPI_UTILITIES_H */