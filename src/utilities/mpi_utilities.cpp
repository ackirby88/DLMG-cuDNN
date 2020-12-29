/**
 * File:   mpi_utilities.cpp
 * Author: akirby
 *
 * Created on April 17, 2020, 12:07 PM
 */

/* header files */
#include "mpi_utilities.h"

int mpi_initialize(int argc, char **argv, compute_ctx_t *global, partition_info_t *partition){
    MPI_Group orig_group;

    int mpi_return;
    int max_ranks;
    int max_usuable_ranks;

    /* initialize MPI */
    mpi_return = MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &max_ranks);

    /* check ranks per model: set model parallelism to use all ranks */
    if (partition->nranks_per_model > max_ranks) {
        printf("[DLMG] *** WARNING ***: "
               "Number of requested model ranks more that total MPI ranks [%d > %d]. "
               "Resetting nranks_per_model = %d\n",
                partition->nranks_per_model, max_ranks, max_ranks);
        partition->nranks_per_model = max_ranks;
    } else if (max_ranks % partition->nranks_per_model != 0) {
        for (int i = partition->nranks_per_model; i >= 1; --i) {
            if (max_ranks % i == 0) {
                partition->nranks_per_model = i;
                break;
            }
        }
        printf("[DLMG] *** WARNING ***: "
               "Number of requested model ranks does not partition total MPI ranks evenly. "
               "Resetting nranks_per_model = %d\n",
                partition->nranks_per_model);
    }

    /* extract the original group handle */
    MPI_Comm_group(MPI_COMM_WORLD, &orig_group);

    /* integer division for whole groups */
    partition->nbatch_groups = max_ranks/partition->nranks_per_model;

    /* number of global ranks which form complete model groups */
    max_usuable_ranks = partition->nranks_per_model * partition->nbatch_groups;

    /* set global rank range information */
    global->group_range[0][0] = 0;                      /* start rank */
    global->group_range[0][1] = max_usuable_ranks - 1;  /* end rank */
    global->group_range[0][2] = 1;                      /* stride */

    mpi_range_group(MPI_COMM_WORLD, orig_group, global, "Global-");

    /* set ranks information */
    mpi_set_info(global);

    /* build HALF precision MPI datatype */
    MPI_Datatype MPI_HALF; // extern global
    MPI_Type_contiguous(2, MPI_BYTE, &MPI_HALF);
    MPI_Type_commit(&MPI_HALF);

    return mpi_return;
}

void mpi_finalize(){
    MPI_Finalize();
}

void mpi_set_info(compute_ctx_t *ctx){
    /* assign MPI rank and size */
    if (ctx->mpi_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(ctx->mpi_comm,&ctx->rank);
        MPI_Comm_size(ctx->mpi_comm,&ctx->nranks);
        DEBUG_MESG(printf("[DLMG] My MPI Rank: %d\n",ctx->rank))
    }
}

void mpi_range_group(MPI_Comm orig_comm, MPI_Group orig_group, compute_ctx_t *group, std::string base){
    /* divide tasks into distinct groups based upon group range */
    MPI_Group_range_incl(orig_group, 1, group->group_range, &group->mpi_group);

    /* create new communicator for group */
    MPI_Comm_create(orig_comm, group->mpi_group, &group->mpi_comm);

    /* set rank and size info */
    mpi_set_info(group);

    std::string name = base + std::to_string(group->group_id);
    const char * commname = name.c_str();
    MPI_Comm_set_name(group->mpi_comm, commname);
}

Real mpi_timer(){
    return MPI_Wtime();
}