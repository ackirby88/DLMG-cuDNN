/**
 * File:   dlmg_functions.cu
 * Author: akirby
 *
 * Created on April 17, 2020, 12:07 PM
 */

/* header files */
#include <stdio.h>
#include <iostream>

#include "DLMG.h"

#define MIN(x,y)  (x)<(y) ? (x):(y)
#define MAX(x,y)  (x)>(y) ? (x):(y)

DLMG::DLMG(int argc,char **argv){
    initialize(argc,argv);
}

DLMG::~DLMG(){}

int DLMG::initialize(int argc,char **argv){
    compute_ctx_t *global = &ctx.global_compute_ctx;

    /* initialize GPU devices as seen from this rank */
    checkCudaErrors(cudaGetDeviceCount(&global->ngpus));

    /* read inputs */
    multigrid.nlevels = 2;
    multigrid_flag = (multigrid.nlevels > 1) ? 1:0;

    /*set number of ranks per network for network parallelism */
    partition_info.sequential = 0;
    partition_info.nranks_per_model = 128;
    partition_info.nlayers_per_block = 64;

    /* initialize MPI */
    int ret = mpi_initialize(argc, argv, global, &partition_info);

    /* partition ranks for model and data parallelism */
    form_model_groups();
    form_data_groups();

    /* set device to allocate cuda handles */
    int gpu_id = global->rank % global->ngpus;
    checkCudaErrors(cudaSetDevice(gpu_id));
    initalize_cuda_handles();

    return ret;
}

void DLMG::initalize_cuda_handles(){
    cudaStreamCreate(&partition_info.cudaStream);
}

void DLMG::form_model_groups(){
    int nranks_per_model = partition_info.nranks_per_model;
    compute_ctx_t *global = &ctx.global_compute_ctx;
    compute_ctx_t *model = &ctx.model_compute_ctx;

    /* =========================================== *
     * Model Parallelism MPI Rank Ranges:          *
     * [0*nranks_per_model, 1*nranks_per_model-1], *
     * [1*nranks_per_model, 2*nranks_per_model-1], *
     * ...                                         *
     *                                             *
     * The range for THIS global rank only include *
     * the global ranks in my MY model set.        *
     * =========================================== */
    /* set number of GPUs seen from this rank */
    model->ngpus = global->ngpus;

    int max_rank_ind = global->nranks - 1;
    int model_index = global->rank / nranks_per_model;  /* integer division for floor */

    /* model group ranges */
    model->group_id = model_index;
    int s_index = model_index * nranks_per_model;
    int e_index = MIN(s_index + nranks_per_model - 1,max_rank_ind);
    int stride  = 1;

    model->group_range[0][0] = s_index;
    model->group_range[0][1] = e_index;
    model->group_range[0][2] = stride;

    /* set MPI model ranks and communicators */
    mpi_range_group(global->mpi_comm, global->mpi_group, model, "Model-");

    DEBUG_MESG(
        printf("Model Group for rank = %d, model rank = %d: ",global->rank,model->rank);
        for(int i = s_index; i <= e_index; i+=stride) printf(" %d",i); printf("\n");
    )
}

void DLMG::form_data_groups(){
    int nranks_per_model = partition_info.nranks_per_model;
    compute_ctx_t *global = &ctx.global_compute_ctx;
    compute_ctx_t *data = &ctx.data_compute_ctx;

    /* ============================================ *
     * Data Parallelism MPI Rank Ranges:            *
     * The range for THIS global rank only include  *
     * the global ranks in my MY data group which   *
     * share the same model layers.                 *
     *                                              *
     * e.g. if nranks_per_model = 4, then           *
     *     data_group_0 = [0, 4,  8, ...]           *
     *     data_group_1 = [1, 5,  9, ...]           *
     *     data_group_2 = [2, 6, 10, ...]           *
     *     data_group_3 = [3, 7, 11, ...]           *
     * ============================================ */
    /* set ngpus seen from this rank */
    data->ngpus = global->ngpus;

    /* model group ranges */
    data->group_id = global->rank % nranks_per_model;
    int s_index = global->rank % nranks_per_model;
    int e_index = global->nranks - 1;
    int stride  = nranks_per_model;

    data->group_range[0][0] = s_index;
    data->group_range[0][1] = e_index;
    data->group_range[0][2] = stride;

    mpi_range_group(global->mpi_comm, global->mpi_group, data, "Data-");

    DEBUG_MESG(
        printf("Data Group for rank = %d: ",global->rank);
        for(int i = s_index; i <= e_index; i+=stride) printf(" %d",i); printf("\n");
    )
}