/**
 * File:   Network.cu
 * Author: akirby
 *
 * Created on April 23, 2020, 12:37 PM
 */

/* header files */
#include "Network.h"
#include <unistd.h>

#define MIN(x,y)  (x)<(y) ? (x):(y)
#define MAX(x,y)  (x)>(y) ? (x):(y)

#define REVERSE_BLOCK_LOOP(level) \
    for (int i = blocks[(level)].size() - 1; i >= 0; --i)

#define ROOT if(global_rank == 0)

void Network::bwd_sync(Real *d_adjoint,int level){
    compute_ctx_t *model = &dlmg.ctx.model_compute_ctx;

    /* receive adjoint from next model rank */
    if(model_rank != model_nranks - 1){
        MPI_Status status;

        /* receive data on CPU */
        Real *host_ghost_data = rightGhost[level]->getAdjointHost();
        MPI_Recv(host_ghost_data,
                 rightGhost[level]->getInSize()*batch_size,
                 MPI_DLREAL,
                 model_rank+1,
                 0,
                 model->mpi_comm,
                 &status);

        /* copy data from CPU to GPU */
        checkCudaErrors(cudaMemcpy(d_adjoint,
                                   host_ghost_data,
                                   sizeof(Real)*rightGhost[level]->getInSize()*batch_size,
                                   cudaMemcpyHostToDevice));
    }

    /* solve */
    REVERSE_BLOCK_LOOP(level) {
        d_adjoint = blocks[level][i]->bwd(d_adjoint);
    }

    /* send adjoint to previous model rank */
    if(model_rank != 0){
        Real *host_adjoint = getAdjointHost(level);

        /* copy data from GPU to CPU */
        checkCudaErrors(cudaMemcpy(host_adjoint,
                                   d_adjoint,
                                   sizeof(Real)*getInSize(level)*batch_size,
                                   cudaMemcpyDeviceToHost));

        /* send data on CPU */
        MPI_Send(host_adjoint,
                 getInSize(level)*batch_size,
                 MPI_DLREAL,
                 model_rank-1,
                 0,
                 model->mpi_comm);
    }
}