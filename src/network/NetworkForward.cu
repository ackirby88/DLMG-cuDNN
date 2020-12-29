/**
 * File:   NetworkForward.cu
 * Author: akirby
 *
 * Created on April 23, 2020, 12:37 PM
 */

/* header files */
#include "Network.h"
#include <unistd.h>

#define MIN(x,y)  (x)<(y) ? (x):(y)
#define MAX(x,y)  (x)>(y) ? (x):(y)

#define BLOCK_LOOP(level) \
    for (int i = 0; i < blocks[(level)].size(); ++i)

#define ROOT if(global_rank == 0)

void Network::fwd_sync(timerdata_t *time_data,data_t *data,const int nsamples,int level,char add_source){
    compute_ctx_t *model = &dlmg.ctx.model_compute_ctx;
    data_t *output;

    checkCudaErrors(cudaDeviceSynchronize());
    double t1 = mpi_timer();
    /* receive last layer output from previous model rank */
    if(model_rank !=  0){
        MPI_Status status;

        /* receive data on CPU */
        data = leftGhost[level]->getOutDevice();

        data_t *host_ghost_data = leftGhost[level]->getOutHost();
        MPI_Recv(host_ghost_data->ptr,
                 getInSize(level)*nsamples,
                 MPI_DLREAL,
                 model_rank-1,
                 0,
                 model->mpi_comm,
                 &status);

        /* copy data from CPU to GPU */
        checkCudaErrors(cudaMemcpy(data->ptr,
                                   host_ghost_data->ptr,
                                   sizeof(Real)*getInSize(level)*nsamples,
                                   cudaMemcpyHostToDevice));
    }
    double t2 = mpi_timer();
    time_data->comm_time += t2-t1;

    /* solve */
    t1 = mpi_timer();
    BLOCK_LOOP(level){
        output = blocks[level][i]->fwd(data,nsamples,add_source);
        data = output;
    }
    t2 = mpi_timer();
    double comp_time = t2-t1;

    /* send last layer output to next model rank */
    t1 = mpi_timer();
    if(model_rank != model_nranks - 1){
        /* copy data from GPU to CPU */
        Real *host_out = getOutHost(level);
        checkCudaErrors(cudaMemcpy(host_out,
                                   output->ptr,
                                   sizeof(Real)*getOutSize(level)*nsamples,
                                   cudaMemcpyDeviceToHost));

        /* send data on CPU */
        MPI_Send(host_out,
                 getOutSize(level)*nsamples,
                 MPI_DLREAL,
                 model_rank+1,
                 0,
                 model->mpi_comm);
    }
    t2 = mpi_timer();
    time_data->comm_time += t2-t1;
    checkCudaErrors(cudaDeviceSynchronize());

    double comp_time_total = 0.0;
    MPI_Allreduce(&comp_time, &comp_time_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    time_data->comp_time += comp_time_total;
    double total_loc = time_data->comm_time + time_data->comp_time;
//    printf(" TIME BREAKDOWN rank[%d]: total = %f, computation = %f (%.1f\%), communication = %f (%.1f\%)\n",global_rank,
//            total_loc,
//            time_data->comp_time,time_data->comp_time/total_loc*100.0,
//            time_data->comm_time,time_data->comm_time/total_loc*100.0);

}