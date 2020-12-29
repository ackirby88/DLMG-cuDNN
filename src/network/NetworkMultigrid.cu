/**
 * File:   NetworkMultigrid.cu
 * Author: akirby
 *
 * Created on May 20, 2020, 2:46 PM
 */

/* system header files */
#include <unistd.h>
#include <omp.h>
#include "nvToolsExt.h"

/* header files */
#include "Network.h"
#include "Layer.h"

#define MIN(x,y)  (x)<(y) ? (x):(y)
#define MAX(x,y)  (x)>(y) ? (x):(y)

#define BLOCK_LOOP(sind,level) \
    for (int i = (sind); i < blocks[(level)].size(); ++i)

#define REVERSE_BLOCK_LOOP(sind,level) \
    for (int i = blocks[(level)].size() - 1; i >= (sind); --i)

#define ROOT if(global_rank == 0)

void Network::Multigrid_reset_states(int nsamples,int level){
#pragma omp parallel for num_threads(blocks[(level)].size())
    BLOCK_LOOP(0,level) {
        std::vector<Layer *> &layers = blocks[level][i]->layers;
        for (int l = 0; l < layers.size(); ++l) {
            data_t *output = layers[l]->getOutDevice();
            int noutputs = layers[l]->getOutSize();

            g_blas_memset_async(output->ptr,
                                0,
                                sizeof(Real)*nsamples*noutputs,
                                layers[l]->getCudaStream());
        }
    }
}
/* ========================================================================== */
/* SMOOTHERS                                                                  */
/* ========================================================================== */
void Network::Multigrid_F_relaxation(int nsamples,int level,char add_source){
//    nvtxRangePushA("F_relaxation");

    /* forward propagate only within each block's layers */
#pragma omp parallel for num_threads(blocks[level].size()) firstprivate(nsamples,level)
    BLOCK_LOOP(0,level) {
        //int tid = omp_get_thread_num();
        //printf("Hello world from omp thread %d\n", tid);

        cudaSetDevice(gpu_id);
        blocks[level][i]->f_relaxation(nsamples,add_source);
    }
//    nvtxRangePop();
}

void Network::Multigrid_CF_relaxation(data_t *data,int nsamples,int level,char add_source){
//    nvtxRangePushA("CF_relaxation");

    /* forward propagate only within each block's layers */
#pragma omp parallel for num_threads(blocks[(level)].size()) firstprivate(data,nsamples,level)
    BLOCK_LOOP(0,level) {
        cudaSetDevice(gpu_id);
        if (i != 0) {
            Layer *last_layer = blocks[level][i-1]->layers.back();
            data = last_layer->getOutDevice();

            /* CF-relaxation */
            blocks[level][i]->fwd(data,nsamples,add_source);
        } else {
            blocks[level][i]->fwd(data,nsamples,add_source);
        }
    }
//    nvtxRangePop();
}

void Network::Multigrid_communicate_ghost(data_t **data,int nsamples,int level){
    compute_ctx_t *model = &dlmg.ctx.model_compute_ctx;
    MPI_Request request[2];
    int nrequest = 0;

    /* Communicate */
    /* ====================================================================== */
    /* receive last layer output from previous model rank */
    if(model_rank !=  0){
        *data = leftGhost[level]->getOutDevice();

        data_t *host_ghost_data = leftGhost[level]->getOutHost();
        MPI_Irecv(host_ghost_data->ptr,
                  getInSize(level)*nsamples,
                  MPI_DLREAL,
                  model_rank-1,
                  0,
                  model->mpi_comm,
                  &request[nrequest++]);
    }

    /* send last layer output to next model rank */
    if(model_rank != model_nranks - 1){
        Real *host_out = getOutHost(level);
        data_t *output = getOutDevice(level);

        checkCudaErrors(cudaMemcpy(host_out,
                                   output->ptr,
                                   sizeof(Real)*getOutSize(level)*nsamples,
                                   cudaMemcpyDeviceToHost));

        /* send data on CPU */
        MPI_Isend(host_out,
                  getOutSize(level)*nsamples,
                  MPI_DLREAL,
                  model_rank+1,
                  0,
                  model->mpi_comm,
                  &request[nrequest++]);
    }
    MPI_Waitall(nrequest,request,MPI_STATUSES_IGNORE);

    if(model_rank !=  0){
        /* copy data from CPU to GPU */
        data_t *host_ghost_data = leftGhost[level]->getOutHost();
        checkCudaErrors(cudaMemcpy((*data)->ptr,
                                   host_ghost_data->ptr,
                                   sizeof(Real)*getInSize(level)*nsamples,
                                   cudaMemcpyHostToDevice));
    }
    /* ====================================================================== */
}
void Network::Multigrid_parCF_relaxation(timerdata_t *time_data,data_t *data,int nsamples,int level,char add_source){
    Layer *last_layer;
    data_t *output;

    /* Communicate: updates data pointer */
    double t1 = mpi_timer();
    Multigrid_communicate_ghost(&data,nsamples,level);
    double t2 = mpi_timer();
    time_data->comm_time += t2-t1;

    /* ====================================================================== */
    /* forward propagate only within each block's layers */
    t1 = mpi_timer();
#pragma omp parallel for num_threads(blocks[(level)].size()) firstprivate(data,output,last_layer,nsamples,level)
    BLOCK_LOOP(0,level) {
        checkCudaErrors(cudaSetDevice(gpu_id));

        if (i != 0) {
            last_layer = blocks[level][i-1]->layers.back();
            output = last_layer->getOutDevice();

            /* CF-relaxation */
            blocks[level][i]->fwd(output,nsamples,add_source);
        } else {
            /* data set above from ghost/input data */
            blocks[level][i]->fwd(data,nsamples,add_source);
        }
    }
    t2 = mpi_timer();
    time_data->comp_time += t2-t1;
}

/* ========================================================================== */
/* MULTIGRID OPERATORS                                                        */
/* ========================================================================== */
Real Network::Multigrid_residual(timerdata_t *time_data,int nsamples,int level){
//    nvtxRangePushA("Residual");

    compute_ctx_t *model = &dlmg.ctx.model_compute_ctx;
    Real block_norm_sum = 0.0;
    Real res_norm = 0.0;
    Real neg_one = -1.0;
    data_t *output;

    /* ======================== */
    /* Residual R := G - A(U,W) */
    /* ======================== */
    Real norm = 0.0;

    /* Communicate: updates data pointer */
    double t1 = mpi_timer();
    Multigrid_communicate_ghost(&output,nsamples,level);
    double t2 = mpi_timer();
    time_data->comm_time += t2-t1;

    t1 = mpi_timer();
    int start_block = (model->rank == 0) ? 1:0; /* skip block 0 on start of network */
    BLOCK_LOOP(start_block,level){
        std::vector<Layer *> &layers = blocks[level][i]->layers;

        if (i != 0) {
            Layer *prev_layer = blocks[level][i-1]->layers.back();
            output = prev_layer->getOutDevice();
        }

        int l = 0;
        if(layers[l]->getLayerType() == layerType::SOFTMAX) continue;
        cuda_info_t *cudaHandles = layers[l]->getCudaHandles();
        //checkCudaErrors(cudaStreamSynchronize(layers[l]->getCudaStream()));

        /* form A */
        layers[l]->formA(output,nsamples);
        output = layers[l]->getOutDevice();

        /* form R := G - A */
        // copy G into R
        g_blas_copy(cudaHandles->cublasHandle,
                    layers[l]->getOutSize()*nsamples,
                    layers[l]->getG(),1,
                    layers[l]->getR(),1);

        // R = G - A
        g_blas_axpy(cudaHandles->cublasHandle,
                    layers[l]->getOutSize()*nsamples,
                    &neg_one,
                    layers[l]->getA(),1,
                    layers[l]->getR(),1);

        // norm = |R|_L2
        g_blas_nrm2(cudaHandles->cublasHandle,
                    layers[l]->getOutSize()*nsamples,
                    layers[l]->getR(),
                    1,
                    &norm);

        //checkCudaErrors(cudaStreamSynchronize(layers[l]->getCudaStream()));
        if (isnan(norm)) {
            printf("MG Level[%d]: Layer norm NAN! ",level);
            layers[l]->displayLayerType();
            exit(EXIT_FAILURE);
        }
        //else {
        //    printf("Block[%d] Layer[%d]: norm = %.15f ",i,l,norm);
        //    layers[l]->displayLayerType();
        //}
        block_norm_sum += norm*norm;
    }
    t2 = mpi_timer();
    time_data->comp_time += t2-t1;

    MPI_Allreduce(&block_norm_sum,&res_norm,1,MPI_DLREAL,MPI_SUM,model->mpi_comm);
    res_norm = sqrt(res_norm)/(Real) (nsamples) / (Real) globalLayers_count;
//    nvtxRangePop();
    return res_norm;
}

//void Network::Multigrid_restriction(int nsamples,int level){
//    compute_ctx_t *model = &dlmg.ctx.model_compute_ctx;
//    data_t *f_soln;
//    data_t *c_soln;
//    Real *f_res;
//    Real *c_res;
//
//    Real one = 1.0;
//
//    int start_block = (model->rank == 0) ? 1:0; /* skip block 0 on start of network */
//    BLOCK_LOOP(0,level){
//        Layer *f_layer = blocks[level][i]->layers.front();
//        Layer *c_layer = f_layer->getCoarseLayer();
//
//        cuda_info_t *cudaHandles = c_layer->getCudaHandles();
//        checkCudaErrors(cudaStreamSynchronize(c_layer->getCudaStream()));
//
//        /* device pointers */
//        f_soln = f_layer->getOutDevice();
//        c_soln = c_layer->getOutDevice();
//
//        f_res = f_layer->getR();
//        c_res = c_layer->getR();
//
//        /* VC = c_soln: copy solution */
//        g_blas_copy(cudaHandles->cublasHandle,
//                    f_layer->getOutSize()*nsamples,
//                    f_soln->ptr,1,
//                    c_soln->ptr,1);
//
//        /* RC = c_res: copy residual */
//        g_blas_copy(cudaHandles->cublasHandle,
//                    f_layer->getOutSize()*nsamples,
//                    f_res,1,
//                    c_res,1);
//
//        /* form GC = A(UC,WC) + RC */
//        if (i) {
//            Layer *in_layer = blocks[level][i-1]->layers.front();
//            data_t *in_data = in_layer->getOutDevice();
//
//            // 1.) copy RC into GC -- formA overwrites c_res
//            g_blas_copy(cudaHandles->cublasHandle,
//                        c_layer->getOutSize()*nsamples,
//                        c_res,1,
//                        c_layer->getG(),1);
//
//            // 2.) A(UC,WC)
//            c_layer->formA(in_data,nsamples);
//
//            /* GC += AC */
//            g_blas_axpy(cudaHandles->cublasHandle,
//                        c_layer->getOutSize()*nsamples,
//                        &one,
//                        c_layer->getA(),1,
//                        c_layer->getG(),1);
//        } else {
//            // GC = UC
//            g_blas_copy(cudaHandles->cublasHandle,
//                        c_layer->getOutSize()*nsamples,
//                        c_soln->ptr,1,
//                        c_layer->getG(),1);
//        }
//    }
//}

void Network::Multigrid_restriction(timerdata_t *time_data,int nsamples,int flevel){
    data_t *f_soln;
    data_t *c_soln;
    Real *f_res;
    Real *c_res;
    data_t *output;
    Real one = 1.0;

    int clevel = flevel + 1;

    /* 1. restrict solution from fine to coarse first */
    double t1 = mpi_timer();
    BLOCK_LOOP(0,flevel){
        Layer *f_layer = blocks[flevel][i]->layers.front();
        Layer *c_layer = f_layer->getCoarseLayer();

        cuda_info_t *cudaHandles = c_layer->getCudaHandles();
        checkCudaErrors(cudaStreamSynchronize(c_layer->getCudaStream()));

        /* device pointers */
        f_soln = f_layer->getOutDevice();
        c_soln = c_layer->getOutDevice();

        /* VC = c_soln: copy solution */
        g_blas_copy(cudaHandles->cublasHandle,
                    f_layer->getOutSize()*nsamples,
                    f_soln->ptr,1,
                    c_soln->ptr,1);
    }
    double t2 = mpi_timer();
    time_data->comp_time += t2-t1;

    /* 2. Communicate solutions that were just restricted */
    t1 = mpi_timer();
    Multigrid_communicate_ghost(&output,nsamples,clevel);
    t2 = mpi_timer();
    time_data->comm_time += t2-t1;

    /* 3. Form coarse source term */
    t1 = mpi_timer();
    BLOCK_LOOP(0,flevel){
        Layer *f_layer = blocks[flevel][i]->layers.front();
        Layer *c_layer = f_layer->getCoarseLayer();

        cuda_info_t *cudaHandles = c_layer->getCudaHandles();
        checkCudaErrors(cudaStreamSynchronize(c_layer->getCudaStream()));

        /* device pointers */
        f_soln = f_layer->getOutDevice();
        c_soln = c_layer->getOutDevice();

        f_res = f_layer->getR();
        c_res = c_layer->getR();

        /* VC = c_soln: copy solution */
        g_blas_copy(cudaHandles->cublasHandle,
                    f_layer->getOutSize()*nsamples,
                    f_soln->ptr,1,
                    c_soln->ptr,1);

        /* RC = c_res: copy residual */
        g_blas_copy(cudaHandles->cublasHandle,
                    f_layer->getOutSize()*nsamples,
                    f_res,1,
                    c_res,1);

        /* form GC = A(UC,WC) + RC */
        if (i || model_rank != 0) {
            if (i) {
                Layer *in_layer = blocks[flevel][i-1]->layers.front();
                output = in_layer->getOutDevice();
            }

            // 1.) copy RC into GC -- formA overwrites c_res
            g_blas_copy(cudaHandles->cublasHandle,
                        c_layer->getOutSize()*nsamples,
                        c_res,1,
                        c_layer->getG(),1);

            // 2.) A(UC,WC)
            c_layer->formA(output,nsamples);

            /* GC += AC */
            g_blas_axpy(cudaHandles->cublasHandle,
                        c_layer->getOutSize()*nsamples,
                        &one,
                        c_layer->getA(),1,
                        c_layer->getG(),1);
        } else {
            // GC = UC
            g_blas_copy(cudaHandles->cublasHandle,
                        c_layer->getOutSize()*nsamples,
                        c_soln->ptr,1,
                        c_layer->getG(),1);
        }
    }
    t2 = mpi_timer();
    time_data->comp_time += t2-t1;
}

void Network::Multigrid_correction(int nsamples,int coarse_level){
    data_t *f_soln;
    data_t *c_soln;

    BLOCK_LOOP(0,coarse_level){
        std::vector<Layer *> &layers = blocks[coarse_level][i]->layers;

        for(int l = 0; l < layers.size(); ++l) {
            Layer *c_layer = layers[l];
            Layer *f_layer = c_layer->getFineLayer();

            cuda_info_t *cudaHandles = f_layer->getCudaHandles();
            //checkCudaErrors(cudaStreamSynchronize(c_layer->getCudaStream()));

            /* device pointers */
            c_soln = c_layer->getOutDevice();
            f_soln = f_layer->getOutDevice();

            /* ========================================== */
            /* Fine Layer Update:                         */
            /* U_f = U_f + alpha*(V_c - U_c)              */
            /*     U_f := fine layer solution             */
            /*     U_c := fine layer solution restriction */
            /*     V_c := coarse layer solution           */
            /* ------------------------------------------ */
            /* NOTE: restriction operator is a copy, thus */
            /* U_c = U_f --> U_f = U_f + alpha*(V_c - U_f)*/
            /* ========================================== */

            /* 1. form delta_U = (V_c - U_f), store in c_soln */
            Real neg_one = -1.0;
            g_blas_axpy(cudaHandles->cublasHandle,
                        f_layer->getOutSize()*nsamples,
                        &neg_one,
                        f_soln->ptr,1,
                        c_soln->ptr,1);

            Real norm;
            g_blas_nrm2(cudaHandles->cublasHandle,
                        f_layer->getOutSize()*nsamples,
                        c_soln->ptr,
                        1,
                        &norm);

            //printf("Delta U[%d]: %.15e\n",f_layer->getGlobalIdx(),norm);
            //layers[l]->displayLayer(c_soln->ptr,layers[l]->getOutSize()*nsamples,28*100);

            /* 2. update fine level solution U_f = U_f + alpha*delta_U */
            Real alpha = 1.0;
            g_blas_axpy(cudaHandles->cublasHandle,
                        f_layer->getOutSize()*nsamples,
                        &alpha,
                        c_soln->ptr,1,
                        f_soln->ptr,1);
        }

        /* required sync stream */
        //cudaStreamSynchronize(blocks[coarse_level][i]->getCudaStream());
    }
}

void Network::Multigrid_display(int nsamples,int level){
  // BLOCK_LOOP(0,level) {
   int i = 0; {
        std::vector<Layer *> &layers = blocks[level][i]->layers;

        for (int l = 0; l < layers.size(); ++l) {
            cuda_info_t *cudaHandles = layers[l]->getCudaHandles();
            checkCudaErrors(cudaStreamSynchronize(layers[l]->getCudaStream()));

            Real *dat = layers[l]->getOutDevice()->ptr;
            ConvolutionLayer *conv = (ConvolutionLayer *)layers[l];

            Real norm;
            g_blas_nrm2(cudaHandles->cublasHandle,
                        layers[l]->getInSize()*nsamples,
                        conv->in.ptr,
                        1,
                        &norm);
            printf("in: %.15f ",norm); layers[l]->displayLayer(conv->in.ptr,layers[l]->getInSize()*nsamples,28*100);
            printf("d_bias: "); layers[l]->displayLayer(conv->d_bias,conv->out_channels,conv->out_channels);
            printf("d_z: "); layers[l]->displayLayer(conv->d_z,layers[l]->getOutSize()*nsamples,28*100);
            printf("d_a: "); layers[l]->displayLayer(conv->d_a,layers[l]->getOutSize()*nsamples,28*100);
        }
    }
}

void Network::Multigrid_update_parameters(int level){
    BLOCK_LOOP(0,level){
        Layer *f_layer = blocks[level][i]->layers.front();
        Layer *c_layer = f_layer->getCoarseLayer();

        /* copy layer parameters: weights and bias */
        c_layer->parametersCopyLayerDevice(f_layer);
    }
}

/* ========================================================================== */
/* MULTIGRID FORWARD SOLVE                                                    */
/* ========================================================================== */
void Network::Multigrid_fwd(data_t *data,const int nsamples,int level,int trained_iterations,char add_source){
    multigrid_t &mg = dlmg.multigrid;
    int coarse_level = level+1;
    double t1,t2;

    timerdata_t time_data;
    time_data.comm_time = 0.0;
    time_data.comp_time = 0.0;
    time_data.total_time = 0.0;

    /* copy layer parameters: weights and bias */
    Multigrid_update_parameters(level);

    int iter = 0;
    int max_iter = 2;

    Real tol = 1.0E-8, rnorm = 1.0E2;
    while (rnorm > tol && iter < max_iter){

        /* 1. Pre-Smoothing: FCF-relation */
        nvtxRangePushA("parCF_relaxation");
        Multigrid_parCF_relaxation(&time_data,data,nsamples,level,add_source);
        nvtxRangePop();
        iter++;

        /* 2. Residual Errors: [resid = G - A(U,W)] */
        nvtxRangePushA("Residual");
        rnorm = Multigrid_residual(&time_data,nsamples,level);
        nvtxRangePop();

        ROOT{
            for(int ii = 0; ii < level; ii++) printf("  ");
            printf(" MG[%d][%3.d]: norm = %.15e\n",level,iter,rnorm);
        }
        if (rnorm < tol) {
            ROOT{
                for(int ii = 0; ii < level; ii++) printf("  ");
                printf(" Batch[%.3d] Level %2d Converged: %3d iterations, norm: %e\n",
                    trained_iterations,level,iter,rnorm);
                for(int ii = 0; ii < level; ii++) printf("  ");
                printf("+=============================================+\n");
            }
            return;
        }

        /* 3. Restriction */
        nvtxRangePushA("Restriction");
        Multigrid_restriction(&time_data,nsamples,level);
        nvtxRangePop();

        /* 4. Solve Coarse Level: A(VC,WC) = A(UC,WC) + RC =: GC for VC */
        nvtxRangePushA("Coarse-Solve");
        if(coarse_level == mg.nlevels-1){
            synchronizeNetwork(coarse_level);
                fwd_sync(&time_data,data,nsamples,coarse_level,1);
            asynchronizeNetwork(coarse_level);
        } else {
            Multigrid_fwd(data,nsamples,coarse_level,trained_iterations,1);
        }
        nvtxRangePop();

        /* 5. Correction */
        nvtxRangePushA("Correction");
        t1 = mpi_timer();
        Multigrid_correction(nsamples,coarse_level);
        t2 = mpi_timer();
        nvtxRangePop();
        time_data.comp_time += t2-t1;

        nvtxRangePushA("F_relaxation");
        t1 = mpi_timer();
        Multigrid_F_relaxation(nsamples,level,add_source);
        t2 = mpi_timer();
        nvtxRangePop();
        time_data.comp_time += t2-t1;
    }

    double total_loc = time_data.comm_time + time_data.comp_time;

    struct DataRank{
        double time;
        int mpi_rank;
    };

    DataRank loc_data, max_data;
    loc_data.time = total_loc;
    loc_data.mpi_rank = global_rank;

    MPI_Allreduce(&loc_data, &max_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    if(global_rank == max_data.mpi_rank){
        printf(" TIME BREAKDOWN rank[%d]: total = %f, computation = %f (%.1f\%), communication = %f (%.1f\%)\n",global_rank,
            total_loc,
            time_data.comp_time,time_data.comp_time/total_loc*100.0,
            time_data.comm_time,time_data.comm_time/total_loc*100.0);
    }
}