/**
 * File:   Layer.cpp
 * Author: akirby
 *
 * Created on April 3, 2020, 1:59 PM
 */

#include "Layer.h"

/* ============ */
/* Constructors */
/* ============ */
Layer::Layer(){
    this->out = {nullptr,nullptr};
    this->adjoint = nullptr;
    this->ninputs = 0;
    this->noutputs = 0;
    this->workspaceFwd = nullptr;
    this->workspaceBwd = nullptr;

    fine_level_flag = 1;
    multigrid_flag = 0;
    this->A = nullptr;
    this->G = nullptr;
    this->R = nullptr;
}

Layer::Layer(int layer_type) : Layer(){
    this->layer_type = layer_type;

    checkCudaErrors(cublasCreate(&cudaHandles.cublasHandle));
    checkCUDNN(cudnnCreate(&cudaHandles.cudnnHandle));
}

Layer::~Layer(){}

/* ============= */
/* Layer Methods */
/* ============= */
Real Layer::residual_norm_sqrd(int nsamples){
    Real norm = 0;
    g_blas_nrm2(this->cudaHandles.cublasHandle,
                this->noutputs*nsamples,
                this->R,
                1,
                &norm);
    return (norm*norm);
}

void Layer::setOneVector(Real *d_onevec){
    this->onevec = d_onevec;
}

void Layer::setCudaStream(cudaStream_t *cudaStream){
    this->streamID = cudaStream;
    checkCudaErrors(cublasSetStream(cudaHandles.cublasHandle, streamID[0]));
    checkCUDNN(cudnnSetStream(cudaHandles.cudnnHandle, streamID[0]));
}

void print_layer(Real *image,int nsamples){
    int wrap = nsamples/28;
    Real val;
    for (int j = 0; j < wrap; j++) {
        for (int i = 0; i < 28; i++) {
            val = image[28*j + i];
            if (val > 0.0001) {
                printf("\033[1;96m");
                printf("% 1.5f ", image[28*j + i]);
                printf("\033[0m");
            } else if(val < -0.0001) {
                printf("\033[1;93m");
                printf("%1.5f ", image[28*j + i]);
                printf("\033[0m");
            } else {
                printf("\033[1;90m");
                printf("% 1.1e ", image[28*j + i]);
                printf("\033[0m");
            }
        }
        printf("\n");
    }
}

void Layer::displayLayer(Real *d_output,int nsamples,int ndisp){
    Real *host_output = (Real *) malloc(nsamples*sizeof(Real));

    checkCudaErrors(cudaMemcpy(host_output,
                               d_output,
                               sizeof(Real)*nsamples,
                               cudaMemcpyDeviceToHost));

    printf("Layer[%d]: \n",global_idx); print_layer(host_output,ndisp);
    free(host_output);
}

void Layer::displayLayerType(){
    switch (layer_type) {
        case FULLYCONNECTED:
            printf("Layer Type: Fully Connected, global index: %d, # params: %d\n",global_idx,parametersGetCount());
            break;
        case CONVOLUTION:
            printf("Layer Type: Convolution, global index: %d, # params: %d\n",global_idx,parametersGetCount());
            break;
        case MAXPOOL:
            printf("Layer Type: Max Pool, global index: %d, # params: %d\n",global_idx,parametersGetCount());
            break;
        case SOFTMAX:
            printf("Layer Type: Softmax, global index: %d, # params: %d\n",global_idx,parametersGetCount());
            break;
    }
}