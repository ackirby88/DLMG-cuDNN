/**
 * File:   GhostLayer.cpp
 * Author: akirby
 *
 * Created on April 30, 2020, 1:54 PM
 */

/* header files */
#include "GhostLayer.h"

/* ========================================================================== */
/* ============ */
/* Constructors */
/* ============ */
GhostLayerLeft::GhostLayerLeft(){
    this->h_out = {nullptr,nullptr};
    this->d_out = {nullptr,nullptr};
    this->noutputs = 0;
}

GhostLayerLeft::GhostLayerLeft(Layer *layer,int gpu_id) :
    GhostLayerLeft()
{
    this->gpu_id = gpu_id;
    this->noutputs = layer->getOutSize();
    this->batch_size = layer->getBatchSize();
    this->layer_type = layer->getLayerType();

    /* allocate host and device storage */
    this->h_out.ptr = (Real *) malloc(noutputs*batch_size*sizeof(Real));

    checkCudaErrors(cudaSetDevice(this->gpu_id));
    checkCudaErrors(cudaMalloc(&this->d_out.ptr, noutputs*batch_size*sizeof(Real)));
    this->d_out.tensorDesc = layer->getTensorDesc();
}

GhostLayerLeft::~GhostLayerLeft()
{
    checkCudaErrors(cudaSetDevice(this->gpu_id));
    if(this->h_out.ptr) free(this->h_out.ptr); this->h_out.ptr = nullptr;
    if(this->d_out.ptr) checkCudaErrors(cudaFree((this->d_out.ptr))); this->d_out.ptr = nullptr;
}

/* ================== */
/* GhostLayer Methods */
/* ================== */
cudnnTensorDescriptor_t* GhostLayerLeft::getTensorDesc(){
    return &tensorDesc;
}

void GhostLayerLeft::displayLayerType(){
    switch (this->layer_type) {
        case FULLYCONNECTED:
            printf("GhostLayerLeft Type: Fully Connected\n");
            break;
        case CONVOLUTION:
            printf("GhostLayerLeft Type: Convolution\n");
            break;
        case MAXPOOL:
            printf("GhostLayerLeft Type: max Pool\n");
            break;
        case SOFTMAX:
            printf("GhostLayerLeft Type: Softmax\n");
            break;
    }
}
/* ========================================================================== */

/* ========================================================================== */
/* ============ */
/* Constructors */
/* ============ */
GhostLayerRight::GhostLayerRight(){
    this->h_adjoint = nullptr;
    this->d_adjoint = nullptr;
    this->ninputs = 0;
}

GhostLayerRight::GhostLayerRight(Layer *layer,int gpu_id) :
    GhostLayerRight()
{
    this->gpu_id = gpu_id;
    this->ninputs = layer->getInSize();
    this->batch_size = layer->getBatchSize();
    this->layer_type = layer->getLayerType();

    /* allocate host and device storage */
    this->h_adjoint = (Real *) malloc(ninputs*batch_size*sizeof(Real));

    checkCudaErrors(cudaSetDevice(this->gpu_id));
    checkCudaErrors(cudaMalloc(&this->d_adjoint, ninputs*batch_size*sizeof(Real)));
}

GhostLayerRight::~GhostLayerRight()
{
    checkCudaErrors(cudaSetDevice(this->gpu_id));
    if(this->h_adjoint) free(this->h_adjoint); this->h_adjoint = nullptr;
    if(this->d_adjoint) checkCudaErrors(cudaFree(this->d_adjoint)); this->d_adjoint = nullptr;
}

/* ================== */
/* GhostLayer Methods */
/* ================== */
void GhostLayerRight::displayLayerType(){
    switch (layer_type) {
        case FULLYCONNECTED:
            printf("GhostLayer Type: Fully Connected\n");
            break;
        case CONVOLUTION:
            printf("GhostLayer Type: Convolution\n");
            break;
        case MAXPOOL:
            printf("GhostLayer Type: max Pool\n");
            break;
        case SOFTMAX:
            printf("GhostLayer Type: Softmax\n");
            break;
    }
}
/* ========================================================================== */