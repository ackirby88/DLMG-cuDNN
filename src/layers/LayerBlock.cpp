/**
 * File:   LayerBlock.cpp
 * Author: akirby
 *
 * Created on April 21, 2020, 8:34 PM
 */

#include "LayerBlock.h"

#define MIN(x,y)  (x)<(y) ? (x):(y)
#define MAX(x,y)  (x)>(y) ? (x):(y)

#define LAYER_LOOP \
    for(int i = 0; i < layers.size(); ++i)

#define REVERSE_LAYER_LOOP \
    for(int i = layers.size()-1; i >= 0; --i)

/* ============ */
/* Constructors */
/* ============ */
LayerBlock::LayerBlock(){
    checkCudaErrors(cudaStreamCreate(&cudaStream));

    // set cudaStream properties
    //FIXME
}

LayerBlock::LayerBlock(cudaStream_t cudaStream){
    this->cudaStream = cudaStream;
}

LayerBlock::~LayerBlock(){}

/* ============= */
/* Class Methods */
/* ============= */
size_t LayerBlock::setFwdTensors(cudnnTensorDescriptor_t **tensor, const int nsamples){
    size_t work_size;
    size_t workspace = 0;

    LAYER_LOOP {
        work_size = layers[i]->setFwdTensors(*tensor,nsamples);
        *tensor = layers[i]->getTensorDesc();
        workspace = MAX(workspace,work_size);
    }
    return workspace;
}

size_t LayerBlock::setBwdTensors(cudnnTensorDescriptor_t **tensor){
    size_t work_size;
    size_t workspace = 0;

    LAYER_LOOP {
        work_size = layers[i]->setBwdTensors(*tensor);
        *tensor = layers[i]->getTensorDesc();
        workspace = MAX(workspace,work_size);
    }
    return workspace;
}

void LayerBlock::setOneVector(Real *d_onevec){
    LAYER_LOOP {
        layers[i]->setOneVector(d_onevec);
    }
}

void LayerBlock::setCudaStreams(){
    LAYER_LOOP {
        layers[i]->setCudaStream(&cudaStream);
    }
}

void LayerBlock::setCudaStreams(cudaStream_t *cudaStream){
    /* propagate handles to each layer in block */
    this->cudaStream = *cudaStream;
    setCudaStreams();
}

cudaStream_t LayerBlock::getCudaStream(){
    return cudaStream;
}

void LayerBlock::allocateLayers(){
    allocateLayersHost();
    allocateLayersDevice(this->gpu_id);
}

void LayerBlock::allocateLayersHost(){
    /* allocate layers on CPU */
    LAYER_LOOP {
        layers[i]->layerAllocateHost();
    }
}

void LayerBlock::allocateLayersDevice(char multigrid_flag){
    checkCudaErrors(cudaSetDevice(this->gpu_id));

    /* allocate layers on GPU */
    LAYER_LOOP {
        layers[i]->layerAllocateDevice(this->gpu_id,multigrid_flag);
    }
}

void LayerBlock::allocateLayersDevice(const int gpu_id,char multigrid_flag){
    this->gpu_id = gpu_id;
    checkCudaErrors(cudaSetDevice(gpu_id));

    /* allocate layers on GPU */
    LAYER_LOOP {
        layers[i]->layerAllocateDevice(gpu_id,multigrid_flag);
    }
}

void LayerBlock::parametersInitializeHost(std::mt19937 &gen){
    LAYER_LOOP {
        layers[i]->parametersInitializeHost(gen);
    }
}

void LayerBlock::parametersDeviceToHost(){
    LAYER_LOOP {
        layers[i]->parametersDeviceToHost();
    }
}

void LayerBlock::parametersHostToDevice(){
    LAYER_LOOP {
        layers[i]->parametersHostToDevice();
    }
}

void LayerBlock::parametersHostToDeviceAsync(){
    LAYER_LOOP {
        layers[i]->parametersHostToDeviceAsync();
    }
}

data_t* LayerBlock::fwd(data_t *data,const int nsamples,char add_source){
    data_t *output = data;

    LAYER_LOOP {
        output = layers[i]->fwd(output,nsamples,add_source);
    }
    return output;
}

void LayerBlock::f_relaxation(const int nsamples,char add_source){
    data_t *output = layers[0]->getOutDevice();
    for(int i = 1; i < layers.size(); ++i) {
        output = layers[i]->fwd(output,nsamples,add_source);
    }
}

Real* LayerBlock::bwd(Real *adjoint_prev){
    Real *adjoint = adjoint_prev;

    REVERSE_LAYER_LOOP {
        adjoint = layers[i]->bwd(adjoint);
    }
    return adjoint;
}

void LayerBlock::updateWeights(Real learning_rate){
    LAYER_LOOP {
        layers[i]->updateWeights(learning_rate);
    }
}

Real LayerBlock::residual_norm(int nsamples){
    Real norm = 0.0;
    LAYER_LOOP {
        norm += layers[i]->residual_norm_sqrd(nsamples);
    }
    return norm;
}

bool LayerBlock::fromFile(const char *fileprefix){
    LAYER_LOOP {
        layers[i]->fromFile(fileprefix);
    }
}

void LayerBlock::toFile(const char *fileprefix){
    LAYER_LOOP {
        layers[i]->toFile(fileprefix);
    }
}