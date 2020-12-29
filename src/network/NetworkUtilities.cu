/**
 * File:   NetworkUtilities.cpp
 * Author: akirby
 *
 * Created on April 23, 2020, 12:37 PM
 */

/* header files */
#include "Network.h"
#include <unistd.h>

#define MIN(x,y)  (x)<(y) ? (x):(y)
#define MAX(x,y)  (x)>(y) ? (x):(y)

#define GLOBAL_LOOP(level) \
    for (int i = 0; i < globalLayers[level].size(); ++i)

#define BLOCK_LOOP(level) \
    for (int i = 0; i < blocks[level].size(); ++i)

#define REVERSE_BLOCK_LOOP(level) \
    for (int i = blocks[level].size() - 1; i >= 0; --i)

#define ROOT if(global_rank == 0)

void Network::parametersInitializeHost(std::mt19937 &gen){
    /* initialize all global layers for model parallelism consistency  */
    GLOBAL_LOOP(0) {
        globalLayers[0][i]->parametersInitializeHost(gen);
    }
}

void Network::parametersHostToDeviceAsync(int level){
    /* copy only */
    BLOCK_LOOP(level) {
        blocks[level][i]->parametersHostToDeviceAsync();
    }
}

void Network::setFwdTensors(int nsamples,int level){
    cudnnTensorDescriptor_t *tensor = &dataTensor;

    GLOBAL_LOOP(level){
        globalLayers[level][i]->setFwdTensors(tensor,nsamples);
        tensor = globalLayers[level][i]->getTensorDesc();
    }
}

void Network::setBwdTensors(int level){
    cudnnTensorDescriptor_t *tensor = &dataTensor;

    /* ====================== */
    /* Build Backward Tensors */
    /* ====================== */
    /* forward pass through layers */
    GLOBAL_LOOP(level){
        globalLayers[level][i]->setBwdTensors(tensor);
        tensor = globalLayers[level][i]->getTensorDesc();
    }
}

void Network::setTensors(int nsamples,int level){
    setFwdTensors(nsamples,level);
    setBwdTensors(level);
}

void Network::setNetworkBuffers(int nsamples,int level){
    /* set tensors and calculate workspaceSize */
    setTensors(nsamples,level);

    /* set CUDNN workspace and unit vector pointers */
    BLOCK_LOOP(level) {
        blocks[level][i]->setOneVector(d_onevec);
    }
}

Real* Network::getOutHost(int level){
    return this->h_out[level];
}

Real* Network::getAdjointHost(int level){
    return this->h_adjoint[level];
}

Real* Network::getAdjointDevice(int level){
    LayerBlock *first_block = blocks[level].front();
    Layer *first_layer = first_block->layers.front();
    return first_layer->getAdjoint();
}

int Network::getInSize(int level){
    LayerBlock *first_block = blocks[level].front();
    Layer *first_layer = first_block->layers.front();
    return first_layer->getInSize();
}

int Network::getOutSize(int level){
    LayerBlock *last_block = blocks[level].back();
    Layer *last_layer = last_block->layers.back();
    return last_layer->getOutSize();
}

data_t* Network::getOutDevice(int level){
    LayerBlock *last_block = blocks[level].back();
    Layer *last_layer = last_block->layers.back();
    return last_layer->getOutDevice();
}

cudnnTensorDescriptor_t* Network::getOutTensorDesc(int level){
    LayerBlock *last_block = blocks[level].back();
    Layer *last_layer = last_block->layers.back();
    return last_layer->getTensorDesc();
}

void Network::synchronizeNetwork(int level){
    int sind = 0;
    BLOCK_LOOP(level) {
        std::vector<Layer *> &layers = blocks[level][i]->layers;
        for (int l = 0; l < layers.size(); ++l) {
            cudaStream_t *streamID = layers[l]->getCudaStreamAddress();
            asyncCudaStreams[sind++] = streamID;

            /* reset block CUDA stream to synchronous version */
            layers[l]->setCudaStream(&syncCudaStream);
        }
    }
}

void Network::asynchronizeNetwork(int level){
    int sind = 0;
     BLOCK_LOOP(level) {
        std::vector<Layer *> &layers = blocks[level][i]->layers;
        for (int l = 0; l < layers.size(); ++l) {
            layers[l]->setCudaStream(asyncCudaStreams[sind++]);
        }
    }
}

void Network::display_mnist(Real *image){
    Real val;
    for (int j = 0; j < 28; j++) {
        for (int i = 0; i < 28; i++) {
            val = image[28*j + i];
            if (val > 0.0) {
                printf("\033[1;96m");
                printf("%1.1f ", image[28*j + i]);
                printf("\033[0m");
            } else {
                printf("   ");
            }
        }
        printf("\n");
    }
}

void Network::display_layer(Real *image,int nsamples){
    int wrap = nsamples/28;
    Real val;
    for (int j = 0; j < wrap; j++) {
        for (int i = 0; i < 28; i++) {
            val = image[28*j + i];
            if (val > 0.0001) {
                printf("\033[1;96m");
                printf("%1.1f ", image[28*j + i]);
                printf("\033[0m");
            } else if(val < -0.0001) {
                printf("\033[1;93m");
                printf("%1.1f ", image[28*j + i]);
                printf("\033[0m");
            } else {
                printf("\033[1;91m");
                printf("%1.1f ", image[28*j + i]);
                printf("\033[0m");
            }
        }
        printf("\n");
    }
}

void Network::displayLayerOutput(Real *d_output,int layer_id,int nsamples){
    Real *host_output = (Real *) malloc(nsamples*sizeof(Real));
    checkCudaErrors(cudaMemcpy(host_output,
                               d_output,
                               sizeof(Real)*nsamples,
                               cudaMemcpyDeviceToHost));

    printf("Rank[%d]: Layer[%d]: \n",model_rank,layer_id);
    display_layer(host_output,nsamples);
    free(host_output);
}

void Network::displayFwdOutput(int nsamples,int level){
    if(model_rank == model_nranks - 1) {
        data_t *output = getOutDevice(level);
        checkCudaErrors(cudaMemcpy(h_out[level],
                                   output->ptr,
                                   sizeof(Real)*getOutSize(level)*batch_size,
                                   cudaMemcpyDeviceToHost));

        printf("Rank[%d]: Network Out: \n",model_rank);
        Real *out = h_out[level];
        for(int k = getOutSize(level)*batch_size - nsamples; k < getOutSize(level)*batch_size; ++k){
            printf("out[%d]: %.15f\n",k,out[k]);
        }
    }
}

void Network::displayBwdOutput(int nsamples,int level){
    if(model_rank == 0) {
        Real *adjoint = getAdjointDevice(level);
        checkCudaErrors(cudaMemcpy(h_adjoint[level],
                                   adjoint,
                                   sizeof(Real)*getInSize(level)*batch_size,
                                   cudaMemcpyDeviceToHost));

        printf("\nRank[%d]: Network Adjoint: \n",model_rank);
        Real *adj = h_adjoint[level];
        for(int k = 0; k < nsamples; ++k){
            printf("adjoint[%d]: ",k);
            if(adj[k] > 0.0) printf(" ");
            printf(" %.15e\n",adj[k]);
        }
    }
}

void Network::barrier(const char *str, int sleep_sec){
    printf("Rank[%d]: %s\n",model_rank,str);
    sleep(sleep_sec);
    MPI_Barrier(MPI_COMM_WORLD);
}