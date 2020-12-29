/**
 * File:   LayerBlock.h
 * Author: akirby
 *
 * Created on April 21, 2020, 8:34 PM
 */

#ifndef LAYERBLOCK_H
#define LAYERBLOCK_H

/* system header files */
#include "mpi.h"
#include <vector>

/* header files */
#include "precision_types.h"
#include "Layer.h"
#include "CudaHelper.h"

#ifdef __cplusplus
extern "C" {
#endif

class LayerBlock {
  public:
    cudaStream_t cudaStream;
    MPI_Comm mpi_comm;
    int mpi_rank;
    int gpu_id;

    std::vector<Layer *> layers;

    /* ============ */
    /* Constructors */
    /* ============ */
    LayerBlock();
    LayerBlock(cudaStream_t cudaStream);
   ~LayerBlock();

   /* ============= */
   /* Class Methods */
   /* ============= */
    size_t setFwdTensors(cudnnTensorDescriptor_t **tensor, const int nsamples);
    size_t setBwdTensors(cudnnTensorDescriptor_t **tensor);

    void setOneVector(Real *d_onevec);
    void setCudaStreams();
    void setCudaStreams(cudaStream_t *cudaStream);
    cudaStream_t getCudaStream();

    void allocateLayers();
    void allocateLayersHost();
    void allocateLayersDevice(char multigrid_flag);
    void allocateLayersDevice(const int gpu_id,char multigrid_flag);

    void parametersInitializeHost(std::mt19937 &gen);
    void parametersDeviceToHost();
    void parametersHostToDevice();
    void parametersHostToDeviceAsync();

    void f_relaxation(const int nsamples,char add_source);
    data_t* fwd(data_t *data, const int nsamples,char add_source);
    Real* bwd(Real *adjoint_prev);
    void updateWeights(Real learning_rate);
    Real residual_norm(int nsamples);

    bool fromFile(const char *fileprefix);
    void toFile(const char *fileprefix);
};


#ifdef __cplusplus
}
#endif
#endif /* LAYERBLOCK_H */