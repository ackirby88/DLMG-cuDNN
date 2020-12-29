/**
 * File:   MaxPoolLayer.h
 * Author: akirby
 *
 * Created on April 3, 2020, 12:41 PM
 */

#ifndef MAXPOOLLAYER_H
#define MAXPOOLLAYER_H

/* system header files */
#include <vector>
#include <string>
#include <sstream>

/* header files */
#include "gpu_kernel_defines.h"
#include "Layer.h"
#include "CudaHelper.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Max-pooling layer.
 */
class MaxPoolLayer : public Layer {
  public:
    int in_channels;
    int in_width;
    int in_height;

    int out_channels;
    int out_width;
    int out_height;

    int size;
    int stride;

    /* ======================== */
    /* device data storage: d_? */
    /* ======================== */
    /* layer states */
    Real *d_z;

    /* ===================== */
    /* CUDNN data structures */
    /* ===================== */
    cudnnPoolingDescriptor_t poolDesc;

    /* ============ */
    /* Constructors */
    /* ============ */
    MaxPoolLayer(int batch_size_,
                 int in_channels_,
                 int in_width_,
                 int in_height_,
                 int size_,
                 int stride_);
    MaxPoolLayer(const MaxPoolLayer &layer);
   ~MaxPoolLayer();

    Layer *clone();
    void layerAllocateHost(){};
    void layerAllocateDevice(const int gpu_id,char multigrid_flag);

    /* ============= */
    /* Class Methods */
    /* ============= */
    data_t* formA(data_t *data,const int nsamples);
    data_t* fwd(const data_t *data, const int nsamples, const char add_source);
    Real* bwd(Real *adjoint);

    size_t setFwdTensors(cudnnTensorDescriptor_t *srcTensorDesc, const int nsamples);
};

#ifdef __cplusplus
}
#endif
#endif /* MAXPOOLLAYER_H */