/**
 * File:   SoftmaxLayer.h
 * Author: akirby
 *
 * Created on April 15, 2020, 2:55 PM
 */

#ifndef SOFTMAXLAYER_H
#define SOFTMAXLAYER_H

/* system header files */
#include <vector>
#include <string>
#include <sstream>

/* header files */
#include "dlmg_types.h"
#include "precision_types.h"
#include "gpu_kernel_defines.h"
#include "math_utilities.h"

#include "Layer.h"
#include "CudaHelper.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  layer.
 */
class SoftmaxLayer : public Layer {
  public:
    int in_channels;
    int out_channels;

    /* ======================== */
    /* device data storage: d_? */
    /* ======================== */
    /* layer states */
    Real *d_z;

    /* ============ */
    /* Constructors */
    /* ============ */
    SoftmaxLayer(int batch_size_, int in_channels_);
    SoftmaxLayer(const SoftmaxLayer &layer);
   ~SoftmaxLayer();

    Layer *clone();
    void layerAllocateHost(){};
    void layerAllocateDevice(const int gpu_id,char multigrid_flag);

    /* ============= */
    /* Class Methods */
    /* ============= */
    data_t* formA(data_t *data,const int nsamples);
    data_t* fwd(const data_t *data, const int nsamples, const char add_source);
    Real* bwd(Real *labels);
};

#ifdef __cplusplus
}
#endif
#endif /* SOFTMAXLAYER_H */