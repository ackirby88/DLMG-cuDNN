/**
 * File:   GhostGhostLayer.h
 * Author: akirby
 *
 * Created on April 30, 2020, 1:55 PM
 */

#ifndef GHOSTLAYER_H
#define GHOSTLAYER_H

/* header files */
#include "precision_types.h"
#include "Layer.h"
#include "CudaHelper.h"

#ifdef __cplusplus
extern "C" {
#endif

class GhostLayerLeft {
  protected:
    int layer_type; /**< layer identification tag */
    int batch_size; /**< layer batch size */
    int noutputs;   /**< layer output size */
    int gpu_id;     /**< layer gpu id */

    cudnnTensorDescriptor_t tensorDesc; /**< layer dnn tensor descriptor */
    data_t d_out;   /**< device layer output data (size: noutputs*batch_size) */
    data_t h_out;   /**< host   layer output data (size: noutputs*batch_size) */

    /* ==================== */
    /* Private Constructors */
    /* ==================== */
    GhostLayerLeft();

  public:
    enum layerType {
        FULLYCONNECTED,
        CONVOLUTION,
        MAXPOOL,
        SOFTMAX
    };

    /* ============ */
    /* Constructors */
    /* ============ */
    GhostLayerLeft(Layer *layer,int gpu_id);
   ~GhostLayerLeft();

    /* ====================== */
    /* GhostLayerLeft Methods */
    /* ====================== */
    cudnnTensorDescriptor_t* getTensorDesc();
    data_t* getOutDevice(){return &this->d_out;};
    data_t* getOutHost(){return &this->h_out;};
    int getOutSize(){return this->noutputs;};

    void displayLayerType();
};

class GhostLayerRight {
  protected:
    int layer_type;     /**< layer identification tag */
    int ninputs;        /**< layer input size */
    int batch_size;     /**< layer batch size */
    int gpu_id;         /**< layer gpu id */

    Real *d_adjoint;    /**< device layer input derivative (size: ninputs*batch_size) */
    Real *h_adjoint;    /**< host   layer input derivative (size: ninputs*batch_size) */

    /* ==================== */
    /* Private Constructors */
    /* ==================== */
    GhostLayerRight();

  public:
    enum layerType {
        FULLYCONNECTED,
        CONVOLUTION,
        MAXPOOL,
        SOFTMAX
    };

    /* ============ */
    /* Constructors */
    /* ============ */
    GhostLayerRight(Layer *layer,int gpu_id);
   ~GhostLayerRight();

    /* ======================= */
    /* GhostLayerRight Methods */
    /* ======================= */
    Real* getAdjointDevice(){return d_adjoint;};
    Real* getAdjointHost(){return h_adjoint;};
    int getInSize(){return this->ninputs;};

    void displayLayerType();
};

#ifdef __cplusplus
}
#endif
#endif /* GHOSTLAYER_H */