/**
 * File:   FullyConnectedLayer.h
 * Author: akirby
 *
 * Created on April 3, 2020, 12:32 PM
 */

#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H

/* system header files */
#include "assert.h"
#include <random>
#include <vector>
#include <string>
#include <sstream>

/* header files */
#include "precision_types.h"
#include "gpu_kernel_defines.h"

#include "Layer.h"
#include "CudaHelper.h"
#include "Activation.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Fully Connected layer with bias.
 */
class FullyConnectedLayer : public Layer {
  public:
    int activation;

    /* ====================== */
    /* host data storage: h_? */
    /* ====================== */
    std::vector<Real> h_bias;        /**< layer bias */
    std::vector<Real> h_weights;     /**< layer weights*/
    std::vector<Real> h_bias_bar;    /**< layer derivative bias */
    std::vector<Real> h_weights_bar; /**< layer derivative weights  */

    /* ======================== */
    /* device data storage: d_? */
    /* ======================== */
    /* network parameters */
    Real *d_bias;        /**< layer bias*/
    Real *d_weights;     /**< layer weights*/
    Real *d_bias_bar;    /**< layer derivative bias */
    Real *d_weights_bar; /**< layer derivative weights */

    /* layer states */
    Real *d_z;          /**< layer state */
    Real *d_a;          /**< layer activation */

    /* layer state derivatives */
    Real *d_state_bar;  /**< layer state derivative */

    /* residual layer flag and step size */
    char residual_layer_flag;
    Real dt;

    /* ===================== */
    /* CUDNN data structures */
    /* ===================== */
    cudnnActivationDescriptor_t actDesc;

    /* ================= */
    /* Class Constructor */
    /* ================= */
    /* Traditional Constructor */
    FullyConnectedLayer(int batch_size_,
                        int ninputs_,
                        int noutputs_,
                        int activation_);

    /* ResNet Constructor */
    FullyConnectedLayer(int batch_size_,
                        int ninputs_,
                        int activation_,
                        Real dt_);
    /* Copy Constructor */
    FullyConnectedLayer(const FullyConnectedLayer &layer);
   ~FullyConnectedLayer();

    Layer *clone();
    void layerAllocateHost();
    void layerAllocateDevice(const int gpu_id,char multigrid_flag);

    /* ============= */
    /* Class Methods */
    /* ============= */
    void scale_dt(Real scaler){dt *= scaler;};

    data_t* formA(data_t *data,const int nsamples);
    data_t* fwd(const data_t *data, const int nsamples, const char add_source);
    Real* bwd(Real *adjoint_prev);
    void updateWeights(Real learning_rate);

    size_t setFwdTensors(cudnnTensorDescriptor_t *srcTensorDesc, const int nsamples);

    int parametersGetCount(){return ninputs*noutputs + noutputs;};
    void parametersInitializeHost(std::mt19937 &gen);
    void parametersCopyLayerDevice(Layer *layer2copy);
    void parametersDeviceToHost();
    void parametersHostToDevice();
    void parametersHostToDeviceAsync();

    bool fromFile(const char *fileprefix);
    void toFile(const char *fileprefix);
    void dumpParameters();
};

#ifdef __cplusplus
}
#endif
#endif /* FULLYCONNECTEDLAYER_H */