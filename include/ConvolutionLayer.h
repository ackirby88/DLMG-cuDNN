/**
 * File:   ConvolutionLayer.h
 * Author: akirby
 *
 * Created on April 3, 2020, 12:38 PM
 */

#ifndef CONVOLUTIONLAYER_H
#define CONVOLUTIONLAYER_H

/* system header files */
#include "assert.h"
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
 * Convolutional layer with bias.
 */
class ConvolutionLayer : public Layer {
  public:
    int in_channels;
    int in_width;
    int in_height;

    int kernel_size;
    int out_channels;
    int pad_size;
    int out_width;
    int out_height;

    int activation;

    /* ====================== */
    /* host data storage: h_? */
    /* ====================== */
    std::vector<Real> h_bias;       /**< layer bias */
    std::vector<Real> h_kernel;     /**< layer kernel */
    std::vector<Real> h_bias_bar;   /**< layer derivative bias */
    std::vector<Real> h_kernel_bar; /**< layer derivative kernel */

    /* ======================== */
    /* device data storage: d_? */
    /* ======================== */
    /* network parameters */
    Real *d_bias;        /**< layer bias */
    Real *d_kernel;      /**< layer kernel */
    Real *d_bias_bar;    /**< layer derivative bias */
    Real *d_kernel_bar;  /**< layer derivative kernel */

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
    cudnnTensorDescriptor_t biasTensorDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnActivationDescriptor_t actDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t convFwdAlgo;
    cudnnConvolutionBwdFilterAlgo_t convBwdFilterAlgo;
    cudnnConvolutionBwdDataAlgo_t convBwdDataAlgo;

    /* ================= */
    /* Class Constructor */
    /* ================= */
    /* Traditional Constructor */
    ConvolutionLayer(int batch_size_,
                     int in_channels_,
                     int in_width_,
                     int in_height_,
                     int kernel_size_,
                     int out_channels_,
                     int pad_size_,
                     int activation_);

    /* ResNet Constructor */
    ConvolutionLayer(int batch_size_,
                     int in_channels_,
                     int in_width_,
                     int in_height_,
                     int kernel_size_,
                     int activation_,
                     Real dt_);

    ConvolutionLayer(const ConvolutionLayer &layer);
   ~ConvolutionLayer();

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

    size_t setFwdTensors(cudnnTensorDescriptor_t *srcTensorDesc,const int nsamples);
    size_t setBwdTensors(cudnnTensorDescriptor_t *srcTensorDesc);

    int parametersGetCount(){return  in_channels*out_channels*kernel_size*kernel_size + out_channels;};
    void parametersInitializeHost(std::mt19937 &gen);
    void parametersCopyLayerDevice(Layer *layer2copy);
    void parametersDeviceToHost();
    void parametersHostToDevice();
    void parametersHostToDeviceAsync();

    bool fromFile(const char *fileprefix);
    void toFile(const char *fileprefix);
};

#ifdef __cplusplus
}
#endif
#endif /* CONVOLUTIONLAYER_H */