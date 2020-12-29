/**
 * File:   Layer.h
 * Author: akirby
 *
 * Created on April 3, 2020, 1:59 PM
 */

#ifndef LAYER_H
#define LAYER_H

/* system header files*/
#include <random>

/* header files */
#include "dlmg_types.h"
#include "precision_types.h"
#include "CudaHelper.h"
#include "gpu_kernel_defines.h"

#ifdef __cplusplus
extern "C" {
#endif

class Layer {
  public:
    int level;                  /**< layer level index */
    int global_idx;             /**< layer global index */
    int batch_size;             /**< layer batch size */
    int layer_type;             /**< layer identification tag */
    int ninputs;                /**< layer input size */
    int noutputs;               /**< layer output size */

    cudaStream_t *streamID;     /**< layer gpu stream ID */
    cuda_info_t cudaHandles;    /**< layer gpu handles */
    Real *adjoint;              /**< layer input derivative */
    data_t in;                  /**< layer input data  */
    data_t out;                 /**< layer output data */
    data_t h_out;               /**< layer host output data */

    char multigrid_flag;        /**< multigrid solve flag */
    char fine_level_flag;       /**< multigrid finest level flag */
    Real *A;                    /**< multigrid A-vector */
    Real *G;                    /**< multigrid G-vector */
    Real *R;                    /**< multigrid R-vector */
    Layer *flayer;              /**< multigrid fine level layer */
    Layer *clayer;              /**< multigrid coarse level layer */

    cudnnTensorDescriptor_t tensorDesc; /**< layer DNN library tensor descriptor */
    Real *onevec;               /**< ones vector used for bias computation */
    void *workspaceFwd;         /**< DNN library workspace */
    void *workspaceBwd;         /**< DNN library workspace */
    size_t workspaceSizeFwd;    /**< DNN library workspace size */
    size_t workspaceSizeBwd;    /**< DNN library workspace size */

    /* ============ */
    /* Constructors */
    /* ============ */
    Layer();
    Layer(int layer_type);
    virtual ~Layer() = 0;

  public:
    /* ============= */
    /* Layer Methods */
    /* ============= */
    Real* getAdjoint(){return adjoint;};
    data_t* getOutHost(){return &this->h_out;};
    data_t* getOutDevice(){return &this->out;};
    data_t* getInDevice(){return &this->in;};

    void setGlobalIdx(int idx){this->global_idx = idx;};
    int getGlobalIdx(){return this->global_idx;};

    void setLevel(int level){this->level = level;};
    int getLevel(){return this->level;};

    int getInSize(){return this->ninputs;};
    int getOutSize(){return this->noutputs;};
    int getBatchSize(){return this->batch_size;};
    int getLayerType(){return layer_type;};
    cudnnTensorDescriptor_t* getTensorDesc(){return &tensorDesc;};

    void setCoarseLayer(Layer *clayer){this->clayer = clayer;};
    void setFineLayer(Layer *flayer){this->flayer = flayer;};
    Layer * getCoarseLayer(){return this->clayer;};
    Layer * getFineLayer(){return this->flayer;};

    Real* getA(){return this->A;};
    Real* getG(){return this->G;};
    Real* getR(){return this->R;};
    void restriction(int nsamples);
    Real residual_norm_sqrd(int nsamples);

    void setOneVector(Real *d_onevec);
    void setCudaHandles(cuda_info_t *cudaHandles_);
    cuda_info_t *getCudaHandles(){return &this->cudaHandles;};

    void setCudaStream(cudaStream_t *cudaStream);
    cudaStream_t getCudaStream(){return *streamID;};
    cudaStream_t *getCudaStreamAddress(){return streamID;};

    void displayLayer(Real *d_output,int nsamples,int ndisp);
    void displayLayerType();

    /* ================================ */
    /* Overloaded Derived Class Methods */
    /* ================================ */
    virtual void scale_dt(Real scaler){};

    virtual size_t setFwdTensors(cudnnTensorDescriptor_t *inputTensor, const int nsamples){return 0;};
    virtual size_t setBwdTensors(cudnnTensorDescriptor_t *inputTensor){return 0;};

    virtual void parametersInitializeHost(std::mt19937 &gen){};
    virtual void parametersDeviceToHost(){};
    virtual void parametersHostToDevice(){};
    virtual void parametersHostToDeviceAsync(){};
    virtual int parametersGetCount(){return 0;};

    virtual void updateWeights(Real learning_rate){};
    virtual void parametersCopyLayerDevice(Layer *layer2copy){};

    virtual bool fromFile(const char *fileprefix){return true;};
    virtual void toFile(const char *fileprefix){};

    /* ============================== */
    /* Required Derived Class Methods */
    /* ============================== */
    virtual Layer *clone() = 0;
    virtual void layerAllocateHost() = 0;
    virtual void layerAllocateDevice(const int gpu_id,char multigrid_flag) = 0;
    virtual data_t* formA(data_t *data, const int nsamples) = 0;
    virtual data_t* fwd(const data_t *data, const int nsamples, const char add_source) = 0;
    virtual Real* bwd(Real *adjoint_prev) = 0;
};

#ifdef __cplusplus
}
#endif
#endif /* LAYER_H */