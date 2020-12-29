/**
 * File:   Network.h
 * Author: akirby
 *
 * Created on April 23, 2020, 2:03 PM
 */

#ifndef NETWORK_H
#define NETWORK_H

/* header files */
#include "DLMG.h"
#include "CudaHelper.h"
#include "LayerBlock.h"
#include "math_utilities.h"

#include "SoftmaxLayer.h"
#include "MaxPoolLayer.h"
#include "ConvolutionLayer.h"
#include "FullyConnectedLayer.h"
#include "GhostLayer.h"

#include "gpu_kernel_defines.h"

/* ========================================================================== */
#define BREAK_LINE_1 \
        printf("+"); for(int i = 0; i < 69; ++i){printf("=");} printf("+\n");
#define BREAK_LINE_2 \
        printf("+"); for(int i = 0; i < 69; ++i){printf("-");} printf("+\n");

// Command-line flags
#define BW 128
/* ========================================================================== */
// Constant versions of gflags
#define DEFINE_int32(flag,default_value,description) const int FLAGS_##flag = (default_value)
#define DEFINE_uint64(flag,default_value,description) const unsigned long long FLAGS_##flag = (default_value)
#define DEFINE_bool(flag,default_value,description) const bool FLAGS_##flag = (default_value)
#define DEFINE_double(flag,default_value,description) const double FLAGS_##flag = (default_value)
#define DEFINE_string(flag,default_value,description) const std::string FLAGS_##flag ((default_value))
/* ========================================================================== */
// Application parameters
DEFINE_int32(gpu,0,"The GPU ID to use");
DEFINE_int32(iterations,1000,"Number of iterations for training");
DEFINE_int32(random_seed,10,"Override random seed (default uses std::random_device)");
DEFINE_int32(classify,-1,"Number of images to classify to compute error rate (default uses entire test set)");

// Batch parameters
DEFINE_uint64(batch_size,1,"Batch size for training");

// Filenames
DEFINE_bool(pretrained,false,"Use the pretrained CUDNN model as input");
DEFINE_bool(save_data,false,"Save pretrained weights to file");
DEFINE_string(train_images,"train-images-idx3-ubyte","Training images filename");
DEFINE_string(train_labels,"train-labels-idx1-ubyte","Training labels filename");
DEFINE_string(test_images,"t10k-images-idx3-ubyte","Test images filename");
DEFINE_string(test_labels,"t10k-labels-idx1-ubyte","Test labels filename");

// Solver parameters
DEFINE_double(learning_rate,0.0001,"Base learning rate");
DEFINE_double(lr_gamma,0.0001,"Learning rate policy gamma");
DEFINE_double(lr_power,0.75,"Learning rate policy power");
/* ========================================================================== */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double comm_time;
    double comp_time;
    double total_time;
}
timerdata_t;

struct Network {
  public:
    DLMG dlmg;
    int global_nranks;
    int model_nranks;
    int data_nranks;

    int global_rank;
    int model_rank;
    int data_rank;
    int gpu_id;

    int s_layer[MAX_LEVELS];
    int e_layer[MAX_LEVELS];
    int nblocks;
    int batch_size;

    std::vector<Real *> h_out;
    std::vector<Real *> h_adjoint;
    Real *d_data;
    Real *d_labels;
    Real *d_onevec;
    cudnnTensorDescriptor_t dataTensor;

    std::vector<GhostLayerLeft *> leftGhost;
    std::vector<GhostLayerRight *> rightGhost;

    int globalLayers_count;
    std::vector<std::vector<Layer *>> globalLayers;
    std::vector<std::vector<LayerBlock *>> blocks;

    cudaStream_t **asyncCudaStreams;
    cudaStream_t syncCudaStream;

    /* ============ */
    /* Constructors */
    /* ============ */
    /* Disable copying */
    Network& operator=(const Network&) = delete;
    Network(const Network&) = delete;

    Network(DLMG &dlmg_);
    Network(DLMG &dlmg_, std::vector<Layer *> &layers_);
   ~Network();

    /* ============= */
    /* Class Methods */
    /* ============= */
    void add(Layer *layer);
    void initialize(int max_batch_size);
    void parametersInitializeHost(std::mt19937 &gen);
    void parametersHostToDeviceAsync(int level);

    void setFwdTensors(int nsamples,int level);
    void setBwdTensors(int level);
    void setTensors(int nsamples,int level);

    void setNetworkBuffers(int batch_size,int level);

    Real* getOutHost(int level);
    data_t* getOutDevice(int level);
    Real* getAdjointHost(int level);
    Real* getAdjointDevice(int level);

    int getInSize(int level);
    int getOutSize(int level);
    cudnnTensorDescriptor_t* getOutTensorDesc(int level);

    void synchronizeNetwork(int level);
    void asynchronizeNetwork(int level);

    void fwd_sync(timerdata_t *time_data,data_t *data,const int nsamples,int level,char add_source);
    void bwd_sync(Real *labels,int level);

    void Multigrid_update_parameters(int level);
    Real Multigrid_residual(timerdata_t *time_data,int nsamples,int level);
    void Multigrid_correction(int nsamples,int level);
    void Multigrid_restriction(timerdata_t *time_data,int nsamples,int level);
    void Multigrid_F_relaxation(int nsamples,int level,char add_source);
    void Multigrid_CF_relaxation(data_t *data,int nsamples,int level,char add_source);
    void Multigrid_parCF_relaxation(timerdata_t *time_data,data_t *data,int nsamples,int level,char add_source);
    void Multigrid_communicate_ghost(data_t **data,int nsamples,int level);
    void Multigrid_reset_states(int nsamples,int level);
    void Multigrid_display(int nsamples,int level);
    void Multigrid_fwd(data_t *data,const int nsamples,int level,int trained_iterations,char add_source);

    void update(Real learning_rate);
    void fit(dataset_t *dataset, int batch_size, int nepochs);
    void evaluate(dataset_t *dataset, int nclassifications, const char str[]);

    void barrier(const char *str, int sleep_sec);
    void display_mnist(Real *image);
    void display_layer(Real *image,int nsamples);
    void displayLayerOutput(Real *d_output,int layer_id,int nsamples);
    void displayFwdOutput(int nsamples,int level);
    void displayBwdOutput(int nsamples,int level);
};

#ifdef __cplusplus
}
#endif
#endif /* NETWORK_H */
