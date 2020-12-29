/**
 * File:   MaxPoolLayer.cpp
 * Author: akirby
 *
 * Created on April 3, 2020, 12:41 PM
 */

/* header files */
#include "MaxPoolLayer.h"

MaxPoolLayer::MaxPoolLayer(int batch_size_,
                           int in_channels_,
                           int in_width_,
                           int in_height_,
                           int size_,
                           int stride_) :
    Layer(layerType::MAXPOOL),
    in_channels(in_channels_),
    in_width(in_width_),
    in_height(in_height_),
    size(size_),
    stride(stride_)
{
    batch_size = batch_size_;
    out_channels = in_channels;
    out_width = in_width/stride;
    out_height = in_height/stride;

    ninputs = in_channels * in_height * in_width;
    noutputs = out_channels * out_width * out_height;

    /* CUDNN Data Structures */
    checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&tensorDesc));

    setFwdTensors(nullptr,batch_size);
    checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc,
                              /* mode = */ CUDNN_POOLING_MAX,
                  /* maxpoolingNanOpt = */ CUDNN_PROPAGATE_NAN,
                      /* windowHeight = */ size,
                       /* windowWidth = */ size,
                   /* verticalPadding = */ 0,
                 /* horizontalPadding = */ 0,
                    /* verticalStride = */ stride,
                  /* horizontalStride = */ stride));
}

MaxPoolLayer::MaxPoolLayer(const MaxPoolLayer &layer) :
    Layer(layerType::MAXPOOL),
    in_channels(layer.in_channels),
    in_width(layer.in_width),
    in_height(layer.in_height),
    size(layer.size),
    stride(layer.stride),
    out_channels(layer.out_channels),
    out_width(layer.out_width),
    out_height(layer.out_height)
{
    fine_level_flag = 0;
    ninputs = layer.ninputs;
    noutputs = layer.noutputs;
    batch_size = layer.batch_size;
    global_idx = layer.global_idx;

    /* CUDNN Data Structures */
    checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&tensorDesc));

    setFwdTensors(nullptr,batch_size);
    checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc,
                              /* mode = */ CUDNN_POOLING_MAX,
                  /* maxpoolingNanOpt = */ CUDNN_PROPAGATE_NAN,
                      /* windowHeight = */ size,
                       /* windowWidth = */ size,
                   /* verticalPadding = */ 0,
                 /* horizontalPadding = */ 0,
                    /* verticalStride = */ stride,
                  /* horizontalStride = */ stride));
}

MaxPoolLayer::~MaxPoolLayer(){
    checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(tensorDesc));

    checkCudaErrors(cudaFree(d_z));
    checkCudaErrors(cudaFree(adjoint));

    if(A) checkCudaErrors(cudaFree(A));
    if(G) checkCudaErrors(cudaFree(G));
    if(R) checkCudaErrors(cudaFree(R));
}

Layer* MaxPoolLayer::clone(){
    return new MaxPoolLayer(*this);
}

void MaxPoolLayer::layerAllocateDevice(const int gpu_id,char multigrid_flag){
    /* ============================ */
    /* Allocate GPU Data Structures */
    /* ============================ */
    /* set GPU id to allocate layer */
    checkCudaErrors(cudaSetDevice(gpu_id));

    /* Device Layer States    | Buffer | Data Type    | N          | C            | H          | W        | */
    checkCudaErrors(cudaMalloc(&d_z,     sizeof(Real) * batch_size * out_channels * out_height * out_width));
    checkCudaErrors(cudaMalloc(&adjoint, sizeof(Real) * batch_size *  in_channels *  in_height *  in_width));

    if (multigrid_flag) {
        this->multigrid_flag = multigrid_flag;
        checkCudaErrors(cudaMalloc(&A, sizeof(Real) * batch_size * out_channels * out_height * out_width));
        checkCudaErrors(cudaMalloc(&G, sizeof(Real) * batch_size * out_channels * out_height * out_width));
        checkCudaErrors(cudaMalloc(&R, sizeof(Real) * batch_size * out_channels * out_height * out_width));

        g_blas_memset_async(A, 0, sizeof(Real) * batch_size * noutputs, streamID[0]);
        g_blas_memset_async(G, 0, sizeof(Real) * batch_size * noutputs, streamID[0]);
        g_blas_memset_async(R, 0, sizeof(Real) * batch_size * noutputs, streamID[0]);
    }
    g_blas_memset_async(d_z, 0, sizeof(Real) * batch_size * noutputs, streamID[0]);

    /* set output data info */
    out.tensorDesc = &tensorDesc;
    out.ptr = d_z;
}

data_t* MaxPoolLayer::fwd(const data_t *data, const int nsamples, const char add_source){
    (void) nsamples;

    Real alpha = 1.0;
    Real beta = 0.0;

    /* save input data for adjoint */
    in.ptr = data->ptr;
    in.tensorDesc = data->tensorDesc;

    //checkCudaErrors(cudaStreamSynchronize(streamID[0]));
    checkCUDNN(cudnnPoolingForward(cudaHandles.cudnnHandle,
                                   poolDesc,
                                   &alpha,
                                   data->tensorDesc[0],data->ptr,
                                   &beta,
                                   tensorDesc,d_z));

    if (add_source) {
        g_blas_axpy(cudaHandles.cublasHandle,
                    noutputs*batch_size,
                    &alpha,
                    G,1,
                    d_z,1);
    }

    /* set output data info */
    out.tensorDesc = &tensorDesc;
    out.ptr = d_z;
    return &out;
}

data_t* MaxPoolLayer::formA(data_t *data,const int nsamples){
    Real malpha = -1.0;
    Real alpha = 1.0;
    Real beta = 0.0;

    //checkCudaErrors(cudaStreamSynchronize(streamID[0]));

    /* form A:= d_z(old) - d_z(new) */
    g_blas_copy(cudaHandles.cublasHandle,
                noutputs*batch_size,
                d_z,1,
                A,1);

    checkCUDNN(cudnnPoolingForward(cudaHandles.cudnnHandle,
                                   poolDesc,
                                   &alpha,
                                   data->tensorDesc[0],data->ptr,
                                   &beta,
                                   tensorDesc,R));

    g_blas_axpy(cudaHandles.cublasHandle,
                noutputs*batch_size,
                &malpha,
                R,1,
                A,1);

    return nullptr;
}

Real* MaxPoolLayer::bwd(Real *adjoint_prev){
    Real alpha = 1.0;
    Real beta = 0.0;
    checkCUDNN(cudnnPoolingBackward(cudaHandles.cudnnHandle,poolDesc,
                                    &alpha,
                                    tensorDesc,d_z,
                                    tensorDesc,adjoint_prev,
                                    in.tensorDesc[0],in.ptr,
                                    &beta,
                                    in.tensorDesc[0],adjoint));
    return adjoint;
}

size_t MaxPoolLayer::setFwdTensors(cudnnTensorDescriptor_t *srcTensorDesc, const int nsamples){
    (void) srcTensorDesc;
    checkCUDNN(cudnnSetTensor4dDescriptor(tensorDesc,
                           /* format = */ CUDNN_TENSOR_NCHW,
                         /* dataType = */ DNN_REAL,
                                /* n = */ nsamples,
                                /* c = */ out_channels,
                                /* h = */ out_height,
                                /* w = */ out_width));
    return 0;
}