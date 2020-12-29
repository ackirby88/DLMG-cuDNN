/**
 * File:   SoftmaxLayer.cu
 * Author: akirby
 *
 * Created on April 15, 2020, 2:55 PM
 */

/* header files */
#include "SoftmaxLayer.h"

/* ============ */
/* CUDA Kernels */
/* ============ */
/**
 * ================================================================================ *
 * Gradient of the Softmax + Cross Entropy Loss for each result in batch.           *
 * Uses the softmax values from forward propagation to compute the difference.      *
 * ================================================================================ *
 * Softmax + Cross Entropy Loss Functions:                                          *
 *   q(x):= softmax(z(x)) = prediction probabilities                                *
 *   p(x):= one-hot labels = (0,1,0,0)                                              *
 *                                                                                  *
 * Cross Entropy Loss:                                                              *
 *   Loss = -SUM_x(p(x) * log(q(x)))                                                *
 *                                                                                  *
 * Cross Entropy Loss Derivative using Softmax:                                     *
 *    Grad_Loss = q(x) - p(x) = q_i - 1(y_i = 1)                                    *
 * ================================================================================ *
 * http://machinelearningmechanic.com/assets/pdfs/cross_entropy_loss_derivative.pdf *
 * ================================================================================ *
 * @param [in] label        Training batch label values.
 * @param [in] nlabels      Number of possible labels.
 * @param [in] batch_size   Size of the trained batch.
 * @param [inout] diff      Resulting gradient: input contains softmax.
 */
__global__ void SoftmaxLossGradient(const Real *label,
                                    int nlabels,
                                    int batch_size,
                                    Real *diff){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= batch_size) return;

    const int label_value = static_cast<int>(label[idx]);

    /* Softmax + Cross Entropy Loss Derivative */
    diff[idx * nlabels + label_value] -= 1.0;
}

/* Softmax + Cross Entropy Loss Function
 * @param [in] label        [batch_size]: Training batch label values.
 * @param [in] nlabels      Number of possible labels.
 * @param [in] batch_size   Size of the trained batch.
 * @param [in] softmax      Softmax values
 * @param [out] loss        [batch_size]: loss values.
 */
__global__ void SoftmaxLoss(const Real *label,
                            int nlabels,
                            int batch_size,
                            Real *softmax,
                            Real *loss){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= batch_size) return;

    const int label_value = static_cast<int>(label[idx]);

    /* Softmax + Cross Entropy Loss */
    loss[idx] = -log(softmax[idx * nlabels + label_value]);
}

/* ============= */
/* Softmax Layer */
/* ============= */
SoftmaxLayer::SoftmaxLayer(int batch_size_, int in_channels_) :
    Layer(layerType::SOFTMAX),
    in_channels(in_channels_)
{
    batch_size = batch_size_;
    out_channels = in_channels;
    ninputs = in_channels;
    noutputs = out_channels;
}

SoftmaxLayer::SoftmaxLayer(const SoftmaxLayer &layer) :
    Layer(layerType::SOFTMAX),
    in_channels(layer.in_channels),
    out_channels(layer.out_channels)
{
    fine_level_flag = 0;
    ninputs = layer.ninputs;
    noutputs = layer.noutputs;
    batch_size = layer.batch_size;
    global_idx = layer.global_idx;
}

SoftmaxLayer::~SoftmaxLayer(){
    checkCudaErrors(cudaFree(d_z));
    checkCudaErrors(cudaFree(adjoint));

    if(A) checkCudaErrors(cudaFree(A));
    if(G) checkCudaErrors(cudaFree(G));
    if(R) checkCudaErrors(cudaFree(R));
}

Layer* SoftmaxLayer::clone(){
    return new SoftmaxLayer(*this);
}

void SoftmaxLayer::layerAllocateDevice(const int gpu_id,char multigrid_flag){
    /* ============================ */
    /* Allocate GPU Data Structures */
    /* ============================ */
    /* set GPU id to allocate layer */
    checkCudaErrors(cudaSetDevice(gpu_id));

    /* Device Layer States    | Buffer | Data Type    | N          | C           | */
    checkCudaErrors(cudaMalloc(&d_z,     sizeof(Real) * batch_size * out_channels));
    checkCudaErrors(cudaMalloc(&adjoint, sizeof(Real) * batch_size *  in_channels));

    if (multigrid_flag) {
        this->multigrid_flag = multigrid_flag;
        checkCudaErrors(cudaMalloc(&A, sizeof(Real) * batch_size * out_channels));
        checkCudaErrors(cudaMalloc(&G, sizeof(Real) * batch_size * out_channels));
        checkCudaErrors(cudaMalloc(&R, sizeof(Real) * batch_size * out_channels));

        g_blas_memset_async(A, 0, sizeof(Real) * batch_size * noutputs, streamID[0]);
        g_blas_memset_async(G, 0, sizeof(Real) * batch_size * noutputs, streamID[0]);
        g_blas_memset_async(R, 0, sizeof(Real) * batch_size * noutputs, streamID[0]);
    }
    g_blas_memset_async(d_z, 0, sizeof(Real) * batch_size * noutputs, streamID[0]);

    /* set output data info */
    out.tensorDesc = nullptr;
    out.ptr = d_z;
}

data_t* SoftmaxLayer::fwd(const data_t *data, const int nsamples, const char add_source){
    (void) nsamples;

    Real alpha = 1.0;
    Real beta  = 0.0;

    /* save input data for adjoint */
    in.ptr = data->ptr;
    in.tensorDesc = data->tensorDesc;

    cudnnTensorDescriptor_t tensorDescLoc = data->tensorDesc[0];

    //checkCudaErrors(cudaStreamSynchronize(streamID[0]));
    checkCUDNN(cudnnSoftmaxForward(cudaHandles.cudnnHandle,
                                   CUDNN_SOFTMAX_ACCURATE,
                                   CUDNN_SOFTMAX_MODE_CHANNEL,
                                   &alpha,
                                   data->tensorDesc[0],data->ptr,
                                   &beta,
         /* use incoming tensor */ tensorDescLoc,d_z));

    if (add_source) {
        /* add G source term */
        g_blas_axpy(cudaHandles.cublasHandle,
                    noutputs*batch_size,
                    &alpha,
                    G,1,
                    d_z,1);
    }

    /* set output data info */
    out.tensorDesc = nullptr;
    out.ptr = d_z;
    return &out;
}

data_t* SoftmaxLayer::formA(data_t *data,const int nsamples){
    Real malpha = -1.0;
    Real alpha = 1.0;
    Real beta  = 0.0;

    //checkCudaErrors(cudaStreamSynchronize(streamID[0]));

    /* form A:= d_z(old) - d_z(new) */
    g_blas_copy(cudaHandles.cublasHandle,
                noutputs*batch_size,
                d_z,1,
                A,1);

    checkCUDNN(cudnnSoftmaxForward(cudaHandles.cudnnHandle,
                                   CUDNN_SOFTMAX_ACCURATE,
                                   CUDNN_SOFTMAX_MODE_CHANNEL,
                                   &alpha,
                                   data->tensorDesc[0],data->ptr,
                                   &beta,
         /* use incoming tensor */ data->tensorDesc[0],R));

    g_blas_axpy(cudaHandles.cublasHandle,
                noutputs*batch_size,
                &malpha,
                R,1,
                A,1);

    return nullptr;
}

Real* SoftmaxLayer::bwd(Real *labels){
    Real oneObatch_size = 1.0 / static_cast<Real>(batch_size);

    /* ================================================= */
    /* Softmax + Cross Entropy Loss Functions:           */
    /*   q(x):= softmax(z(x)) = prediction probabilities */
    /*   p(x):= one-hot labels = (0,1,0,0)               */
    /*                                                   */
    /* Cross Entropy Loss:                               */
    /*   Loss = -SUM_x(p(x) * log(q(x)))                 */
    /*                                                   */
    /* Cross Entropy Loss Derivative using Softmax:      */
    /*    Grad_Loss = q(x) - p(x) = q_i - 1(y_i = 1)     */
    /* ================================================= */
    /* Labels is NOT a one-hot coded vector --           */
    /*   Labels[idx] = (class-id), of sample idx.        */
    /* Thus, we don't use cudnnSoftmaxBackard function.  */
    /* ========================================================================= */
    /* machinelearningmechanic.com/assets/pdfs/cross_entropy_loss_derivative.pdf */
    /* ========================================================================= */
#define THREADS_PER_BLOCK 128

    /* fill adjoint with softmax values */
    checkCudaErrors(cudaMemcpyAsync(adjoint,d_z,
                                    sizeof(Real)*batch_size*noutputs,
                                    cudaMemcpyDeviceToDevice));

    /* softmax + cross-entropy loss gradient */
    const int NBLOCKS = RoundUp(batch_size,THREADS_PER_BLOCK);
    SoftmaxLossGradient<<<NBLOCKS,THREADS_PER_BLOCK>>>(labels,
                                                       noutputs,
                                                       batch_size,
                                                       adjoint);

    /* scale by batch size for SGD */
    checkCudaErrors(g_blas_scal(cudaHandles.cublasHandle,
                                noutputs*batch_size,
                                &oneObatch_size,
                                adjoint,
                                1));
    return adjoint;
}