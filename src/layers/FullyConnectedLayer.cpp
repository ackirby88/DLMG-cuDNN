/**
 * File:   FullyConnectedLayer.cpp
 * Author: akirby
 *
 * Created on April 3, 2020, 12:32 PM
 */

/* header files */
#include "FullyConnectedLayer.h"
#include "Layer.h"

FullyConnectedLayer::FullyConnectedLayer(int batch_size_,
                                         int ninputs_,
                                         int noutputs_,
                                         int activation_) :
    Layer(layerType::FULLYCONNECTED),
    activation(activation_),
    residual_layer_flag(0),
    dt(1.0)
{
    batch_size = batch_size_;
    ninputs = ninputs_;
    noutputs = noutputs_;

    /* CUDNN Data Structures */
    checkCUDNN(cudnnCreateTensorDescriptor(&tensorDesc));
    checkCUDNN(cudnnCreateActivationDescriptor(&actDesc));

    if (activationSelect(activation)) {
        checkCUDNN(cudnnSetActivationDescriptor(actDesc,
                                   /* mode = */ activationSelect(activation),
                             /* reluNanOpt = */ CUDNN_PROPAGATE_NAN,
                                   /* coef = */ 0.0));
    }
}

FullyConnectedLayer::FullyConnectedLayer(int batch_size_,
                                         int ninputs_,
                                         int activation_,
                                         Real dt_) :
    Layer(layerType::FULLYCONNECTED),
    activation(activation_),
    residual_layer_flag(1),
    dt(dt_)
{
    batch_size = batch_size_;
    ninputs = ninputs_;
    noutputs = ninputs_;

    /* make sure dimensions match for residual network */
    assert(ninputs == noutputs);

    /* CUDNN Data Structures */
    checkCUDNN(cudnnCreateTensorDescriptor(&tensorDesc));
    checkCUDNN(cudnnCreateActivationDescriptor(&actDesc));

    if (activationSelect(activation)) {
        checkCUDNN(cudnnSetActivationDescriptor(actDesc,
                                   /* mode = */ activationSelect(activation),
                             /* reluNanOpt = */ CUDNN_PROPAGATE_NAN,
                                   /* coef = */ 0.0));
    }
}

FullyConnectedLayer::FullyConnectedLayer(const FullyConnectedLayer &layer) :
    Layer(layerType::FULLYCONNECTED),
    activation(layer.activation),
    residual_layer_flag(layer.residual_layer_flag),
    dt(layer.dt)
{
    ninputs = layer.ninputs;
    noutputs = layer.noutputs;
    batch_size = layer.batch_size;
    global_idx = layer.global_idx;

    /* alias layer parameters */
    fine_level_flag = 0;
    d_bias = layer.d_bias;
    d_weights = layer.d_weights;
    d_bias_bar = layer.d_bias_bar;
    d_weights_bar = layer.d_weights_bar;

    /* CUDNN Data Structures */
    checkCUDNN(cudnnCreateTensorDescriptor(&tensorDesc));
    checkCUDNN(cudnnCreateActivationDescriptor(&actDesc));

    if (activationSelect(activation)) {
        checkCUDNN(cudnnSetActivationDescriptor(actDesc,
                                   /* mode = */ activationSelect(activation),
                             /* reluNanOpt = */ CUDNN_PROPAGATE_NAN,
                                   /* coef = */ 0.0));
    }
}

FullyConnectedLayer::~FullyConnectedLayer(){
    checkCUDNN(cudnnDestroyTensorDescriptor(tensorDesc));

    if (fine_level_flag) {
        checkCudaErrors(cudaFree(d_bias));
        checkCudaErrors(cudaFree(d_weights));
        checkCudaErrors(cudaFree(d_bias_bar));
        checkCudaErrors(cudaFree(d_weights_bar));
    }

    checkCudaErrors(cudaFree(d_z));
    checkCudaErrors(cudaFree(d_state_bar));
    checkCudaErrors(cudaFree(adjoint));

    if(A) checkCudaErrors(cudaFree(A));
    if(G) checkCudaErrors(cudaFree(G));
    if(R) checkCudaErrors(cudaFree(R));

    if (activationSelect(activation)) {
        checkCUDNN(cudnnDestroyActivationDescriptor(actDesc));
        checkCudaErrors(cudaFree(d_a));
    }
}

Layer* FullyConnectedLayer::clone(){
    return new FullyConnectedLayer(*this);
}

void FullyConnectedLayer::layerAllocateHost(){
    h_weights.resize(ninputs * noutputs);
    h_bias.resize(noutputs);
}

void FullyConnectedLayer::layerAllocateDevice(const int gpu_id,char multigrid_flag){
    /* ============================ */
    /* Allocate GPU Data Structures */
    /* ============================ */
    /* set GPU id to allocate layer */
    checkCudaErrors(cudaSetDevice(gpu_id));

    /* Device Network Parameters */
    if (fine_level_flag) {
        checkCudaErrors(cudaMalloc(&d_bias, sizeof(Real) * h_bias.size()));
        checkCudaErrors(cudaMalloc(&d_weights, sizeof(Real) * h_weights.size()));
        checkCudaErrors(cudaMalloc(&d_bias_bar, sizeof(Real) * h_bias.size()));
        checkCudaErrors(cudaMalloc(&d_weights_bar, sizeof(Real) * h_weights.size()));
    }

    /* Device Layer States    | Buffer     | Data Type    | N          | C        | H | W | */
    checkCudaErrors(cudaMalloc(&d_z,         sizeof(Real) * batch_size * noutputs * 1 * 1));
    checkCudaErrors(cudaMalloc(&d_state_bar, sizeof(Real) * batch_size * noutputs * 1 * 1));
    checkCudaErrors(cudaMalloc(&adjoint,     sizeof(Real) * batch_size * ninputs  * 1 * 1));

    if (multigrid_flag) {
        this->multigrid_flag = multigrid_flag;
        checkCudaErrors(cudaMalloc(&A, sizeof(Real) * batch_size * noutputs * 1 * 1));
        checkCudaErrors(cudaMalloc(&G, sizeof(Real) * batch_size * noutputs * 1 * 1));
        checkCudaErrors(cudaMalloc(&R, sizeof(Real) * batch_size * noutputs * 1 * 1));

        g_blas_memset_async(A, 0, sizeof(Real) * batch_size * noutputs, streamID[0]);
        g_blas_memset_async(G, 0, sizeof(Real) * batch_size * noutputs, streamID[0]);
        g_blas_memset_async(R, 0, sizeof(Real) * batch_size * noutputs, streamID[0]);
    }

    if (activationSelect(activation)) {
        checkCudaErrors(cudaMalloc(&d_a, sizeof(Real) * batch_size * noutputs * 1 * 1));
    } else {
        /* alias activation with state */
        d_a = d_z;
    }
    g_blas_memset_async(d_a, 0, sizeof(Real) * batch_size * noutputs, streamID[0]);

    /* set output data info */
    out.tensorDesc = &tensorDesc;
    out.ptr = d_a;
}

data_t* FullyConnectedLayer::fwd(const data_t *data, const int nsamples, const char add_source){
    Real alpha = 1.0;
    Real beta = 0.0;

    /* forward propagate neurons using weights (d_z = [weights]'*[data]) */
    /* ========================================== */
    /* [gemm] C := alpha*op( A )*op( B ) + beta*C */
    /* ========================================== */

    /* save input data for adjoint */
    in.ptr = data->ptr;
    in.tensorDesc = data->tensorDesc;

    //checkCudaErrors(cudaStreamSynchronize(streamID[0]));
    checkCudaErrors(g_blas_gemm(cudaHandles.cublasHandle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                noutputs,nsamples,ninputs,
                                &alpha,
                                d_weights,ninputs,
                                data->ptr,ninputs,
                                &beta,
                                d_z,noutputs));

    /* add bias using GEMM's "beta" (d_z += [bias]*[1_vec]') */
    checkCudaErrors(g_blas_gemm(cudaHandles.cublasHandle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                noutputs,nsamples,1,
                                &alpha,
                                d_bias,noutputs,
                                onevec,1,
                                &alpha,
                                d_z,noutputs));

    /* apply activation function (d_a = dt*activation(d_z)) */
    if (activationSelect(activation)) {
        checkCUDNN(cudnnActivationForward(cudaHandles.cudnnHandle,actDesc,
                                          &dt,tensorDesc,d_z,
                                          &beta,tensorDesc,d_a));
    } else {
        /* d_a is already aliased to d_z */
        if (residual_layer_flag) {
            /* rescale by dt */
            g_blas_scal(cudaHandles.cublasHandle,
                        noutputs*batch_size,
                        &dt,
                        d_a,1);
        }
    }

    /*  residual network: add input data to output */
    if (residual_layer_flag) {
        /*  [axpy] y := alpha*x + y */
        g_blas_axpy(cudaHandles.cublasHandle,
                    noutputs*batch_size,
                    &alpha,
                    data->ptr,1,
                    d_a,1);
    }

    if (add_source) {
        g_blas_axpy(cudaHandles.cublasHandle,
                    noutputs*batch_size,
                    &alpha,
                    G,1,
                    d_a,1);
    }

    /* set output data info */
    out.tensorDesc = &tensorDesc;
    out.ptr = d_a;
    return &out;
}

data_t* FullyConnectedLayer::formA(data_t *data,const int nsamples){
    Real malpha = -1.0;
    Real alpha = 1.0;
    Real beta = 0.0;

    //checkCudaErrors(cudaStreamSynchronize(streamID[0]));

    /* form A:= d_a(old) - d_a(new) */
    g_blas_copy(cudaHandles.cublasHandle,
                noutputs*batch_size,
                d_a,1,
                A,1);

    /* forward propagate neurons using weights (d_z = [weights]'*[data]) */
    checkCudaErrors(g_blas_gemm(cudaHandles.cublasHandle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                noutputs,nsamples,ninputs,
                                &alpha,
                                d_weights,ninputs,
                                data->ptr,ninputs,
                                &beta,
                                R,noutputs));

    /* add bias using GEMM's "beta" (d_z += [bias]*[1_vec]') */
    checkCudaErrors(g_blas_gemm(cudaHandles.cublasHandle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                noutputs,nsamples,1,
                                &alpha,
                                d_bias,noutputs,
                                onevec,1,
                                &alpha,
                                R,noutputs));

    /* apply activation function (d_a = dt*activation(d_z)) */
    if (activationSelect(activation)) {
        checkCUDNN(cudnnActivationForward(cudaHandles.cudnnHandle,actDesc,
                                          &dt,tensorDesc,R,
                                          &beta,tensorDesc,R));
    } else {
        /* d_a is already aliased to d_z */
        if (residual_layer_flag) {
            /* rescale by dt */
            g_blas_scal(cudaHandles.cublasHandle,
                        noutputs*batch_size,
                        &dt,
                        R,1);
        }
    }

    /*  residual network: add input data to output */
    if (residual_layer_flag) {
        /*  [axpy] y := alpha*x + y */
        g_blas_axpy(cudaHandles.cublasHandle,
                    noutputs*batch_size,
                    &alpha,
                    data->ptr,1,
                    R,1);
    }

    g_blas_axpy(cudaHandles.cublasHandle,
                noutputs*batch_size,
                &malpha,
                R,1,
                A,1);

    return nullptr;
}

Real* FullyConnectedLayer::bwd(Real *adjoint_prev){
    Real alpha = 1.0;
    Real beta = 0.0;

    /* ========================================== */
    /* [gemm] C := alpha*op( A )*op( B ) + beta*C */
    /* ========================================== */

    /* Derivative w.r.t activation */
    if (activationSelect(activation)) {
        checkCUDNN(cudnnActivationBackward(cudaHandles.cudnnHandle,actDesc,
                                           &dt,
                                           tensorDesc,d_a,
                                           tensorDesc,adjoint_prev,
                                           tensorDesc,d_z,
                                           &beta,
                                           tensorDesc,d_state_bar));
    } else {
        /* copy adjoint_prev into d_state_bar */
        g_blas_copy(cudaHandles.cublasHandle,
                    noutputs*batch_size,
                    adjoint_prev,1,
                    d_state_bar,1);

        /* rescale by dt */
        g_blas_scal(cudaHandles.cublasHandle,
                    noutputs*batch_size,
                    &dt,
                    d_state_bar,1);
    }

    /* Derivative w.r.t. weights:
     *      d_weights_bar = [d_input_ptr] * [adjoint_prev]^T
     */
    checkCudaErrors(g_blas_gemm(cudaHandles.cublasHandle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_T,
                                ninputs,noutputs,batch_size,
                                &alpha,
                                in.ptr,ninputs,
                                d_state_bar,noutputs,
                                &beta,
                                d_weights_bar,ninputs));

    /* Derivative w.r.t. bias:
     *      d_bias_bar = [adjoint_prev] * {1}
     */
    checkCudaErrors(g_blas_gemv(cudaHandles.cublasHandle,
                                CUBLAS_OP_N,
                                noutputs,batch_size,
                                &alpha,
                                d_state_bar,noutputs,
                                onevec,1,
                                &beta,
                                d_bias_bar,1));

    /* Derivative with respect to data (for previous layer):
     *      adjoint = dt*[d_weights]*[update_bar]
     */
    checkCudaErrors(g_blas_gemm(cudaHandles.cublasHandle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                ninputs,batch_size,noutputs,
                                &alpha,
                                d_weights,ninputs,
                                d_state_bar,noutputs,
                                &beta,
                                adjoint,ninputs));

    /*  residual network: add adjoint_prev to adjoint */
    if (residual_layer_flag) {
        /*  [axpy] y := alpha*x + y */
        g_blas_axpy(cudaHandles.cublasHandle,
                    noutputs*batch_size,
                    &alpha,
                    adjoint_prev,1,
                    adjoint,1);
    }
    return adjoint;
}

void FullyConnectedLayer::updateWeights(Real learning_rate){
    Real alpha = -learning_rate;
    /*  weights */
    checkCudaErrors(g_blas_axpy(cudaHandles.cublasHandle,
                                static_cast<int>(h_weights.size()),
                                &alpha,
                                d_weights_bar,1,
                                d_weights,1));
    /* bias */
    checkCudaErrors(g_blas_axpy(cudaHandles.cublasHandle,
                                static_cast<int>(h_bias.size()),
                                &alpha,
                                d_bias_bar,1,
                                d_bias,1));
}

size_t FullyConnectedLayer::setFwdTensors(cudnnTensorDescriptor_t *srcTensorDesc, const int nsamples){
    (void) srcTensorDesc;
    checkCUDNN(cudnnSetTensor4dDescriptor(tensorDesc,
                           /* format = */ CUDNN_TENSOR_NCHW,
                         /* dataType = */ DNN_REAL,
                                /* n = */ nsamples,
                                /* c = */ noutputs,
                                /* h = */ 1,
                                /* w = */ 1));
    return 0;
}

void FullyConnectedLayer::parametersInitializeHost(std::mt19937 &gen){
    /* Xavier weight filling */
    Real wfc = sqrt(3.0 / (ninputs * noutputs));

    /* distribution generator */
    std::uniform_real_distribution<> dfc(-wfc,wfc);

    /* randomize network */
    for(auto&& iter : h_weights)  iter = static_cast<Real>(dfc(gen));
    for(auto&& iter : h_bias)     iter = static_cast<Real>(dfc(gen));
}

void FullyConnectedLayer::parametersCopyLayerDevice(Layer *layer2copy){
    FullyConnectedLayer *fc = (FullyConnectedLayer *) layer2copy;

    checkCudaErrors(cudaMemcpy(d_weights,fc->d_weights,
                               sizeof(Real)*h_weights.size(),
                               cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemcpy(d_bias,fc->d_bias,
                               sizeof(Real)*h_bias.size(),
                               cudaMemcpyDeviceToDevice));
}

void FullyConnectedLayer::parametersDeviceToHost(){
    checkCudaErrors(cudaMemcpy(&h_weights[0],d_weights,sizeof(Real)*h_weights.size(),cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&h_bias[0],d_bias,sizeof(Real)*h_bias.size(),cudaMemcpyDeviceToHost));
}

void FullyConnectedLayer::parametersHostToDevice(){
    checkCudaErrors(cudaMemcpy(d_weights,&h_weights[0],sizeof(Real)*h_weights.size(),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_bias,&h_bias[0],sizeof(Real)*h_bias.size(),cudaMemcpyHostToDevice));
}

void FullyConnectedLayer::parametersHostToDeviceAsync(){
    checkCudaErrors(cudaMemcpyAsync(d_weights,&h_weights[0],sizeof(Real)*h_weights.size(),cudaMemcpyHostToDevice,streamID[0]));
    checkCudaErrors(cudaMemcpyAsync(d_bias,&h_bias[0],sizeof(Real)*h_bias.size(),cudaMemcpyHostToDevice,streamID[0]));
}

bool FullyConnectedLayer::fromFile(const char *fileprefix){
    std::stringstream ssf, ssbf;
    ssf << fileprefix << ".bin";
    ssbf << fileprefix << ".bias.bin";

    /* Read weights file */
    FILE *fp = fopen(ssf.str().c_str(), "rb");
    if (!fp) {
        printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
        return false;
    }
    fread(&h_weights[0], sizeof(Real), ninputs * noutputs, fp);
    fclose(fp);

    /* Read bias file */
    fp = fopen(ssbf.str().c_str(), "rb");
    if (!fp) {
        printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
        return false;
    }
    fread(&h_bias[0], sizeof(Real), noutputs, fp);
    fclose(fp);
    return true;
}

void FullyConnectedLayer::toFile(const char *fileprefix){
    std::stringstream ssf, ssbf;
    ssf << fileprefix << ".bin";
    ssbf << fileprefix << ".bias.bin";

    /* Write weights file */
    FILE *fp = fopen(ssf.str().c_str(), "wb");
    if (!fp) {
        printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
        exit(2);
    }
    fwrite(&h_weights[0], sizeof(Real), ninputs * noutputs, fp);
    fclose(fp);

    /* Write bias file */
    fp = fopen(ssbf.str().c_str(), "wb");
    if (!fp) {
        printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
        exit(2);
    }
    fwrite(&h_bias[0], sizeof(Real), noutputs, fp);
    fclose(fp);
}

void FullyConnectedLayer::dumpParameters(){
    parametersDeviceToHost();
    printf("FC Weights:\n");
    for(int i = 0; i < h_weights.size(); ++i){
        printf("W[%d] = ",i);
        (h_weights[i] < 0) ?
            printf( "%.15f\n",h_weights[i]):
            printf(" %.15f\n",h_weights[i]);
    }
}