#include "ConvolutionLayer.h"

ConvolutionLayer::ConvolutionLayer(int batch_size_,
                                   int in_channels_,
                                   int in_width_,
                                   int in_height_,
                                   int kernel_size_,
                                   int out_channels_,
                                   int pad_size_,
                                   int activation_) :
    Layer(layerType::CONVOLUTION),
    in_channels(in_channels_),
    in_width(in_width_),
    in_height(in_height_),
    kernel_size(kernel_size_),
    out_channels(out_channels_),
    pad_size(pad_size_),
    activation(activation_),
    residual_layer_flag(0),
    dt(1.0)
{
    out_width = in_width - kernel_size_ + 2*pad_size + 1;
    out_height = in_height - kernel_size_ + 2*pad_size + 1;

    batch_size = batch_size_;
    ninputs = in_channels * in_width * in_height;
    noutputs = out_channels * out_height * out_width;

    /* network parameters */
    d_state_bar = nullptr;
    adjoint = nullptr;
    d_z = nullptr;
    d_a = nullptr;
    d_bias = nullptr;
    d_kernel = nullptr;
    d_bias_bar = nullptr;
    d_kernel_bar = nullptr;

    /* CUDNN Data Structures */
    checkCUDNN(cudnnCreateTensorDescriptor(&tensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc,
                           /* format = */ CUDNN_TENSOR_NCHW,
                         /* dataType = */ DNN_REAL,
                                /* n = */ 1,
                                /* c = */ out_channels,
                                /* h = */ 1,
                                /* w = */ 1));

    if (activationSelect(activation)) {
        checkCUDNN(cudnnCreateActivationDescriptor(&actDesc));
        checkCUDNN(cudnnSetActivationDescriptor(actDesc,
                                   /* mode = */ activationSelect(activation),
                             /* reluNanOpt = */ CUDNN_PROPAGATE_NAN,
                                   /* coef = */ 0.0));
    }
}

/* ResNet Constructor */
ConvolutionLayer::ConvolutionLayer(int batch_size_,
                                   int in_channels_,
                                   int in_width_,
                                   int in_height_,
                                   int kernel_size_,
                                   int activation_,
                                   Real dt_) :
    Layer(layerType::CONVOLUTION),
    in_channels(in_channels_),
    in_width(in_width_),
    in_height(in_height_),
    out_channels(in_channels_),
    activation(activation_),
    residual_layer_flag(1),
    dt(dt_)
{
    /* check if kernel size is odd; if even, we need to modify it to make sizes conform */
    kernel_size = (kernel_size_%2==1) ? (kernel_size_):(kernel_size_- 1);

    pad_size = (kernel_size - 1) / 2;
    out_width = in_width - kernel_size + 2*pad_size + 1;
    out_height = in_height - kernel_size + 2*pad_size + 1;

    batch_size = batch_size_;
    ninputs = in_channels * in_width * in_height;
    noutputs = out_channels * out_height * out_width;

    /* make sure dimensions match for residual network */
    assert(noutputs == ninputs);
    assert(out_width == in_width);
    assert(out_height == in_height);

    /* network parameters */
    d_state_bar = nullptr;
    adjoint = nullptr;
    d_z = nullptr;
    d_a = nullptr;
    d_bias = nullptr;
    d_kernel = nullptr;
    d_bias_bar = nullptr;
    d_kernel_bar = nullptr;

    /* CUDNN Data Structures */
    checkCUDNN(cudnnCreateTensorDescriptor(&tensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc,
                           /* format = */ CUDNN_TENSOR_NCHW,
                         /* dataType = */ DNN_REAL,
                                /* n = */ 1,
                                /* c = */ out_channels,
                                /* h = */ 1,
                                /* w = */ 1));

    if (activationSelect(activation)) {
        checkCUDNN(cudnnCreateActivationDescriptor(&actDesc));
        checkCUDNN(cudnnSetActivationDescriptor(actDesc,
                                   /* mode = */ activationSelect(activation),
                             /* reluNanOpt = */ CUDNN_PROPAGATE_NAN,
                                   /* coef = */ 0.0));
    }
}

ConvolutionLayer::ConvolutionLayer(const ConvolutionLayer &layer) :
    Layer(layerType::CONVOLUTION),
    in_channels(layer.in_channels),
    in_width(layer.in_width),
    in_height(layer.in_height),
    out_channels(layer.out_channels),
    activation(layer.activation),
    residual_layer_flag(layer.residual_layer_flag),
    dt(layer.dt),
    kernel_size(layer.kernel_size),
    pad_size(layer.pad_size),
    out_width(layer.out_width),
    out_height(layer.out_height)
{
    ninputs = layer.ninputs;
    noutputs = layer.noutputs;
    batch_size = layer.batch_size;
    global_idx = layer.global_idx;

    /* network parameters */
    d_state_bar = nullptr;
    adjoint = nullptr;
    d_z = nullptr;
    d_a = nullptr;

    /* alias layer parameters */
    fine_level_flag = 0;
    d_bias = layer.d_bias;
    d_kernel = layer.d_kernel;
    d_bias_bar = layer.d_bias_bar;
    d_kernel_bar = layer.d_kernel_bar;

    /* CUDNN Data Structures */
    checkCUDNN(cudnnCreateTensorDescriptor(&tensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc,
                           /* format = */ CUDNN_TENSOR_NCHW,
                         /* dataType = */ DNN_REAL,
                                /* n = */ 1,
                                /* c = */ out_channels,
                                /* h = */ 1,
                                /* w = */ 1));

    if (activationSelect(activation)) {
        checkCUDNN(cudnnCreateActivationDescriptor(&actDesc));
        checkCUDNN(cudnnSetActivationDescriptor(actDesc,
                                   /* mode = */ activationSelect(activation),
                             /* reluNanOpt = */ CUDNN_PROPAGATE_NAN,
                                   /* coef = */ 0.0));
    }
}

ConvolutionLayer::~ConvolutionLayer(){
    checkCUDNN(cudnnDestroyTensorDescriptor(tensorDesc));
    checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));

    if (fine_level_flag) {
        checkCudaErrors(cudaFree(d_bias));
        checkCudaErrors(cudaFree(d_kernel));
        checkCudaErrors(cudaFree(d_bias_bar));
        checkCudaErrors(cudaFree(d_kernel_bar));
    }

    checkCudaErrors(cudaFree(d_state_bar));
    checkCudaErrors(cudaFree(d_z));
    checkCudaErrors(cudaFree(adjoint));

    if(A) checkCudaErrors(cudaFree(A));
    if(G) checkCudaErrors(cudaFree(G));
    if(R) checkCudaErrors(cudaFree(R));

    if (activationSelect(activation)) {
        checkCUDNN(cudnnDestroyActivationDescriptor(actDesc));
        checkCudaErrors(cudaFree(d_a));
    }
}

Layer* ConvolutionLayer::clone(){
    return new ConvolutionLayer(*this);
}

void ConvolutionLayer::layerAllocateHost(){
    h_kernel.resize(in_channels * kernel_size * kernel_size * out_channels);
    h_bias.resize(out_channels);
}

void ConvolutionLayer::layerAllocateDevice(const int gpu_id,char multigrid_flag){
    /* ============================ */
    /* Allocate GPU Data Structures */
    /* ============================ */
    /* set GPU id to allocate layer */
    checkCudaErrors(cudaSetDevice(gpu_id));

    /* Device Network Parameters */
    if (fine_level_flag) {
        checkCudaErrors(cudaMalloc(&d_bias, sizeof(Real) * h_bias.size()));
        checkCudaErrors(cudaMalloc(&d_kernel, sizeof(Real) * h_kernel.size()));
        checkCudaErrors(cudaMalloc(&d_bias_bar, sizeof(Real) * h_bias.size()));
        checkCudaErrors(cudaMalloc(&d_kernel_bar, sizeof(Real) * h_kernel.size()));
    }

    /* Device Layer States    | Buffer     | Data Type    | N          | C            | H          | W        | */
    checkCudaErrors(cudaMalloc(&d_z,         sizeof(Real) * batch_size * out_channels * out_height * out_width));
    checkCudaErrors(cudaMalloc(&d_state_bar, sizeof(Real) * batch_size * out_channels * out_height * out_width));
    checkCudaErrors(cudaMalloc(&adjoint,     sizeof(Real) * batch_size *  in_channels *  in_height *  in_width));

    if (multigrid_flag) {
        this->multigrid_flag = multigrid_flag;
        checkCudaErrors(cudaMalloc(&A, sizeof(Real) * batch_size * out_channels * out_height * out_width));
        checkCudaErrors(cudaMalloc(&G, sizeof(Real) * batch_size * out_channels * out_height * out_width));
        checkCudaErrors(cudaMalloc(&R, sizeof(Real) * batch_size * out_channels * out_height * out_width));

        g_blas_memset_async(A, 0, sizeof(Real) * batch_size * noutputs, streamID[0]);
        g_blas_memset_async(G, 0, sizeof(Real) * batch_size * noutputs, streamID[0]);
        g_blas_memset_async(R, 0, sizeof(Real) * batch_size * noutputs, streamID[0]);
    }

    if (activationSelect(activation)) {
        checkCudaErrors(cudaMalloc(&d_a,     sizeof(Real) * batch_size * out_channels * out_height * out_width));
    } else {
        /* alias activation state with state */
       d_a = d_z;
    }
    g_blas_memset_async(d_z, 0, sizeof(Real) * batch_size * noutputs, streamID[0]);
    g_blas_memset_async(d_a, 0, sizeof(Real) * batch_size * noutputs, streamID[0]);

    /* set output data info */
    out.tensorDesc = &tensorDesc;
    out.ptr = d_a;
}

data_t* ConvolutionLayer::fwd(const data_t *data, const int nsamples, const char add_source){
    (void) nsamples;

    Real malpha = -1.0;
    Real alpha = 1.0;
    Real beta = 0.0;

    /* save input data for adjoint */
    in.ptr = data->ptr;
    in.tensorDesc = data->tensorDesc;

    if(!residual_layer_flag && add_source) {
        g_blas_copy(cudaHandles.cublasHandle,
                    noutputs*batch_size,
                    G,1,
                    d_a,1);
        out.tensorDesc = &tensorDesc;
        out.ptr = d_a;
        return &out;
    }

    //checkCudaErrors(cudaStreamSynchronize(streamID[0]));
    checkCUDNN(cudnnConvolutionForward(cudaHandles.cudnnHandle,
                                       &alpha,
                                       data->tensorDesc[0],data->ptr,
                                       filterDesc,d_kernel,
                                       convDesc,convFwdAlgo,
                                       workspaceFwd,workspaceSizeFwd,
                                       &beta,
                                       tensorDesc,d_z));

    checkCUDNN(cudnnAddTensor(cudaHandles.cudnnHandle,
                              &alpha,biasTensorDesc,d_bias,
                              &alpha,tensorDesc,d_z));

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

data_t* ConvolutionLayer::formA(data_t *data,const int nsamples){
    Real malpha = -1.0;
    Real alpha = 1.0;
    Real beta = 0.0;

    //checkCudaErrors(cudaStreamSynchronize(streamID[0]));

    /* ============================ */
    /* form A:= d_a(old) - d_a(new) */
    /* ============================ */

    /* copy d_a(old) into A */
    g_blas_copy(cudaHandles.cublasHandle,
                noutputs*batch_size,
                d_a,1,
                A,1);

    /* return if opening layer */
    if(!residual_layer_flag) {
        g_blas_copy(cudaHandles.cublasHandle,
                noutputs*batch_size,
                d_a,1,
                G,1);
        return nullptr;
    }

    checkCUDNN(cudnnConvolutionForward(cudaHandles.cudnnHandle,
                                       &alpha,
                                       data->tensorDesc[0],data->ptr,
                                       filterDesc,d_kernel,
                                       convDesc,convFwdAlgo,
                                       workspaceFwd,workspaceSizeFwd,
                                       &beta,
                                       tensorDesc,R));

    checkCUDNN(cudnnAddTensor(cudaHandles.cudnnHandle,
                              &alpha,biasTensorDesc,d_bias,
                              &alpha,tensorDesc,R));

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

    /* ======================== */
    /* y[j] = alpha*x[j] + y[j] */
    /* y := A contains d_a(old) */
    /* x := R contains d_a(new) */
    /* alpha = -1.0             */
    /* ======================== */
    g_blas_axpy(cudaHandles.cublasHandle,
                noutputs*batch_size,
                &malpha,
  /* x, incx */ R,1,
  /* y, incy */ A,1);
    return nullptr;
}

Real* ConvolutionLayer::bwd(Real *adjoint_prev){
    Real alpha = 1.0;
    Real beta = 0.0;

    /* Derivative w.r.t activation */
    checkCUDNN(cudnnActivationBackward(cudaHandles.cudnnHandle,actDesc,
                                       &dt,
                                       tensorDesc,d_a,
                                       tensorDesc,adjoint_prev,
                                       tensorDesc,d_z,
                                       &beta,
                                       tensorDesc,d_state_bar));

    checkCUDNN(cudnnConvolutionBackwardBias(cudaHandles.cudnnHandle,
                                            &alpha,
                                            tensorDesc,d_state_bar,
                                            &beta,
                                            biasTensorDesc,d_bias_bar));

    checkCUDNN(cudnnConvolutionBackwardFilter(cudaHandles.cudnnHandle,
                                              &alpha,
                                              in.tensorDesc[0],in.ptr,
                                              tensorDesc,d_state_bar,
                                              convDesc,convBwdFilterAlgo,
                                              workspaceBwd,workspaceSizeBwd,
                                              &beta,
                                              filterDesc,d_kernel_bar));

    checkCUDNN(cudnnConvolutionBackwardData(cudaHandles.cudnnHandle,
                                            &alpha,
                                            filterDesc,d_kernel,
                                            tensorDesc,d_state_bar,
                                            convDesc,convBwdDataAlgo,
                                            workspaceBwd,workspaceSizeBwd,
                                            &beta,
                                            in.tensorDesc[0],adjoint));

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

void ConvolutionLayer::updateWeights(Real learning_rate){
    Real alpha = -learning_rate;
    /* weights */
    checkCudaErrors(g_blas_axpy(cudaHandles.cublasHandle,
                                static_cast<int>(h_kernel.size()),
                                &alpha,
                                d_kernel_bar,1,
                                d_kernel,1));
    /* bias */
    checkCudaErrors(g_blas_axpy(cudaHandles.cublasHandle,
                                static_cast<int>(h_bias.size()),
                                &alpha,
                                d_bias_bar,1,
                                d_bias,1));
}

size_t ConvolutionLayer::setFwdTensors(cudnnTensorDescriptor_t *srcTensorDesc,
                                       const int nsamples){
    size_t sizeInBytes = 0;

    int n = nsamples;
    int c = in_channels;
    int h = in_height;
    int w = in_width;

    checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc[0],
                     /*       format = */ CUDNN_TENSOR_NCHW,
                     /*     dataType = */ DNN_REAL,
                     /*   batch_size = */ n,
                     /*     channels = */ c,
                     /* image_height = */ h,
                     /*  image_width = */ w));

    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
                     /*     dataType = */ DNN_REAL,
                     /*       format = */ CUDNN_TENSOR_NCHW,
                     /* out_channels = */ out_channels,
                     /*  in_channels = */ in_channels,
                     /*kernel_height = */ kernel_size,
                     /* kernel_width = */ kernel_size));

#if CUDNN_MAJOR > 5
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                     /*        pad_height = */ pad_size,
                     /*         pad_width = */ pad_size,
                     /*   vertical_stride = */ 1,
                     /* horizontal_stride = */ 1,
                     /*   dilation_height = */ 1,
                     /*    dilation_width = */ 1,
                     /*              mode = */ CUDNN_CROSS_CORRELATION,
                     /*       computeType = */ DNN_REAL));
#else
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                     /*        pad_height = */ pad_size,
                     /*         pad_width = */ pad_size,
                     /*   vertical_stride = */ 1,
                     /* horizontal_stride = */ 1,
                     /*   dilation_height = */ 1,
                     /*    dilation_width = */ 1,
                     /*              mode = */ CUDNN_CROSS_CORRELATION));
#endif

    // Find dimension of convolution output
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                     srcTensorDesc[0],
                                                     filterDesc,
                                                     &n,&c,&h,&w));

    checkCUDNN(cudnnSetTensor4dDescriptor(tensorDesc,
                                          CUDNN_TENSOR_NCHW,
                                          DNN_REAL,
                                          n,c,h,w));

    cudnnConvolutionFwdAlgoPerf_t results[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    int returnedAlgoCount = -1;

    checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudaHandles.cudnnHandle,
                                                    srcTensorDesc[0],
                                                    filterDesc,
                                                    convDesc,
                                                    tensorDesc,
                                                    requestedAlgoCount,
                                                    &returnedAlgoCount,
                                                    results));
    convFwdAlgo = results[0].algo;

//    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudaHandles.cudnnHandle,
//                                                   srcTensorDesc[0],
//                                                   filterDesc,
//                                                   convDesc,
//                                                   tensorDesc,
//                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
//                                                   0,
//                                                   &convFwdAlgo));

    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudaHandles.cudnnHandle,
                                                       srcTensorDesc[0],
                                                       filterDesc,
                                                       convDesc,
                                                       tensorDesc,
                                                       convFwdAlgo,
                                                       &sizeInBytes));

    this->workspaceSizeFwd = sizeInBytes;
    CudaMallocCheck(&this->workspaceFwd,this->workspaceSizeFwd);
    return sizeInBytes;
}

size_t ConvolutionLayer::setBwdTensors(cudnnTensorDescriptor_t *srcTensorDesc){
    size_t sizeInBytes = 0;
    size_t tmpsize = 0;

    // If backprop filter algorithm was requested
    if (convBwdFilterAlgo) {
//        checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudaHandles.cudnnHandle,
//                                                                 srcTensorDesc[0],
//                                                                 tensorDesc,
//                                                                 convDesc,
//                                                                 filterDesc,
//                                                                 CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
//                                                                 0,
//                                                                 &convBwdFilterAlgo));
    convBwdFilterAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

        checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudaHandles.cudnnHandle,
                                                                  srcTensorDesc[0],
                                                                  tensorDesc,
                                                                  convDesc,
                                                                  filterDesc,
                                                                  convBwdFilterAlgo,
                                                                  &tmpsize));

        sizeInBytes = std::max(sizeInBytes,tmpsize);
    }

    // If backprop data algorithm was requested
    if (convBwdDataAlgo) {
//        checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudaHandles.cudnnHandle,
//                                                            filterDesc,
//                                                            tensorDesc,
//                                                            convDesc,
//                                                            srcTensorDesc[0],
//                                                            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
//                                                            0,
//                                                            &convBwdDataAlgo));
        convBwdDataAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

        checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudaHandles.cudnnHandle,
                                                                filterDesc,
                                                                tensorDesc,
                                                                convDesc,
                                                                srcTensorDesc[0],
                                                                convBwdDataAlgo,
                                                                &tmpsize));
        sizeInBytes = std::max(sizeInBytes,tmpsize);
    }
    this->workspaceSizeBwd = sizeInBytes;
    CudaMallocCheck(&this->workspaceBwd,this->workspaceSizeBwd);
    return sizeInBytes;
}

void ConvolutionLayer::parametersInitializeHost(std::mt19937 &gen){
    /* Xavier weight filling */
    Real wconv = sqrt(3.0 / (kernel_size * kernel_size * in_channels));

    /* distribution generator */
    std::uniform_real_distribution<> dconv(-wconv,wconv);

    /* randomize network */
    for(auto&& iter : h_kernel) iter = static_cast<Real>(dconv(gen));
    for(auto&& iter : h_bias)   iter = static_cast<Real>(dconv(gen));
}

void ConvolutionLayer::parametersCopyLayerDevice(Layer *layer2copy){
    ConvolutionLayer *conv = (ConvolutionLayer *) layer2copy;

    checkCudaErrors(cudaMemcpy(d_kernel,conv->d_kernel,
                               sizeof(Real)*h_kernel.size(),
                               cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemcpy(d_bias,conv->d_bias,
                               sizeof(Real)*h_bias.size(),
                               cudaMemcpyDeviceToDevice));
}

void ConvolutionLayer::parametersDeviceToHost(){
    checkCudaErrors(cudaMemcpy(&h_kernel[0],d_kernel,sizeof(Real)*h_kernel.size(),cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&h_bias[0],d_bias,sizeof(Real)*h_bias.size(),cudaMemcpyDeviceToHost));
}

void ConvolutionLayer::parametersHostToDevice(){
    checkCudaErrors(cudaMemcpy(d_kernel,&h_kernel[0],sizeof(Real)*h_kernel.size(),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_bias,&h_bias[0],sizeof(Real)*h_bias.size(),cudaMemcpyHostToDevice));
}

void ConvolutionLayer::parametersHostToDeviceAsync(){
    checkCudaErrors(cudaMemcpyAsync(d_kernel,&h_kernel[0],sizeof(Real)*h_kernel.size(),cudaMemcpyHostToDevice,streamID[0]));
    checkCudaErrors(cudaMemcpyAsync(d_bias,&h_bias[0],sizeof(Real)*h_bias.size(),cudaMemcpyHostToDevice,streamID[0]));
}

bool ConvolutionLayer::fromFile(const char *fileprefix){
    std::stringstream ssf, ssbf;
    ssf << fileprefix << ".bin";
    ssbf << fileprefix << ".bias.bin";

    // Read kernel file
    FILE *fp = fopen(ssf.str().c_str(), "rb");
    if (!fp)
    {
        printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
        return false;
    }
    fread(&h_kernel[0], sizeof(Real), in_channels * out_channels * kernel_size * kernel_size, fp);
    fclose(fp);

    // Read bias file
    fp = fopen(ssbf.str().c_str(), "rb");
    if (!fp)
    {
        printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
        return false;
    }
    fread(&h_bias[0], sizeof(Real), out_channels, fp);
    fclose(fp);
    return true;
}

void ConvolutionLayer::toFile(const char *fileprefix){
    std::stringstream ssf, ssbf;
    ssf << fileprefix << ".bin";
    ssbf << fileprefix << ".bias.bin";

    // Write kernel file
    FILE *fp = fopen(ssf.str().c_str(), "wb");
    if (!fp) {
        printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
        exit(2);
    }
    fwrite(&h_kernel[0], sizeof(Real), in_channels * out_channels * kernel_size * kernel_size, fp);
    fclose(fp);

    // Write bias file
    fp = fopen(ssbf.str().c_str(), "wb");
    if (!fp) {
        printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
        exit(2);
    }
    fwrite(&h_bias[0], sizeof(Real), out_channels, fp);
    fclose(fp);
}