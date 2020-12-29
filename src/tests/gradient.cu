
/* header files */
#include "gradient.h"

__global__ void FillOnes(Real *vec, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) vec[idx] = 1.0;
}

void displayFwdOutput(Real *h_out, Real *d_out, int nsamples){
    checkCudaErrors(cudaMemcpy(h_out,
                               d_out,
                               sizeof(Real)*nsamples,
                               cudaMemcpyDeviceToHost));

    for(int k = 0; k < nsamples; ++k){
        printf("out[%d]: %.15f\n",k,h_out[k]);
    }
}

void gradient_check(int argc, char **argv){
    cudnnTensorDescriptor_t dataTensor;
    data_t train;
    std::mt19937 gen(1);

    /* ================== */
    /* network parameters */
    /* ================== */
    int batch_size = 1;
    int ninputs = 4;
    int noutputs = 1;
    //Real dt = 0.1;

    /* =============== */
    /* data structures */
    /* =============== */
    /* GPU data */
    Real *d_data,*d_onevec;
    checkCudaErrors(cudaMalloc(&d_data, sizeof(Real) * batch_size * ninputs));
    checkCudaErrors(cudaMalloc(&d_onevec,sizeof(Real) * batch_size));

    FillOnes<<<1,batch_size*ninputs>>>(d_data, batch_size * ninputs);
    FillOnes<<<1,batch_size>>>(d_onevec, batch_size);

    /* CPU data */
    Real *h_out = (Real *) malloc(batch_size*noutputs*sizeof(Real));

    /* ========================== */
    /* Build Network Architecture */
    /* ========================== */
    DLMG dlmg(argc,argv);
    Network model(dlmg);

    /* Network Layers */
    FullyConnectedLayer fc1(batch_size,
                            ninputs,
                            noutputs,
                            RELU);

    SoftmaxLayer softmax((int) batch_size,
                         fc1.getOutSize());

    model.add(fc1);
    model.add(softmax);

    model.initialize(batch_size);
    model.parametersInitializeHost(gen);
    model.parametersHostToDeviceAsync(0);

    /* ==================== */
    /* Data Input Structure */
    /* ==================== */
    train.tensorDesc = &dataTensor;
    train.ptr = d_data;

    /* Derivative Parameters */
    double eps = 1.0E-30;
    cuDoubleComplex h = make_cuDoubleComplex(0.0,eps);

    /* setup fully connected layer */
    fc1.setOneVector(d_onevec);
    fc1.setFwdTensors(&dataTensor, ninputs);
    fc1.dumpParameters();

    /* forward */
    data_t *output = fc1.fwd(&train,1);
    displayFwdOutput(h_out, output->ptr, batch_size*noutputs);

    /* shutdown */
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_onevec));
    free(h_out);
}
