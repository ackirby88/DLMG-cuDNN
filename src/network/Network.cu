/**
 * File:   Network.cu
 * Author: akirby
 *
 * Created on April 23, 2020, 12:37 PM
 */

/* header files */
#include "Network.h"
#include <unistd.h>

#define MIN(x,y)  (x)<(y) ? (x):(y)
#define MAX(x,y)  (x)>(y) ? (x):(y)

#define GLOBAL_LOOP \
    for (int i = 0; i < globalLayers[0].size(); ++i)

#define BLOCK_LOOP \
    for (int i = 0; i < blocks[0].size(); ++i)

#define REVERSE_BLOCK_LOOP \
    for (int i = blocks[0].size() - 1; i >= 0; --i)

#define ROOT if(global_rank == 0)

/* =========== */
/* GPU Kernels */
/* =========== */
__global__ void FillOnes(Real *vec, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    vec[idx] = 1.0;
}

/* ================= */
/* Class Constructor */
/* ================= */
Network::Network(DLMG &dlmg_) :
    dlmg(dlmg_),
    globalLayers_count(0)
{
    globalLayers.resize(dlmg.multigrid.nlevels);
    blocks.resize(dlmg.multigrid.nlevels);
    h_out.resize(dlmg.multigrid.nlevels);
    h_adjoint.resize(dlmg.multigrid.nlevels);

    for(int i = 0; i < h_out.size(); ++i) h_out[i] = nullptr;
    for(int i = 0; i < h_adjoint.size(); ++i) h_adjoint[i] = nullptr;

    asyncCudaStreams = nullptr;
    d_onevec = nullptr;
    d_data = nullptr;
    d_labels = nullptr;
    checkCUDNN(cudnnCreateTensorDescriptor(&dataTensor));
};

Network::Network(DLMG &dlmg_, std::vector<Layer*> &layers_) :
    dlmg(dlmg_),
    globalLayers_count(0)
{
    globalLayers.resize(dlmg.multigrid.nlevels);

    for (int i = 0; i < layers_.size(); ++i) {
        globalLayers_count++;
        globalLayers[0].push_back(layers_[i]);
    }

    blocks.resize(dlmg.multigrid.nlevels);
    h_out.resize(dlmg.multigrid.nlevels);
    h_adjoint.resize(dlmg.multigrid.nlevels);

    for(int i = 0; i < h_out.size(); ++i) h_out[i] = nullptr;
    for(int i = 0; i < h_adjoint.size(); ++i) h_adjoint[i] = nullptr;

    d_onevec = nullptr;
    d_data = nullptr;
    d_labels = nullptr;
    checkCUDNN(cudnnCreateTensorDescriptor(&dataTensor));
}

Network::~Network(){
    checkCUDNN(cudnnDestroyTensorDescriptor(dataTensor));
    checkCudaErrors(cudaFree(d_data)); d_data = nullptr;
    checkCudaErrors(cudaFree(d_labels)); d_labels = nullptr;
    checkCudaErrors(cudaFree(d_onevec)); d_onevec = nullptr;

    for (int i = 0; i < h_out.size(); ++i) {
        if(h_out[i] != nullptr) free(h_out[i]); h_out[i] = nullptr;
    }
    for (int i = 0; i < h_adjoint.size(); ++i) {
        if(h_adjoint[i] != nullptr) free(h_adjoint[i]); h_adjoint[i] = nullptr;
    }
}

/* ============= */
/* Class Methods */
/* ============= */
void Network::update(Real learning_rate){
    BLOCK_LOOP {
        blocks[0][i]->updateWeights(learning_rate);
    }
}

void Network::fit(dataset_t *dataset, int batch_size, int nepochs){
    multigrid_t &mg = dlmg.multigrid;
    int train_size = dataset->train_size;
    int channels = dataset->channels;
    int height = dataset->height;
    int width = dataset->width;

    data_t *data_ptr;
    Real *label_ptr;

    timerdata_t time_data;
    time_data.comm_time = 0.0;
    time_data.comp_time = 0.0;
    time_data.total_time = 0.0;

    size_t bdata_size = batch_size*width*height*channels;
    this->batch_size = batch_size;

    /* ====================================================================== */
    /* unit vector for gradient across batches */
    checkCudaErrors(cudaMalloc(&d_onevec,sizeof(Real) * batch_size));
    FillOnes<<<RoundUp(batch_size, BW), BW>>>(d_onevec, batch_size);

    /* set data pointers for model parallelism */
    if (model_rank == 0) {
        /* GPU data structures    | Buffer | Element     | N          | C        | H      | W    */
        checkCudaErrors(cudaMalloc(&d_data, sizeof(Real) * batch_size * channels * height * width));

        data_ptr = new data_t;
        data_ptr->tensorDesc = &dataTensor;
        data_ptr->ptr = d_data;
    } else {
        data_ptr = leftGhost[0]->getOutDevice();
    }

    /* set adjoint pointers for model parallelism */
    if (model_rank == model_nranks - 1) {
        /* GPU data structures    | Buffer   | Element     | N          | C  | H | W */
        checkCudaErrors(cudaMalloc(&d_labels, sizeof(Real) * batch_size * 1  * 1 * 1));
        label_ptr = d_labels;
    } else {
        label_ptr = rightGhost[0]->getAdjointDevice();
    }

    /* set buffers */
    for (int level = 0; level < mg.nlevels; ++level) {
        setNetworkBuffers(batch_size,level);
    }
    /* ====================================================================== */

    /* Train network */
    /* ====================================================================== */
    ROOT{printf("\n"); BREAK_LINE_1; printf("[DLMG] Training...\n"); BREAK_LINE_2;}

    double time_total = 0.0;
    Real s_time = mpi_timer();
    synchronizeNetwork(0);
    uint trained_iterations = 0;
    for (int epoch = 0; epoch < nepochs; ++epoch) {
        Real t1 = mpi_timer();
        for (int iter = 0; iter < train_size; iter += batch_size) {
            int imageid = iter % (train_size / batch_size);

            /* set data and label values */
            if (model_rank == 0) {
                /* copy current batch to device */
                checkCudaErrors(cudaMemcpy(d_data,
                                           &dataset->h_train_data[imageid*bdata_size],
                                           sizeof(Real)*bdata_size,
                                           cudaMemcpyHostToDevice));
                                         //blocks[0].front()->cudaHandles.streamID));
            }
            if (model_rank == model_nranks - 1) {
                checkCudaErrors(cudaMemcpy(d_labels,
                                           &dataset->h_train_labels[imageid*batch_size],
                                           sizeof(Real)*batch_size,
                                           cudaMemcpyHostToDevice));
                                         //blocks[0].back()->cudaHandles.streamID));
            }

            /* Forward propagation */
            ROOT{printf("Iteration %d complete\n",iter);}
            MPI_Barrier(MPI_COMM_WORLD);
            double st = mpi_timer();
            if(iter){
                Multigrid_fwd(data_ptr, batch_size, 0, trained_iterations, 0);
            } else {
                synchronizeNetwork(0);
                time_data.comm_time = 0.0;
                time_data.comp_time = 0.0;
                time_data.total_time = 0.0;
                fwd_sync(&time_data,data_ptr, batch_size, 0, 0);
            }
            checkCudaErrors(cudaDeviceSynchronize());
            MPI_Barrier(MPI_COMM_WORLD);
            double et = mpi_timer();
            ROOT{printf("Fwd Time[%d]: %f\n",iter,et-st);}
            time_total += et - st;

            //displayFwdOutput(10,0);

            synchronizeNetwork(0);
            //fwd_sync(data_ptr, batch_size, 0, 0);
            bwd_sync(label_ptr, 0); //displayBwdOutput(10,0);

            /* Compute learning rate  (decaying ~1/T) */
            Real learningRate = (Real) FLAGS_learning_rate * pow((1.0 + FLAGS_lr_gamma*trained_iterations),(-FLAGS_lr_power));
            ++trained_iterations;

            /* Update weights */
            update(learningRate);
            asynchronizeNetwork(0);
        }
        checkCudaErrors(cudaDeviceSynchronize());
        Real t2 = mpi_timer();
        ROOT {
            printf("[train] epoch: %d, cpu time: %f sec, fwd time: %f sec, iterations: %d\n",
                    epoch,t2-t1,time_total,trained_iterations);
        }

//        /* Classification Error Testing */
//        if ((epoch+1) % dataset->test_accuracy_interval == 0) {
//            synchronizeNetwork(0);
//                evaluate(dataset, FLAGS_classify, "test");
//
//                /* reset tensors */
//                setTensors(batch_size,0);
//            asynchronizeNetwork(0);
//        }
        ROOT {BREAK_LINE_2;}
    }
    Real e_time = mpi_timer();

    ROOT {
        printf("[DLMG] Training complete.\n");
        printf("[DLMG] Training time: %f sec\n",e_time - s_time);
        BREAK_LINE_1;
    }

    /* free data structures */
    if(model_rank == 0) delete data_ptr;
}

void Network::evaluate(dataset_t *dataset, int nclassifications, const char str[]){
    uint8_t *labels = dataset->h_test_labels;
    Real *images = dataset->h_test_data;
    int test_size = dataset->test_size;
    int channels = dataset->channels;
    int height = dataset->height;
    int width = dataset->width;

    data_t *data_ptr;
    size_t data_size = width*height*channels;

    timerdata_t time_data;
    time_data.comm_time = 0.0;
    time_data.comp_time = 0.0;
    time_data.total_time = 0.0;

    /* ====================================================================== */
    /* unit vector for gradient across batches */
    if(d_onevec == nullptr) {
        checkCudaErrors(cudaMalloc(&d_onevec,sizeof(Real)));
        FillOnes<<<RoundUp(1, BW), BW>>>(d_onevec, 1);
    }

    /* GPU data structures */
     /* set data pointers for model parallelism */
    if (model_rank == 0) {
        /* GPU data structures                          | Buffer | Element     | N | C        | H      | W     */
        if(d_data == nullptr) checkCudaErrors(cudaMalloc(&d_data, sizeof(Real) * 1 * channels * height * width));

        data_ptr = new data_t;
        data_ptr->tensorDesc = &dataTensor;
        data_ptr->ptr = d_data;
    } else {
        data_ptr = leftGhost[0]->getOutDevice();
    }

    /* set buffers */
    setNetworkBuffers(1,0);

    /* set network to sequential for testing */
    synchronizeNetwork(0);
    /* ====================================================================== */
    /* Classification Error Testing */
    /* ====================================================================== */
    double classification_error = 1.0;
    if(nclassifications < 0) nclassifications = (int)test_size;

    checkCudaErrors(cudaDeviceSynchronize());

    int num_errors = 0;
    double t1 = mpi_timer();
    for (int i = 0; i < nclassifications; ++i) {

        /* set data and label values */
        if(model_rank == 0) {
            /* copy current batch to device */
            checkCudaErrors(cudaMemcpy(d_data,
                                       &images[i*data_size],
                                       sizeof(Real)*data_size,
                                       cudaMemcpyHostToDevice));

        }

        /* Synchronous forward propagation test image */
        fwd_sync(&time_data,data_ptr, 1, 0, 0);

        /* Perform classification : FIXME SIZED */
        if(model_rank == model_nranks - 1) {
            std::vector<Real> class_vec(10);

            data_t *result = getOutDevice(0);
            checkCudaErrors(cudaMemcpy(&class_vec[0],
                                       result->ptr,
                                       sizeof(Real)*10,
                                       cudaMemcpyDeviceToHost));

            /* Determine classification according to maximal response */
            int chosen = 0;
            for (int id = 1; id < 10; ++id) {
                if (class_vec[chosen] < class_vec[id]) chosen = id;
            }
            if(chosen != labels[i]) ++num_errors;
        }
    }
    double t2 = mpi_timer();

    if(model_rank == model_nranks - 1) {
        printf("[%s] classification time: %f sec\n",str,t2 - t1);
        classification_error = (double) num_errors / (double) nclassifications;
        printf("[%s] classification accuracy (top-1): %.2f%% error (used %d images)\n",
                str,classification_error * 100.0, nclassifications);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    /* reset network CUDA Handles */
    asynchronizeNetwork(0);

    /* free data structures */
    if(model_rank == 0) delete data_ptr;
}
