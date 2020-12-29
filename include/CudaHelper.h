/**
 * File:   CudaHelper.h
 * Author: akirby
 *
 * Created on April 7, 2020, 1:23 PM
 */

#ifndef CUDAHELPER_H
#define CUDAHELPER_H

/* system header files */
#include <iostream>
#include <sstream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>
#include <nvToolsExt.h>

typedef struct {
    cudnnTensorDescriptor_t *tensorDesc;
    Real *ptr;
}
data_t;

typedef struct {
    cublasHandle_t cublasHandle;
    cudnnHandle_t cudnnHandle;
}
cuda_info_t;

/* ========================================== *
 * Error Handling:                            *
 * Adapted from the CUDNN classification code *
 * sample: https://developer.nvidia.com/cuDNN *
 * ========================================== */
#define FatalError(s) do {                                             \
    std::stringstream _where,_message;                                 \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

static inline
void CudaMallocCheck(void **d_vec, size_t sizeInBytes){
    if(*d_vec != nullptr) cudaFree(*d_vec);
    checkCudaErrors(cudaMalloc(d_vec,sizeInBytes));
}

#endif /* CUDAHELPER_H */