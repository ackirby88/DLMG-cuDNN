cmake_minimum_required(VERSION 2.6)
project(DLMG-CUDNN)

# -------------------------------------------------------------------------- #
find_package(CUDA 6.5 REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# -------------------------------------------------------------------------- #
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug Mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
#     -gencode; arch=compute_35,code=sm_35;
#     -gencode; arch=compute_52,code=sm_52;
#     -gencode; arch=compute_50,code=compute_50;
     -std=c++11; -g; -lineinfo; -Xcompiler -fopenmp; -ggdb)
else()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
#     -gencode; arch=compute_35,code=sm_35;
#     -gencode; arch=compute_52,code=sm_52;
#     -gencode; arch=compute_50,code=compute_50;
     -std=c++11; -O3; -DNDEBUG; -Xcompiler -fopenmp; -DNDEBUG)
endif()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Addresses a bug where code is not compiled as C++11 in non-CUDA code and older g++ versions
##set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# -------------------------------------------------------------------------- #
# ==================== #
# Library Header Files #
# ==================== #
#set(CUDNN_PATH /usr/local/cudnn-8.0.0.180-cuda11.0/include)
set(CUDNN_PATH /state/partition1/llgrid/pkg/anaconda/anaconda3-2020b/pkgs/cudnn-7.6.5-cuda10.1_0)
set(CUDA_PATH /state/partition1/llgrid/pkg/anaconda/anaconda3-2020b/pkgs/cudatoolkit-10.1.243-h6bb024c_0)

include_directories(${CUDNN_PATH} ${CUDNN_PATH}/include)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
link_directories(${CUDNN_PATH} ${CUDNN_PATH}/lib ${CUDNN_PATH}/lib64)
link_directories(${CUDA_PATH}/lib)

#include_directories(${CUDNN_PATH} ${CUDNN_PATH}/include)
#include_directories(SYSTEM ${MPI_INCLUDE_PATH})
#link_directories(${CUDNN_PATH} ${CUDNN_PATH}/lib ${CUDNN_PATH}/lib64)

# ========================= #
# NVTX CUDA Profile Library #
# ========================= #
find_library(NVTX_LIBRARY nvToolsExt PATHS /state/partition1/llgrid/pkg/anaconda/anaconda3-2020b/pkgs/cudatoolkit-10.1.243-h6bb024c_0/lib)
find_library(CUBLAS_LIBRARY cublas PATHS /state/partition1/llgrid/pkg/anaconda/anaconda3-2020b/pkgs/cudatoolkit-10.1.243-h6bb024c_0/lib)
find_library(CUBLASLT_LIBRARY cublasLt PATHS /state/partition1/llgrid/pkg/anaconda/anaconda3-2020b/pkgs/cudatoolkit-10.1.243-h6bb024c_0/lib)

# -------------------------------------------------------------------------- #
# ========================= #
# GPU Architecture Language #
# ========================= #
add_definitions(-D__DLMG_CUDA__)
#add_definitions(-D__DLMG_HIP__)

# ========= #
# Precision #
# ========= #
#add_definitions(-DHALF_PRECISION)
add_definitions(-DSINGLE_PRECISION)
#add_definitions(-DDOUBLE_PRECISION)
#add_definitions(-DCOMPLEX_PRECISION)
#add_definitions(-DDOUBLE_COMPLEX_PRECISION)

# --------------------------------------------------------------------------- #
# ==================== #
# Project header files #
# ==================== #
include_directories(include)

# ==================== #
# Project source files #
# ==================== #
add_subdirectory(src)
