# force GNU linker to include all shared libraries
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")

set(DLMG_SRC
    utilities/mpi_utilities.cpp
    utilities/readubyte.cpp
    kernels/CudaKernels.cu
    containers/DLMG.cu
    layers/Layer.cpp
    layers/LayerBlock.cpp
    layers/SoftmaxLayer.cu
    layers/MaxPoolLayer.cpp
    layers/FullyConnectedLayer.cpp
    layers/ConvolutionLayer.cpp
    layers/GhostLayer.cu
    network/NetworkUtilities.cu
    network/NetworkInitialize.cu
    network/NetworkForward.cu
    network/NetworkBackward.cu
    network/NetworkMultigrid.cu
    network/Network.cu
)

cuda_add_library(dlmg SHARED ${MG_CUDA_H} ${DLMG_SRC})
#cuda_add_cublas_to_target(dlmg)

target_link_libraries(dlmg cudnn ${CUBLAS_LIBRARY} ${CUBLASLT_LIBRARY} ${MPI_C_LIBRARIES} ${NVTX_LIBRARY})
target_link_libraries(dlmg OpenMP::OpenMP_CXX)

cuda_add_executable(dlmg.gpu main.cu)
#cuda_add_cublas_to_target(dlmg.gpu)

target_link_libraries(dlmg.gpu dlmg)
target_link_libraries(dlmg.gpu cudnn ${CUBLAS_LIBRARY} ${CUBLASLT_LIBRARY} ${NVTX_LIBRARY})
target_link_libraries(dlmg.gpu ${MPI_C_LIBRARIES})

# ================================================= #

OPTION(UNIT_TEST "build unit tests" OFF)
if (UNIT_TEST)
  add_subdirectory(tests)
endif (UNIT_TEST)
