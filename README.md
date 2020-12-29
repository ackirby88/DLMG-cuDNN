# DLMG-cudnn
**D**eep **L**earning **M**ulti**G**rid (DLMG)-cudnn is a C++ implementation of an *layer-wise* parallelism algorithm based on nonlinear **F**ull **A**pproximation **S**cheme Multigrid for ODE deep neural networks. The deep learning kernels use CUDA's CUDNN library as the backend wrapped as C++ *Layer* Objects. The software implements parallelism via MPI.  

Details of the implementation can be found in [Layer-Parallel Training with GPU Concurrency of Deep Residual Neural Networks Via Nonlinear Multigrid](https://arxiv.org/abs/2007.07336). This work was submitted to and recieved a best paper finalist award at the [2020 IEEE HPEC Conference](http://www.ieee-hpec.org/).

## Dependencies
- CUDNN - 10.1 or higher
- CUDA - 6.5 or higher
- MPI
- OpenMP
- CMAKE - 2.6 or higher
- (optional) NVTX for profiling

## Compiling
At present, the CMAKE file is hard-coded for the CUDNN and CUDA library dependencies.  
In `DLMG-cudnn/CMakeLists.txt`, edit lines 35, 36, 50-52 pointing the proper libraries.

In the source directory, make a new folder named `build`. Proceed into `build`, and configure using cmake:
```
cd build
cmake ..
```
After configuring, build the code by typing `make`.

## Execution
After compiling the code, there should be a new directory `DLMG-cudnn/build/src` which will contain the following executables and libraries: 
```
dlmg.gpu
libdlmg.so
```
Be sure to check the codes have the proper library linking: `ldd dlmg.gpu`.

Next, copy (or hyperlink) the training data and labels to the current directory `DLMG-cudnn/build/src`.  
**NOTE:** Only the MNIST data set is supoosed and presently hard-coded in `DLMG-cudnn/include/Network.h` lines 50-56. 

Once the data is in place, the command-line executions is as follows:  
```
mpirun -np <nranks> ./dlmg.gpu
```

## Hard-Coded Configurations
1. Several parameters and file paths are hardcoded here (e.g. training data, learning rate, etc.) : `DLMG-cudnn/include/Network.h`.  
2. The network architecture is hardcoded here: `DLMG-cudnn/src/main.cu`.  
3. The following layer-parallelization parameters are hardcoded here: `DLMG-cudnn/src/containers/DLMG.cu`.  
Lines 34-36.
```C++
    /*set number of ranks per network for network parallelism */
    partition_info.sequential = 0;         // serial-wise layer execution
    partition_info.nranks_per_model = 128; // number of MPI ranks to partition the network layers
    partition_info.nlayers_per_block = 64; // number of layers per layer block (see paper)
```
