#pragma once

#include <iostream>
#include <cuda_runtime.h>

class CudaManager
{
private:
    int deviceCount;
    cudaDeviceProp deviceProp;
    cudaError_t cudaStatus;
    unsigned int usedBlocks, usedThreads, usedSharedMemory, usedRegisters;

public:
    CudaManager();
    ~CudaManager();
    void printDeviceProperties();
    void updateDeviceProperties();
    void setUsedBlocks(unsigned int &usedBlocks)
    {
        this->usedBlocks = usedBlocks;
    }
    void setUsedThreads(unsigned int &usedThreads)
    {
        this->usedThreads = usedThreads;
    }
    void setUsedSharedMemory(unsigned int &usedSharedMemory)
    {
        this->usedSharedMemory = usedSharedMemory;
    }
    void setUsedRegisters(unsigned int &usedRegisters)

    {
        this->usedRegisters = usedRegisters;
    }
    void getLeftoverResources(unsigned int &leftoverBlocks,
                              unsigned int &leftoverThreads,
                              unsigned int &leftoverSharedMemory,
                              unsigned int &leftoverRegisters)
    {
        leftoverBlocks = this->deviceProp.maxBlocksPerMultiProcessor - this->usedBlocks;
        leftoverThreads = this->deviceProp.maxThreadsPerMultiProcessor - this->usedThreads;
        leftoverSharedMemory = this->deviceProp.sharedMemPerBlock - this->usedSharedMemory;
        leftoverRegisters = this->deviceProp.regsPerBlock - this->usedRegisters;
    }
};

CudaManager::CudaManager()
{

    cudaStatus = cudaGetDeviceCount(&deviceCount);

    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "cudaGetDeviceProperties failed!. Error:" << cudaStatus << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Found " << deviceCount << " CUDA-capable GPU(s)" << std::endl;

    cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);

    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "cudaGetDeviceProperties failed!. Error:" << cudaStatus << std::endl;
        exit(EXIT_FAILURE);
    }

    // set device properties
}

CudaManager::~CudaManager()
{
    std::cout << "CudaManager destructor called" << std::endl;
}

void CudaManager::printDeviceProperties()
{

    std::cout << "Device Name: " << deviceProp.name << std::endl;
    std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Total global memory: " << deviceProp.totalGlobalMem << std::endl;
    std::cout << "Total shared memory per block: " << deviceProp.sharedMemPerBlock << std::endl;
    std::cout << "Total registers per block: " << deviceProp.regsPerBlock << std::endl;
    std::cout << "Warp size: " << deviceProp.warpSize << std::endl;
    std::cout << "Maximum memory pitch: " << deviceProp.memPitch << std::endl;
    std::cout << "Maximum threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "Maximum dimension of block: " << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << std::endl;
    std::cout << "Maximum dimension of grid: " << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << std::endl;
    std::cout << "Clock rate: " << deviceProp.clockRate << std::endl;
    std::cout << "Total constant memory: " << deviceProp.totalConstMem << std::endl;
    std::cout << "Texture alignment: " << deviceProp.textureAlignment << std::endl;
    std::cout << "Concurrent copy and execution: " << (deviceProp.deviceOverlap ? "Yes" : "No") << std::endl;
    std::cout << "Number of multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "Threads per multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max num of Blocks per multiprocessor: " << deviceProp.maxBlocksPerMultiProcessor << std::endl;
}

void CudaManager::updateDeviceProperties()
{
    cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
    try
    {
        if (cudaStatus != cudaSuccess)
        {
            throw cudaStatus;
        }
    }
    catch (cudaError_t cudaStatus)
    {
        std::cerr << "cudaGetDeviceProperties failed!. Error:" << cudaStatus << std::endl;
        exit(EXIT_FAILURE);
    }
}