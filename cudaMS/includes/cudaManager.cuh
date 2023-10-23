#pragma once

#include <iostream>
#include <cuda_runtime.h>

class CudaManager
{
private:
    int deviceCount;
    cudaDeviceProp deviceProp;
    cudaError_t cudaStatus;

public:
    CudaManager(bool getDeviceProperties = true);
    ~CudaManager();
    void printDeviceProperties();
    void updateDeviceProperties();
};

CudaManager::CudaManager(bool gerDeviceProperties)
{
    if (gerDeviceProperties)
    {

        cudaStatus = cudaGetDeviceCount(&deviceCount);
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

        std::cout << "Found " << deviceCount << " CUDA-capable GPU(s)" << std::endl;

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