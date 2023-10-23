#pragma once
#include "cudaManager.cuh"
#include <vector>
#include <algorithm>
#include <list>

template <typename T>
class CudaMemoryManager : public CudaManager
{
private:
    std::vector<T *> devicePtr;
    int freeMemory, totalMemory;

public:
    CudaMemoryManager() : CudaManager(false) {}
    ~CudaMemoryManager();
    void allocate(T *&ptr, size_t size);
    void deallocate(T *&ptr);
    void deallocateAll();
    void copyToDevice(T *&devicePtr, const T *data, size_t size);
    void copyToHost(T *&hostPtr, const T *devicePtr, size_t size);
    void checkMemoryUsage();
};

template <typename T>
void CudaMemoryManager<T>::allocate(T *&ptr, size_t size)
{
    checkMemoryUsage();
    if ((size * sizeof(T)) > this->freeMemory)
    {
        std::cerr << "Not enough memory on GPU" << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaError_t cudaMemoryAllocationError = cudaMalloc((void **)&ptr, size * sizeof(T));

    if (cudaMemoryAllocationError != cudaSuccess)
    {
        std::cerr << "Error allocating memory on GPU: " << cudaGetErrorString(cudaMemoryAllocationError) << std::endl;
        deallocateAll();
        exit(EXIT_FAILURE);
    }

    this->devicePtr.push_back(ptr);
}

template <typename T>
void CudaMemoryManager<T>::deallocate(T *&ptr)
{
    cudaError_t cudaMemoryDeallocationError = cudaFree(ptr);

    if (cudaMemoryDeallocationError != cudaSuccess)
    {
        std::cerr << "Error deallocating memory on GPU: " << cudaGetErrorString(cudaMemoryDeallocationError) << std::endl;
        deallocateAll();
        exit(EXIT_FAILURE);
    }

    auto it = std::find(this->devicePtr.begin(), this->devicePtr.end(), ptr);
    *it = nullptr;
}

template <typename T>
CudaMemoryManager<T>::~CudaMemoryManager()
{
    deallocateAll();
    std::cout << "Deallocated all memory on GPU" << std::endl;
}

template <typename T>
void CudaMemoryManager<T>::deallocateAll()
{
    for (auto it = this->devicePtr.begin(); it != this->devicePtr.end(); ++it)
    {
        if (*it != nullptr)
            this->deallocate(*it);
    }
}

template <typename T>
void CudaMemoryManager<T>::copyToDevice(T *&devicePtr, const T *data, size_t size)
{
    // Make sure data is already allocated on GPU
    if (devicePtr == nullptr)
    {
        allocate(devicePtr, size);
    }

    cudaError_t cudaMemoryCopyError = cudaMemcpy(devicePtr, data, size * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaMemoryCopyError != cudaSuccess)
    {
        std::cerr << "Error copying data to GPU: " << cudaGetErrorString(cudaMemoryCopyError) << std::endl;
        deallocateAll();
        exit(EXIT_FAILURE);
    }
}

template <typename T>
void CudaMemoryManager<T>::checkMemoryUsage()
{
    size_t freeMemory, totalMemory;
    cudaError_t cudaMemoryUsageError = cudaMemGetInfo(&freeMemory, &totalMemory);

    if (cudaMemoryUsageError != cudaSuccess)
    {
        std::cerr << "Error getting memory usage on GPU: " << cudaGetErrorString(cudaMemoryUsageError) << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Free memory: " << freeMemory << " Total memory: " << totalMemory << std::endl;

    this->freeMemory = freeMemory;
    this->totalMemory = totalMemory;
}

template <typename T>
void CudaMemoryManager<T>::copyToHost(T *&hostPtr, const T *devicePtr, size_t size)
{
    // Make sure data is already allocated on CPU
    if (hostPtr == nullptr)
    {
        hostPtr = new T[size];
    }


    cudaError_t cudaMemoryCopyError = cudaMemcpy(hostPtr, devicePtr, size * sizeof(T), cudaMemcpyDeviceToHost);
    if (cudaMemoryCopyError != cudaSuccess)
    {
        std::cerr << "Error copying data to CPU: " << cudaGetErrorString(cudaMemoryCopyError) << std::endl;
        deallocateAll();
        exit(EXIT_FAILURE);
    }
}