#include <iostream>
#include <cuda_runtime.h>
#include "./cudaMemoryManager.hpp"
#include "./cudaManager.hpp"
#include "../lib/includes/matrix.cuh"

template <typename T>
class CudaKernelManager : public CudaManager<T>
{
    public:
        CudaKernelManager() : CudaManager<T>() {};
        ~CudaKernelManager();
};
