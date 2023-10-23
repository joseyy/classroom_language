#include <iostream>
#include "../includes/cudaManager.cuh"
#include "../includes/cudaMemoryManager.cuh"

int main(int argc, char **argv)
{

    CudaMemoryManager<float> cudaMemoryManager;

    float *hostArr = new float[10];
    for (int i = 0; i < 10; i++)
    {
        hostArr[i] = i % 10 / 10.0f;
    }

    float *deviceArr;
    float *hostArr2 = new float[10];
    cudaMemoryManager.allocate(deviceArr, 10);
    cudaMemoryManager.copyToDevice(deviceArr, hostArr, 10);
    cudaMemoryManager.copyToHost(hostArr2, deviceArr, 10);

    for (int i = 0; i < 10; i++)
    {
        std::cout << hostArr2[i] << std::endl;
    }

    return 0;
}
