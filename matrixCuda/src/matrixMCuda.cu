#include <iostream>
#include <chrono>

__global__ void matrixMultiply(float *a, float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
        {
            sum += a[i * n + k] * b[k * n + j];
        }
        c[i * n + j] = sum;
    }
}

int main(int argc, char **argv)
{
    try
    {
        if (argc != 3)
        {
            throw "Usage: ./matrixMul <N>";
        }
    }
    catch (const char *msg)
    {
        std::cerr << msg << std::endl;
        return 1;
    }

    int N = atoi(argv[1]);
    int maxNumThreads = N * N;
    int numThreadsPrBlock = atoi(argv[2]);

    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    int size = N * N * sizeof(float);

    // Allocate memory on the host
    a = new float[N * N];
    b = new float[N * N];
    c = new float[N * N];

    // Initialize matrices
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            a[i * N + j] = i + j;
            b[i * N + j] = i - j;
        }
    }

    // Allocate memory on the device
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy matrices from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 threadsPerBlock(numThreadsPrBlock, numThreadsPrBlock);
    dim3 numBlocks(maxNumThreads / threadsPerBlock.x, maxNumThreads / threadsPerBlock.y);

    // Launch kernel and measure time for multiple loops to get average
    int numLoops = 10;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numLoops; ++i)
    {
        matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    // Copy result from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Print result
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            std::cout << c[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Free memory on host
    delete[] a;
    delete[] b;
    delete[] c;

    // Print time
    std::cout << "Time taken to multiply two " << N << "x" << N << " matrices: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / numLoops << " nanoseconds" << std::endl;

    return 0;
}