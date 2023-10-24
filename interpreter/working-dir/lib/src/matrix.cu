#include "../includes/matrix.cuh"

__global__ void addMatrices(float *a, float *b, size_t n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
    {
        a[index] += b[index];
    }
}

__global__ void addMatrices2(float *a, float *b, size_t n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        a[i] += b[i];
    }
}

void addMatrices_Wrapper(float *a, float *b, size_t n, dim3 numBlocks, dim3 blockSize)
{
    addMatrices<<<numBlocks, blockSize>>>(a, b, n);
    cudaDeviceSynchronize();
}

void addMatrices2_Wrapper(float *a, float *b, size_t n, dim3 blocksPerGrid, dim3 threadsPerBlock)
{
    addMatrices2<<<blocksPerGrid, threadsPerBlock>>>(a, b, n);
    cudaDeviceSynchronize();
}

__global__ void sigmoid(float *a, size_t n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        a[i] = 1 / (1 + exp(-a[i]));
    }
}

__global__ void sigmoid_derivative(float *a, size_t n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    while (index < n)
    {
        if (a[index] > 0 && a[index] < 1)
        {
            a[index] = a[index] * (1 - a[index]);
        }
        index += blockDim.x * gridDim.x;
    }
}

void sigmoid_wrapper(float *a, size_t n, dim3 numBlocks, dim3 blockSize)
{
    sigmoid<<<numBlocks, blockSize>>>(a, n);
    cudaDeviceSynchronize();
}

void sigmoid_derivative_wrapper(float *a, size_t n, dim3 numBlocks, dim3 blockSize)
{
    sigmoid_derivative<<<numBlocks, blockSize>>>(a, n);
    cudaDeviceSynchronize();
}

__global__ void categorical_cross_entropy(float *a, float *b, float *result, size_t n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    *result = 0;
    for (int i = index; i < n; i += stride)
    {
        *result += b[i] * log(a[i]);
    }
    *result = -(*result);
}

__global__ void categorical_cross_entropy_derivative(float *a, float *b, float *result, size_t n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    while (index < n)
    {
        if (a[index] > 0 && b[index] > 0)
        {
            result[index] = a[index] / b[index];
        }
        else
        {
            result[index] = 0;
        }
        index += blockDim.x * gridDim.x;
    }
}

void categorical_cross_entropy_wrapper(float *a, float *b, float *result, size_t n, dim3 numBlocks, dim3 blockSize)
{
    categorical_cross_entropy<<<numBlocks, blockSize>>>(a, b, result, n);
    cudaDeviceSynchronize();
}

void categorical_cross_entropy_derivative_wrapper(float *a, float *b, float *result, size_t n, dim3 numBlocks, dim3 blockSize)
{
    categorical_cross_entropy_derivative<<<numBlocks, blockSize>>>(a, b, result, n);
    cudaDeviceSynchronize();
}

__global__ void softmax_kernel(float *mat, float *softmax, size_t m, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m)
    {
        float row_max = -INFINITY;
        for (size_t j = 0; j < n; ++j)
        {
            row_max = fmaxf(row_max, mat[i * n + j]);
        }
        float row_sum = 0;
        for (size_t j = 0; j < n; ++j)
        {
            softmax[i * n + j] = expf(mat[i * n + j] - row_max);
            row_sum += softmax[i * n + j];
        }
        for (size_t j = 0; j < n; ++j)
        {
            if (row_sum > 0)
            {
                softmax[i * n + j] /= row_sum;
            }
            else
            {
                softmax[i * n + j] = 0;
            }
        }
    }
}

__global__ void softmax_derivative(float *softmax, float *derivative, size_t m, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m)
    {
        for (size_t j = 0; j < n; ++j)
        {
            if (j == i)
            {
                derivative[i * n + j] = softmax[i * n + j] * (1 - softmax[i * n + j]);
            }
            else
            {
                derivative[i * n + j] = -softmax[i * n + i] * softmax[i * n + j];
            }
        }
    }
}

void softmax_wrapper(float *mat, float *softmax, size_t m, size_t n, dim3 numBlocks, dim3 blockSize)
{
    softmax_kernel<<<numBlocks, blockSize>>>(mat, softmax, m, n);
    cudaDeviceSynchronize();
}

void softmax_derivative_wrapper(float *softmax, float *derivative, size_t m, size_t n, dim3 numBlocks, dim3 blockSize)
{
    softmax_derivative<<<numBlocks, blockSize>>>(softmax, derivative, m, n);
    cudaDeviceSynchronize();
}

__global__ void multiply_matrix_by_scalar(float *d_mat, float scalar, size_t n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (index < n)
    {
        d_mat[index] *= scalar;
        index += stride;
    }
}

void multiply_matrix_by_scalar_wrapper(float *d_mat, float scalar, size_t n, dim3 blocksPerGrid, dim3 threadsPerBlock)
{
    multiply_matrix_by_scalar<<<blocksPerGrid, threadsPerBlock>>>(d_mat, scalar, n);
    cudaDeviceSynchronize();
}

__global__ void multiply_matrices(float *a,
                                  float *b,
                                  float *result,
                                  size_t m,
                                  size_t n,
                                  size_t p)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < p)
    {
        float sum = 0;
        for (int i = 0; i < n; ++i)
        {
            sum += a[row * n + i] * b[i * p + col];
        }
        result[row * p + col] = sum;
    }
}

void multiply_matrices_wrapper(float *a,
                               float *b,
                               float *result,
                               size_t m,
                               size_t n,
                               size_t p,
                               dim3 blocksPerGrid,
                               dim3 threadsPerBlock)
{
    multiply_matrices<<<blocksPerGrid, threadsPerBlock>>>(a, b, result, m, n, p);
    cudaDeviceSynchronize();
}

__global__ void transpose_matrix(float *d_mat, float *d_result, size_t m, size_t n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row_stride = gridDim.y * blockDim.y;
    int col_stride = gridDim.x * blockDim.x;
    for (int i = row; i < m; i += row_stride) {
        for (int j = col; j < n; j += col_stride) {
            d_result[j * m + i] = d_mat[i * n + j];
        }
    }
}

void transpose_matrix_wrapper(float *d_mat, float *d_result,size_t m, size_t n, dim3 blocksPerGrid, dim3 threadsPerBlock){
    transpose_matrix<<<blocksPerGrid, threadsPerBlock>>>(d_mat, d_result, m, n);
    cudaDeviceSynchronize();
}