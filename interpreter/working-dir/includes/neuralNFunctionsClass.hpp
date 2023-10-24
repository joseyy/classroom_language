#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include "../lib/includes/matrix.cuh"
#include "./cudaMemoryManager.hpp"

template <typename T>
class NeuralNFunctions
{
private:
    CudaMemoryManager<T> &cmm;

public:
    NeuralNFunctions(CudaMemoryManager<T> &cmm) : cmm(cmm){};
    void sigmoid(T *mat, size_t m, size_t n);
    void derivativeOfSigmoid(T *mat, size_t m, size_t n);
    void categoricalCrossEntropy(T *y, T *x, T &result, size_t m, size_t n);
    void derivativeOfCategoricalCrossEntropy(T *y, T *x, T *&result, size_t m, size_t n);
    void softmax(T *mat, size_t m, size_t n);
    void derivativeOfSoftmax(T *mat, size_t m, size_t n);
};

template <typename T>
void NeuralNFunctions<T>::sigmoid(T *mat, size_t m, size_t n)
{
    // for (int i = 0; i < m * n; ++i)
    // {
    //     mat[i] = 1 / (1 + exp(-mat[i]));
    // }

    // Implementing sigmoid using CUDA kernels
    T *d_mat;
    cmm.allocate(d_mat, m * n);
    cmm.copyToDevice(d_mat, mat, m * n);
    dim3 blocksize(32, 32);
    dim3 grid((m + blocksize.x - 1) / blocksize.x, (n + blocksize.y - 1) / blocksize.y);
    sigmoid_wrapper(d_mat, m * n, grid, blocksize);
    cmm.copyToHost(mat, d_mat, m * n);
    cmm.deallocate(d_mat);

    // Implementing sigmoid using thrust
    // thrust::device_ptr<T> d_mat = thrust::device_pointer_cast(mat);
    // thrust::transform(d_mat, d_mat + m * n, d_mat, sigmoid_functor<T>());
}

template <typename T>
void NeuralNFunctions<T>::derivativeOfSigmoid(T *mat, size_t m, size_t n)
{
    // for (int i = 0; i < m * n; ++i)
    // {
    //     mat[i] = mat[i] * (1 - mat[i]);
    // }

    // Implementing derivative of sigmoid using CUDA kernels
    T *d_mat;
    cmm.allocate(d_mat, m * n);
    cmm.copyToDevice(d_mat, mat, m * n);
    dim3 blocksize(32, 32);
    dim3 grid((m + blocksize.x - 1) / blocksize.x, (n + blocksize.y - 1) / blocksize.y);
    sigmoid_derivative_wrapper(d_mat, m * n, grid, blocksize);
    cmm.copyToHost(mat, d_mat, m * n);
    cmm.deallocate(d_mat);
}

template <typename T>
void NeuralNFunctions<T>::categoricalCrossEntropy(T *y, T *x, T &result, size_t m, size_t n)
{
    // result = 0;
    // for (int i = 0; i < m * n; ++i)
    // {
    //     result += y[i] * log(x[i]);
    // }
    // result *= -1;

    // Implementing categorical cross entropy using CUDA kernels
    T *d_y, *d_x, *d_result;
    cmm.allocate(d_y, m * n);
    cmm.allocate(d_x, m * n);
    cmm.allocate(d_result, 1);
    cmm.copyToDevice(d_y, y, m * n);
    cmm.copyToDevice(d_x, x, m * n);
    dim3 blocksize(32, 32);
    dim3 grid((m + blocksize.x - 1) / blocksize.x, (n + blocksize.y - 1) / blocksize.y);
    categorical_cross_entropy_wrapper(d_y, d_x, d_result, m * n, grid, blocksize);
    cmm.copyToHost(&result, d_result, 1);
    cmm.deallocate(d_y);
    cmm.deallocate(d_x);
    cmm.deallocate(d_result);
}

/**
 * @brief Calculates the derivative of categorical cross entropy
 *
 * @tparam T
 * @param y  : label matrix
 * @param x : prediction matrix
 * @param result : result matrix
 * @param m : number of rows
 * @param n : number of columns
 */
template <typename T>
void NeuralNFunctions<T>::derivativeOfCategoricalCrossEntropy(T *y, T *x, T *&result, size_t m, size_t n)
{
    // for (int i = 0; i < m * n; ++i)
    // {
    //     result[i] = y[i] / x[i];
    // }

    // Implementing derivative of categorical cross entropy using CUDA kernels
    T *d_y, *d_x, *d_result;
    cmm.allocate(d_y, m * n);
    cmm.allocate(d_x, m * n);
    cmm.allocate(d_result, m * n);
    cmm.copyToDevice(d_y, y, m * n);
    cmm.copyToDevice(d_x, x, m * n);
    dim3 blocksize(32, 32);
    dim3 grid((m + blocksize.x - 1) / blocksize.x, (n + blocksize.y - 1) / blocksize.y);
    categorical_cross_entropy_derivative_wrapper(d_y, d_x, d_result, m * n, grid, blocksize);
    cmm.copyToHost(result, d_result, m * n);
    cmm.deallocate(d_y);
    cmm.deallocate(d_x);
    cmm.deallocate(d_result);
}

template <typename T>
void NeuralNFunctions<T>::softmax(T *mat, size_t m, size_t n)
{
    // for (size_t i = 0; i < m; ++i) {
    //     T row_max = -std::numeric_limits<T>::infinity();
    //     for (size_t j = 0; j < n; ++j) {
    //         row_max = std::max(row_max, mat[i * n + j]);
    //     }
    //     T row_sum = 0;
    //     for (size_t j = 0; j < n; ++j) {
    //         mat[i * n + j] = std::exp(mat[i * n + j] - row_max);
    //         row_sum += mat[i * n + j];
    //     }
    //     for (size_t j = 0; j < n; ++j) {
    //         mat[i * n + j] /= row_sum;
    //     }
    // }

    // Implementing softmax using CUDA kernels
    T *d_mat;
    T *softmax_result;
    cmm.allocate(d_mat, m * n);
    cmm.allocate(softmax_result, m * n);
    cmm.copyToDevice(d_mat, mat, m * n);
    dim3 blocksize(32, 32);
    dim3 grid((m + blocksize.x - 1) / blocksize.x, (n + blocksize.y - 1) / blocksize.y);
    softmax_wrapper(d_mat, softmax_result, m, n, grid, blocksize);
    cmm.copyToHost(mat, softmax_result, m * n);
    cmm.deallocate(d_mat);
    cmm.deallocate(softmax_result);
}

template <typename T>
void NeuralNFunctions<T>::derivativeOfSoftmax(T *mat, size_t m, size_t n)
{
    // for (size_t i = 0; i < m; ++i)
    // {
    //     T row_max = -std::numeric_limits<T>::infinity();
    //     for (size_t j = 0; j < n; ++j)
    //     {
    //         row_max = std::max(row_max, mat[i * n + j]);
    //     }
    //     T row_sum = 0;
    //     for (size_t j = 0; j < n; ++j)
    //     {
    //         mat[i * n + j] = std::exp(mat[i * n + j] - row_max);
    //         row_sum += mat[i * n + j];
    //     }
    //     for (size_t j = 0; j < n; ++j)
    //     {
    //         mat[i * n + j] /= row_sum;
    //         if (j == i)
    //         {
    //             mat[i * n + j] *= (1 - mat[i * n + j]);
    //         }
    //         else
    //         {
    //             mat[i * n + j] *= (-mat[i * n + j]);
    //         }
    //     }
    // }

    // Implementing derivative of softmax using CUDA kernels
    T *d_mat;
    T *softmax_derivative_result;
    cmm.allocate(d_mat, m * n);
    cmm.allocate(softmax_derivative_result, m * n);
    cmm.copyToDevice(d_mat, mat, m * n);
    dim3 blocksize(32, 32);
    dim3 grid((m + blocksize.x - 1) / blocksize.x, (n + blocksize.y - 1) / blocksize.y);
    softmax_derivative_wrapper(d_mat, softmax_derivative_result, m, n, grid, blocksize);
    cmm.copyToHost(mat, softmax_derivative_result, m * n);
    cmm.deallocate(d_mat);
    cmm.deallocate(softmax_derivative_result);
}