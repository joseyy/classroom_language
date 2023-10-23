#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

__global__ void addMatrices(float *a, float *b, size_t n);

__global__ void addMatrices2(float *a, float *b, size_t n);

void addMatrices_Wrapper(float *a, float *b, size_t n, dim3 numBlocks, dim3 blockSize);

void addMatrices2_Wrapper(float *a, float *b, size_t n, dim3 blocksPerGrid, dim3 threadsPerBlock);

__global__ void sigmoid(float *a, size_t n);

__global__ void sigmoid_derivative(float *a, size_t n);

void sigmoid_wrapper(float *a, size_t n, dim3 numBlocks, dim3 blockSize);

void sigmoid_derivative_wrapper(float *a, size_t n, dim3 numBlocks, dim3 blockSize);

__global__ void categorical_cross_entropy(float *a, float *b, float *result, size_t n);

__global__ void categorical_cross_entropy_derivative(float *a, float *b, float *result, size_t n);

void categorical_cross_entropy_wrapper(float *a, float *b, float *result, size_t n, dim3 numBlocks, dim3 blockSize);

void categorical_cross_entropy_derivative_wrapper(float *a, float *b, float *result, size_t n, dim3 numBlocks, dim3 blockSize);

__global__ void softmax(float *mat, float *softmax, size_t m, size_t n);

__global__ void softmax_derivative(float *softmax, float *derivative, size_t m, size_t n);

void softmax_wrapper(float *mat, float *softmax, size_t m, size_t n, dim3 numBlocks, dim3 blockSize);

void softmax_derivative_wrapper(float *softmax, float *derivative, size_t m, size_t n, dim3 numBlocks, dim3 blockSize);

__global__ void multiply_matrix_by_scalar(float *d_mat, float scalar, size_t n);

void multiply_matrix_by_scalar_wrapper(float *d_mat, float scalar, size_t n, dim3 blocksPerGrid, dim3 threadsPerBlock);
