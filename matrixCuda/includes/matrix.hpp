#include <iostream>
#include <vector>
#include "cudaMemoryManager.hpp"
#include "../lib/includes/matrix.cuh"
#include <cuda_runtime.h>

template <typename T>
class Matrix
{
private:
    T *mat = nullptr;
    size_t size[2] = {};
    static unsigned int objCount;

    void allocateMemory()
    {
        mat = new T[size[0] * size[1]];
    }

public:
    static CudaMemoryManager<T> *cudaMemoryManager;
    Matrix()
    {
        this->size[0] = 0;
        this->size[1] = 0;
        // this->cudaMemoryManager = new CudaMemoryManager<T>();
        objCount++;
    };
    Matrix(const size_t m, const size_t &n)
    {
        this->size[0] = m;
        this->size[1] = n;
        this->fillWithZeros();
        // this->cudaMemoryManager = new CudaMemoryManager<T>();
        objCount++;
    };
    Matrix(T *mat, const size_t *size);
    Matrix(const std::vector<std::vector<T>> &mat);
    Matrix(const Matrix<T> &obj);
    Matrix<T> &operator=(const Matrix<T> &obj);
    Matrix<T> &operator=(const std::vector<std::vector<T>> &mat);
    Matrix<T> &operator=(T *mat);
    Matrix<T> &operator+=(T *mat);
    Matrix<T> &operator+=(const Matrix<T> &other);
    Matrix<T> operator+(const Matrix<T> &other);
    Matrix<T> operator+(const T *mat);
    Matrix<T> &operator*=(const T &scalar);
    Matrix<T> &operator*=(const Matrix<T> &obj);
    Matrix<T> &operator*=(const T *mat);
    Matrix<T> &operator*(const Matrix<T> &obj);
    Matrix<T> &operator*(const T *mat);
    Matrix<T> &inverse(const Matrix<T> &obj);
    Matrix<T> &inverse(const T &mat);
    Matrix<T> &inverse();
    T *data() const
    {
        return mat;
    }

    void print();
    T operator()(const size_t &row, const size_t &col) const;
    inline unsigned int get_size_row() const
    {
        return size[0];
    }
    inline unsigned int get_size_col() const
    {
        return size[1];
    }
    void fillWithZeros()
    {
        if (mat == nullptr)
        {
            // Allocate memory
            this->allocateMemory();
        }
        size_t m = size[0];
        size_t n = size[1];

        for (int i = 0; i < m; ++i)
        {
            for (int k = 0; k < n; ++k)
            {
                mat[i * n + k] = (T)0;
            }
        }
    }
    void identity()
    {
        if (mat == nullptr)
        {
            // Allocate memory
            this->allocateMemory();
        }
        size_t m = size[0];
        size_t n = size[1];

        for (int i = 0; i < m; ++i)
        {
            for (int k = 0; k < n; ++k)
            {
                if (i == k)
                {
                    mat[i * n + k] = (T)1;
                }
                else
                {
                    mat[i * n + k] = (T)0;
                }
            }
        }
    }
    ~Matrix();
};

template <typename T>
unsigned int Matrix<T>::objCount = 0;

template <typename T>
CudaMemoryManager<T> *Matrix<T>::cudaMemoryManager = new CudaMemoryManager<T>();

template <typename T>
Matrix<T>::Matrix(T *mat, const size_t *size)
{

    size_t m = size[0];
    size_t n = size[1];

    this->size[0] = m;
    this->size[1] = n;

    this->allocateMemory();

    for (int i = 0; i < m; ++i)
    {
        for (int k = 0; k < n; ++k)
        {
            this->mat[i * n + k] = mat[i * n + k];
        }
    }

    // this->cudaMemoryManager = new CudaMemoryManager<T>();
    objCount++;
}

template <typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>> &mat)
{
    size_t m = mat.size();
    size_t n = mat[0].size();

    this->size[0] = m;
    this->size[1] = n;

    this->allocateMemory();

    for (int i = 0; i < m; ++i)
    {
        for (int k = 0; k < n; ++k)
        {
            this->mat[i * n + k] = mat[i][k];
        }
    }

    // this->cudaMemoryManager = new CudaMemoryManager<T>();
    objCount++;
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T> &obj)
{
    size_t m = obj.get_size_row();
    size_t n = obj.get_size_col();

    this->size[0] = m;
    this->size[1] = n;

    this->allocateMemory();

    for (int i = 0; i < m; ++i)
    {
        for (int k = 0; k < n; ++k)
        {
            this->mat[i * n + k] = obj(i, k);
        }
    }

    // this->cudaMemoryManager = new CudaMemoryManager<T>();
    objCount++;
}

template <typename T>
Matrix<T>::~Matrix()
{
    --objCount;
    if (objCount == 0)
    {
        delete this->cudaMemoryManager;
    }

    if (mat == nullptr)
    {
        std::cout << "Matrix is empty" << std::endl;
        return;
    }

    delete[] mat;

    std::cout << "Matrix Memory Dellocated" << std::endl;
}

template <typename T>
T Matrix<T>::operator()(const size_t &row, const size_t &col) const
{
    return mat[row * size[1] + col];
}

template <typename T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &obj)
{
    size_t m = obj.get_size_row();
    size_t n = obj.get_size_col();

    if (this == &obj)
    {
        return *this;
    }

    if (mat != nullptr)
    {
        // Check if the size of the matrices are the same In order to not dellocate and allocate memory
        if (obj.get_size_row() != this->size[0] && obj.get_size_col() != this->size[1])
        {
            delete[] this->mat;

            this->size[0] = m;
            this->size[1] = n;

            this->allocateMemory();
        }
    }
    else // if (mat == nullptr) allocate memory
    {
        this->size[0] = m;
        this->size[1] = n;

        this->allocateMemory();
    }

    for (int i = 0; i < m; ++i)
    {
        for (int k = 0; k < n; ++k)
        {
            this->mat[i * n + k] = obj(i, k);
        }
    }
    return *this;
}

template <typename T>
Matrix<T> &Matrix<T>::operator=(const std::vector<std::vector<T>> &mat)
{
    size_t m = mat.size();
    size_t n = mat[0].size();

    if (this->mat != nullptr)
    {
        if (this->size[0] != mat.size() && this->size[1] != mat[0].size())
        {
            delete[] this->mat;

            this->size[0] = m;
            this->size[1] = n;

            this->allocateMemory();
        }
    }
    else
    {

        this->size[0] = m;
        this->size[1] = n;

        this->allocateMemory();
    }

    for (int i = 0; i < m; ++i)
    {
        for (int k = 0; k < n; ++k)
        {
            this->mat[i * n + k] = mat[i][k];
        }
    }

    return *this;
}
template <typename T>
Matrix<T> &Matrix<T>::operator=(T *mat)
{
    size_t m = this->size[0];
    size_t n = this->size[1];

    for (int i = 0; i < m; ++i)
    {
        for (int k = 0; k < n; ++k)
        {
            this->mat[i * n + k] = mat[i * n + k];
        }
    }
    return *this;
}

template <typename T>
Matrix<T> &Matrix<T>::operator+=(T *mat)
{

    size_t m = this->size[0];
    size_t n = this->size[1];

    T *d_mat = nullptr;
    T *d_mat2 = nullptr;
    this->cudaMemoryManager->allocate(d_mat, m * n);
    this->cudaMemoryManager->allocate(d_mat2, m * n);

    this->cudaMemoryManager->copyToDevice(d_mat, this->mat, m * n);
    this->cudaMemoryManager->copyToDevice(d_mat2, mat, m * n);

    // Set Blocks, Threads and Grids
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Call Kernel for addition
    addMatrices2_Wrapper(d_mat, d_mat2, m * n, blocksPerGrid, threadsPerBlock);

    // Copy result back to host
    this->cudaMemoryManager->copyToHost(this->mat, d_mat, m * n);

    // Free memory
    this->cudaMemoryManager->deallocate(d_mat);
    this->cudaMemoryManager->deallocate(d_mat2);

    return *this;
}

template <typename T>
void Matrix<T>::print()
{
    size_t m = this->size[0];
    size_t n = this->size[1];
    for (int i = 0; i < m; ++i)
    {
        for (int k = 0; k < n; ++k)
        {
            std::cout << this->mat[i * n + k] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &other)
{
    // Make sure sizes are the same
    if (this->size[0] != other.size[0] && this->size[1] != other.size[1])
    {
        std::cout << "Matrix sizes are not the same size" << std::endl;
        return *this;
    }

    size_t m = this->size[0];
    size_t n = this->size[1];

    Matrix<T> result(m, n);

    T *d_mat = nullptr;
    T *d_mat2 = nullptr;

    this->cudaMemoryManager->allocate(d_mat, m * n);
    this->cudaMemoryManager->allocate(d_mat2, m * n);
    this->cudaMemoryManager->copyToDevice(d_mat, this->mat, m * n);
    this->cudaMemoryManager->copyToDevice(d_mat2, other.mat, m * n);

    // Set Blocks, Threads and Grids
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Call Kernel for addition
    addMatrices2_Wrapper(d_mat, d_mat2, m * n, blocksPerGrid, threadsPerBlock);

    // Copy result back to host
    this->cudaMemoryManager->copyToHost(result.mat, d_mat, m * n);
    // Free memory
    this->cudaMemoryManager->deallocate(d_mat);
    this->cudaMemoryManager->deallocate(d_mat2);

    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const T *mat)
{
    size_t m = this->size[0];
    size_t n = this->size[1];

    Matrix<T> result(m, n);

    T *d_mat = nullptr;
    T *d_mat2 = nullptr;

    this->cudaMemoryManager->allocate(d_mat, m * n);
    this->cudaMemoryManager->allocate(d_mat2, m * n);
    this->cudaMemoryManager->copyToDevice(d_mat, this->mat, m * n);
    this->cudaMemoryManager->copyToDevice(d_mat2, mat, m * n);

    // Set Blocks, Threads and Grids
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Call Kernel for addition
    addMatrices2_Wrapper(d_mat, d_mat2, m * n, blocksPerGrid, threadsPerBlock);

    // Copy result back to host
    this->cudaMemoryManager->copyToHost(result.mat, d_mat, m * n);
    // Free memory
    this->cudaMemoryManager->deallocate(d_mat);
    this->cudaMemoryManager->deallocate(d_mat2);

    return result;
}

template <typename T>
Matrix<T> &Matrix<T>::operator+=(const Matrix<T> &other)
{
    // make sure sizes are the same
    if (this->size[0] != other.size[0] && this->size[1] != other.size[1])
    {
        std::cout << "Matrix sizes are not the same size" << std::endl;
        return *this;
    }

    size_t m = this->size[0];
    size_t n = this->size[1];

    T *d_mat = nullptr;
    T *d_mat2 = nullptr;

    this->cudaMemoryManager->allocate(d_mat, m * n);
    this->cudaMemoryManager->allocate(d_mat2, m * n);

    this->cudaMemoryManager->copyToDevice(d_mat, this->mat, m * n);
    this->cudaMemoryManager->copyToDevice(d_mat2, other.data(), m * n);

    // Set Blocks, Threads and Grids
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Call Kernel for addition
    addMatrices2_Wrapper(d_mat, d_mat2, m * n, blocksPerGrid, threadsPerBlock);

    // Copy result back to host
    this->cudaMemoryManager->copyToHost(this->mat, d_mat, m * n);
    // Free memory
    this->cudaMemoryManager->deallocate(d_mat);
    this->cudaMemoryManager->deallocate(d_mat2);

    return *this;
}

template <typename T>
Matrix<T> &Matrix<T>::operator*=(const T &scalar)
{
    size_t m = this->size[0];
    size_t n = this->size[1];

    T *d_mat = nullptr;

    this->cudaMemoryManager->allocate(d_mat, m * n);
    this->cudaMemoryManager->copyToDevice(d_mat, this->mat, m * n);

    // Set Blocks, Threads and Grids
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Call Kernel for addition
    multiply_matrix_by_scalar_wrapper(d_mat, scalar, m * n, blocksPerGrid, threadsPerBlock);

    // Copy result back to host
    this->cudaMemoryManager->copyToHost(this->mat, d_mat, m * n);
    // Free memory
    this->cudaMemoryManager->deallocate(d_mat);

    return *this;
}

template <typename T>
Matrix<T> &Matrix<T>::operator*=(const Matrix<T> &obj)
{
    // Make sure rows of this are the same as columns of obj
    if (this->size[1] != obj.size[0])
    {
        std::cout << "Matrix sizes are not the same size" << std::endl;
        return *this;
    }

    size_t m_this = this->size[0];
    size_t n_this = this->size[1];
    size_t m_other = obj.size[0];
    size_t n_other = obj.size[1];

    T *d_mat = nullptr;
    T *d_mat2 = nullptr;
    T *d_result = nullptr;

    this->cudaMemoryManager->allocate(d_mat, m_this * n_this);
    this->cudaMemoryManager->allocate(d_mat2, m_other * n_other);
    this->cudaMemoryManager->allocate(d_result, m * n);

    this->cudaMemoryManager->copyToDevice(d_mat, this->mat, m * n);
    this->cudaMemoryManager->copyToDevice(d_mat2, obj.data(), m * n);

    // Set Blocks, Threads and Grids
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
}

template <typename T>
Matrix<T> &Matrix<T>::operator*=(const T *mat)
{
}