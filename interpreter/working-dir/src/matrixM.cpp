#include <iostream>
#include <boost/random.hpp>
#include <chrono>
#include <vector>
#include "../includes/matrix.hpp"
#include "../includes/cudaMemoryManager.hpp"
#include "../includes/neuralNFunctionsClass.hpp"

int main(int argv, char **argc)
{

    try
    {
        if (argv != 3)
        {
            throw "Error: Invalid number of arguments";
        }
    }
    catch (const char *msg)
    {
        std::cerr << msg << std::endl;
        return 1;
    }

    // create a random matrix of size m x n where the values are double precision floating point numbers between 0 and 10 (uniform distribution)

    // create a matrix of size m x n
    size_t m = std::stoi(argc[1]);
    size_t n = std::stoi(argc[2]);

    float *A = new float[2 * 3];
    float *B = new float[3 * 2];

    // fill A
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
            A[i * 3 + j] = i * 3 + j + 1;
    }

    // fill B
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 2; j++)
            B[i * 2 + j] = i * 2 + j + 7;
    }

    size_t size_A[2] = {2, 3};
    size_t size_B[2] = {3, 2};
    Matrix<float> m1(A, size_A);
    Matrix<float> m2(B, size_B);
    Matrix<float> m3 = m1 * m2;

    m1 *= m2;
    m3.transpose();

    m1.print();
    m3.print();
    return 0;
}
