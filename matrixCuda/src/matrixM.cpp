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

    float *mat = new float[m * n];

    // fill mat with 0 values
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            mat[i * n + j] = 1.0;
        }
    }
    size_t size[2] = {m, n};
    Matrix<float> matObj10(mat, size);
    Matrix<float> matObj11(mat, size);
    matObj10 = matObj11 + mat + matObj11 + matObj10;

    matObj10 *= 4.0 * 51.4;

    // Print matObj12
    matObj10.print();

    delete[] mat; // free memory
    return 0;
}
