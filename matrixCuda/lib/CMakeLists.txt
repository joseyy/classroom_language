cmake_minimum_required(VERSION 3.20)

project(my_cuda_library)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)

include_directories(includes)

cuda_add_library(${PROJECT_NAME} src/matrix.cu)