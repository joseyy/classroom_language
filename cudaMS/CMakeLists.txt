cmake_minimum_required(VERSION 3.20)

enable_language(CUDA)

project(cuda-management-system)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wpedantic -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-function -Wno-unused-private-field -Wno-unused-lambda-capture -Wno-unused-local-typedefs -Wno-unused-value -Wno-unused-const-variable -Wno-unused-label -Wno-unused-result -Wno-unused-local-typedefs -Wno-unused-const-variable -Wno-unused-label -Wno-unused-result -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-function -Wno-unused-private-field -Wno-unused-lambda-capture -Wno-unused-value -Wno-unused-parameter -Wno-unused-local-typedefs -Wno-unused-const-variable -Wno-unused-label -Wno-unused-result -Wno-unused-local-typedefs -Wno-unused-const-variable -Wno-unused-label -Wno-unused-result -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-function -Wno-unused-private-field -Wno-unused-lambda-capture -Wno-unused-value -Wno-unused-parameter -Wno-unused-local-typedefs -Wno-unused-const-variable -Wno-unused-label -Wno-unused-result -Wno-unused-local-typedefs -Wno-unused-const-variable -Wno-unused-label -Wno-unused-result")

# Add flags for CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14 -Wall -Wextra")

set(SOURCES src/main.cu)

# set the include path for the project
include_directories(includes)

add_executable(${PROJECT_NAME} ${SOURCES})



