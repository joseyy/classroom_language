cmake_minimum_required(VERSION 3.20)

project(matrix_multiplication_1 LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_STANDARD 11)

include_directories(includes)

# add subdirectory for cuda files
add_subdirectory(lib)

# Add flags for C++ and set version to 14
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")

# Add flags for CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11 -Wall -Wextra")

# add source files for cpp files
set(SOURCE_FILES_CPP src/matrixM.cpp)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/includes)

add_executable(${PROJECT_NAME} ${SOURCE_FILES_CPP})

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

target_link_libraries(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/lib/build/libmy_cuda_library.a)


# add boost library headers
target_include_directories(${PROJECT_NAME} PUBLIC /home/snow/boost_1_83_0)

# add boost library files
target_link_directories(${PROJECT_NAME} PUBLIC /home/snow/boost_1_83_0/stage/lib)

