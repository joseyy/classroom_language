cmake_minimum_required(VERSION 3.15)

project(HelloWorld)


set(CMAKE_CXX_STANDARD 17)

add_executable(HelloWorld src/main.cpp)

target_include_directories(HelloWorld PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/includes)