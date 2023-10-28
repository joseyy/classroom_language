
cmake_minimum_required(VERSION 3.20)


project (interpreter VERSION 1.0.1)


set(CMAKE_CXX_STANDARD 17)

# set cpp flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -Wpedantic -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-function -Wno-unused-private-field -Wno-unused-lambda-capture -Wno-unused-local-typedefs -Wno-unused-value -Wno-un")


# set include directories
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/includes)

set(SOURCE_FILES 
${SOURCE_DIR}/main.cpp
${SOURCE_DIR}/lexer.cpp
${SOURCE_DIR}/parser.cpp
${SOURCE_DIR}/ast.cpp
${SOURCE_DIR}/ast_pointer.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIR})


