cmake_minimum_required(VERSION 3.20)


project(interpreter VERSION 1.0)

# set cxx version
set(CMAKE_CXX_STANDARD 17)

# Add flags for C++ and set version to 17
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra")

include_directories(includes)

# Add the executable
set(SOURCE_FILES
src/main.cpp 
# src/interpreter.cpp 
# src/lexer.cpp 
# src/parser.cpp 
# src/token.cpp 
# src/ast.cpp 
# src/ast_printer.cpp 
# src/interpreter_error.cpp 
# src/interpreter_error_handler.cpp
)


add_executable(interpreter ${SOURCE_FILES})  # add executable






