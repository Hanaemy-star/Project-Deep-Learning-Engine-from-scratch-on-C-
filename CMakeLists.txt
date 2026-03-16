cmake_minimum_required(VERSION 3.31)
project(ml-proj)

set(CMAKE_CXX_STANDARD 20)

add_library(math_lib
        include/vector.hpp
        src/vector.cpp
        include/matrix.hpp
        src/matrix.cpp
        include/tensor.hpp
        src/tensor.cpp
        include/Optimizer.hpp
        src/optimizer.cpp
        include/linear.hpp
        src/linear.cpp
        include/mlp.hpp
        src/mlp.cpp
)

target_include_directories(math_lib PUBLIC include)

add_executable(ml-proj main.cpp
        include/matrix.hpp)

target_link_libraries(ml-proj PRIVATE math_lib)

add_executable(test_runner tests/test_main.cpp)

target_link_libraries(test_runner PRIVATE math_lib)

target_include_directories(test_runner PRIVATE include)