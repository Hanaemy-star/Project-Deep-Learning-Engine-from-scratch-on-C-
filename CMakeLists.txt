cmake_minimum_required(VERSION 3.31)
project(math_biblio)

set(CMAKE_CXX_STANDARD 20)

add_library(math_lib
        include/vector.hpp
        src/vector.cpp
        include/matrix.hpp
        src/matrix.cpp
        include/tensor.hpp
        src/tensor.cpp
)

target_include_directories(math_lib PUBLIC include)

add_executable(ml-proj main.cpp
        include/matrix.hpp)

target_link_libraries(ml-proj PRIVATE math_lib)