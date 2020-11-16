cmake_minimum_required(VERSION 3.5)
project(lgtm)

set(CMAKE_CXX_STANDARD 14)
add_executable(lgtm main.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -march=native -O3")

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)
