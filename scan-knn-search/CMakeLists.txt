cmake_minimum_required(VERSION 3.5)
project(scan_knn_search)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")

set(JSON_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/../json/single_include)
include_directories(${JSON_INCLUDE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)

add_executable(scan_knn_search main.cpp)
