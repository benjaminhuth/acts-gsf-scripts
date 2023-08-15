cmake_minimum_required(VERSION 3.16)

project(gsf)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(benchmark/gsf_benchmark)
add_subdirectory(python/gsfanalysis/lib)
